"""Find the board in the frame and warp it flat."""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRACTION = 0.40


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # x - y is negative top-right
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _is_image_border(quad: np.ndarray, shape: tuple[int, ...], tol: float = 2.0) -> bool:
    """True if the quad is just the image rectangle.

    The Otsu fallback happily returns the whole frame when there is no board in
    it (a blank wall, a close-up).  Warping that is a no-op at best and a
    misleading "rectified" flag at worst, so it is not a board.
    """
    h, w = shape[:2]
    rect = order_corners(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32))
    return bool(np.abs(order_corners(quad) - rect).max() <= tol)


def _quads_from_edges(edges: np.ndarray, frame_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if cv2.contourArea(approx) < MIN_AREA_FRACTION * frame_area:
            continue
        found.append(approx.reshape(4, 2).astype(np.float32))
    found = [q for q in found if not _is_image_border(q, edges.shape)]
    return found


def find_board_quad(img: np.ndarray) -> np.ndarray | None:
    """Return the board's four corners (tl, tr, br, bl) or None if not found.

    Canny first; if that misses -- a low-contrast board edge against a pale
    wall is the usual cause -- retry on an Otsu split, where the board is one
    bright blob.
    """
    gray = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2GRAY)
    g8 = np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)
    g8 = cv2.GaussianBlur(g8, (5, 5), 0)
    frame_area = float(g8.shape[0] * g8.shape[1])

    candidates = _quads_from_edges(cv2.Canny(g8, 40, 120), frame_area)
    if not candidates:
        _, otsu = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        candidates = _quads_from_edges(otsu, frame_area)
    if not candidates:
        return None

    best = max(candidates, key=cv2.contourArea)
    return order_corners(best)


def warp_to_quad(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the given quad to a frontal rectangle."""
    tl, tr, br, bl = order_corners(corners)
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    out = cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))


def rectify(
    img: np.ndarray,
    corners: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Rectify the board.  Returns ``(image, corners_used)``.

    With no quad found the photo is passed through unchanged and ``corners`` is
    None; the caller warns.  A slightly skewed sticker still beats no sticker.
    """
    if corners is None:
        corners = find_board_quad(img)
    if corners is None:
        return img, None
    corners = order_corners(corners)
    return warp_to_quad(img, corners), corners


def parse_corners(text: str) -> np.ndarray:
    """Parse ``x1,y1,x2,y2,x3,y3,x4,y4`` into an ordered corner array."""
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 8:
        raise ValueError("--corners needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return order_corners(np.array([float(p) for p in parts], dtype=np.float32).reshape(4, 2))
