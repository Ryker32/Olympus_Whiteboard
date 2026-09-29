"""A whiteboard photo generated in code, so the pipeline can be tested honestly
without waiting for shoot day.

The synthesis follows the same physical model the extractor inverts: translucent
ink composited over an unevenly lit board.  It is deliberately unkind -- a
lighting gradient, a vignette, a glare blob, sensor noise -- so that a change
that only works on a flat scan fails here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HEIGHT = 1600, 1000

# name, RGB in 0-255, peak coverage, word, (x, y) baseline
STROKE_SPECS = [
    ("black", (20, 20, 20), 0.90, "SENIORS", (90, 220)),
    ("yellow", (250, 215, 60), 0.70, "SUNSET", (640, 220)),
    ("blue", (120, 175, 235), 0.65, "LAKE", (1180, 220)),
    ("red", (215, 45, 55), 0.85, "PROM", (110, 720)),
    ("green", (40, 150, 80), 0.80, "FIELD", (620, 720)),
    ("purple", (140, 80, 190), 0.75, "LAST", (1180, 720)),
]

ACHROMATIC = {"black"}


@dataclass
class Stroke:
    name: str
    ink: np.ndarray       # float32 RGB in [0, 1]
    coverage: np.ndarray  # float32 in [0, 1]


def board_lighting(h: int = HEIGHT, w: int = WIDTH, seed: int = 0) -> np.ndarray:
    """A bare board: horizontal gradient 235->255, a vignette, one glare blob."""
    xs = np.linspace(235.0, 255.0, w, dtype=np.float32)
    board = np.repeat(xs[None, :], h, axis=0) / 255.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    board *= 1.0 - 0.10 * np.clip(r, 0, 1.4) ** 2

    glare = np.exp(-(((xx - 0.72 * w) ** 2 + (yy - 0.30 * h) ** 2) / (2 * (0.13 * w) ** 2)))
    board = np.clip(board + 0.11 * glare, 0, 1)

    rgb = np.dstack([board * 1.00, board * 0.995, board * 0.975])  # faint warm cast
    return np.ascontiguousarray(rgb.astype(np.float32))


def _word_coverage(word: str, org: tuple[int, int], peak: float,
                   h: int = HEIGHT, w: int = WIDTH) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(canvas, word, org, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 10, cv2.LINE_AA)
    cov = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 1.6)  # marker edges are soft
    return np.clip(cov, 0, 1) * float(peak)


def make_board(seed: int = 0, noise: float = 1.5 / 255.0):
    """Return ``(photo, strokes)`` -- a synthetic board photo and ground truth."""
    photo = board_lighting()
    strokes: list[Stroke] = []

    for name, rgb, peak, word, org in STROKE_SPECS:
        cov = _word_coverage(word, org, peak)
        ink = np.array(rgb, dtype=np.float32) / 255.0
        photo = cov[..., None] * ink + (1.0 - cov[..., None]) * photo
        strokes.append(Stroke(name, ink, cov))

    if noise:
        rng = np.random.default_rng(seed)
        photo = photo + rng.normal(0.0, noise, photo.shape).astype(np.float32)

    return np.ascontiguousarray(np.clip(photo, 0, 1).astype(np.float32)), strokes


def add_specks(photo: np.ndarray, seed: int = 1, count: int = 40) -> np.ndarray:
    """Dust and board scuffs: dark dots well under ``min_area``."""
    out = photo.copy()
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = int(rng.integers(40, WIDTH - 40))
        y = int(rng.integers(40, HEIGHT - 40))
        cv2.circle(out, (x, y), int(rng.integers(1, 4)), (0.25, 0.25, 0.25), -1, cv2.LINE_AA)
    return out


def add_ghost(photo: np.ndarray, org: tuple[int, int] = (150, 490),
              strength: float = 0.55, grey: float = 0.35):
    """A half-wiped smear: soft-edged, colourless, never as solid as real ink.

    Returns ``(photo, coverage)`` so tests can point at exactly where it went.
    """
    canvas = np.zeros(photo.shape[:2], dtype=np.uint8)
    cv2.putText(canvas, "OLD NOTES", org, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 255, 9, cv2.LINE_AA)
    cov = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 5.0) * float(strength)
    tone = np.float32([grey, grey, grey])
    out = np.clip(cov[..., None] * tone + (1 - cov[..., None]) * photo, 0, 1).astype(np.float32)
    return out, cov


def add_fat_word(photo: np.ndarray, word: str = "BOLD", org: tuple[int, int] = (150, 480),
                 rgb: tuple[int, int, int] = (30, 90, 200), peak: float = 0.85,
                 scale: float = 3.0, grow: int = 3):
    """A chisel-tip marker: strokes wide enough to poison their own background.

    Width is set by dilating the glyphs rather than by ``putText``'s thickness
    argument, which this OpenCV build ignores past a few pixels.  The default
    ``grow`` puts the strokes at roughly 10px median / 18px peak width on the
    1600px synthetic board, which is where the real photos sit and where a
    41px median blur starts landing on ink instead of on board.

    Returns ``(photo, coverage)``.
    """
    canvas = np.zeros(photo.shape[:2], dtype=np.uint8)
    cv2.putText(canvas, word, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 8, cv2.LINE_AA)
    if grow:
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    cov = np.clip(cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 1.6), 0, 1) * peak
    ink = np.float32(rgb) / 255.0
    out = np.clip(cov[..., None] * ink + (1 - cov[..., None]) * photo, 0, 1).astype(np.float32)
    return np.ascontiguousarray(out), cov


def close_up(subject: np.ndarray, intruders: list[tuple[np.ndarray, int, int]],
             size: tuple[int, int] = (1600, 2200), seed: int = 0):
    """One quote photographed on its own, with neighbours running off the edges.

    What a board shot one message at a time actually looks like: the subject
    centred and whole, and whatever was written beside it clipped by the frame.
    ``subject`` and each intruder are ``(coverage, ink)`` pairs as produced by
    :func:`word`; intruders carry their own top-left corner, which may be
    negative or past the edge.  Returns ``(photo, (x, y))`` of the subject.
    """
    h, w = size
    photo = cv2.resize(board_lighting(), (w, h), interpolation=cv2.INTER_LINEAR)

    def paste(cov, ink, x, y):
        ch, cw = cov.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + cw), min(h, y + ch)
        if x1 <= x0 or y1 <= y0:
            return
        c = cov[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
        photo[y0:y1, x0:x1] = c * ink + (1 - c) * photo[y0:y1, x0:x1]

    cov, ink = subject
    sx, sy = (w - cov.shape[1]) // 2, (h - cov.shape[0]) // 2
    paste(cov, ink, sx, sy)
    for icov, ix, iy in intruders:
        paste(icov[0], icov[1], ix, iy)

    rng = np.random.default_rng(seed)
    photo = photo + rng.normal(0.0, 1.5 / 255.0, photo.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(photo, 0, 1).astype(np.float32)), (sx, sy)


def word(text: str, rgb: tuple[int, int, int] = (30, 90, 200), peak: float = 0.85,
         scale: float = 2.2, thickness: int = 7, pad: int = 30):
    """A word on its own transparent canvas, as ``(coverage, ink)``."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    canvas = np.zeros((th + base + 2 * pad, tw + 2 * pad), dtype=np.uint8)
    cv2.putText(canvas, text, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX, scale, 255,
                thickness, cv2.LINE_AA)
    cov = np.clip(cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 1.6), 0, 1) * peak
    return cov, np.float32(rgb) / 255.0


def project_corners(w: int = WIDTH, h: int = HEIGHT, angle_deg: float = 25.0,
                    focal: float = 2200.0, distance: float = 2600.0,
                    canvas: tuple[int, int] = (1250, 1800)) -> np.ndarray:
    """Where the board's corners land when it is yawed ``angle_deg`` from frontal."""
    th = np.deg2rad(angle_deg)
    ch, cw = canvas
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # tl, tr, br, bl
        X, Y = sx * w / 2.0, sy * h / 2.0
        z = distance - X * np.sin(th)
        pts.append((focal * X * np.cos(th) / z + cw / 2.0, focal * Y / z + ch / 2.0))
    return np.array(pts, dtype=np.float32)


def place_on_wall(board: np.ndarray, angle_deg: float = 25.0,
                  canvas: tuple[int, int] = (1250, 1800)):
    """Composite the board onto a darker wall at an angle.  Returns ``(frame, corners)``."""
    h, w = board.shape[:2]
    ch, cw = canvas
    corners = project_corners(w, h, angle_deg, canvas=canvas)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, corners)

    warped = cv2.warpPerspective(board, m, (cw, ch), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), m, (cw, ch), flags=cv2.INTER_NEAREST)

    yy = np.linspace(0.52, 0.44, ch, dtype=np.float32)[:, None]
    wall = np.dstack([np.repeat(yy, cw, axis=1)] * 3) * np.float32([1.0, 0.97, 0.92])

    frame = warped * mask[..., None] + wall * (1 - mask[..., None])
    return np.ascontiguousarray(np.clip(frame, 0, 1).astype(np.float32)), corners


def lasso_crop(photo: np.ndarray, box: tuple[int, int, int, int],
               frame: float = 0.0, seed: int = 3):
    """Cut one quote out of ``photo`` the way a lasso tool does.

    The loop is hand-drawn, so it is irregular rather than rectangular, and the
    pixels outside it are cleared to pure black under alpha 0 -- which is what
    image editors write, and why an alpha-blind load sees a huge black blob.

    ``frame`` paints a dark bar of that tone along the bottom inside the loop:
    the board's own frame or the marker tray, which a loop drawn around a quote
    near the edge of the board catches along with it.

    Returns ``(rgb, alpha)`` ready for :func:`wbx.io.save_rgba`.
    """
    x0, y0, x1, y1 = box
    sub = np.ascontiguousarray(photo[y0:y1, x0:x1]).copy()
    h, w = sub.shape[:2]

    rng = np.random.default_rng(seed)
    wobble = 0.06 * min(h, w)
    pts = []
    for t in np.linspace(0.0, 1.0, 40, endpoint=False):  # once around the perimeter
        edge = t * 4.0
        side, u = int(edge), edge % 1.0
        x, y = [(u * w, 0.0), (float(w), u * h), ((1 - u) * w, float(h)), (0.0, (1 - u) * h)][side]
        inward = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)][side]
        d = rng.uniform(0.2, 1.0) * wobble
        pts.append((x + inward[0] * d, y + inward[1] * d))

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)

    if frame > 0:
        band = max(2, int(round(frame * min(h, w))))
        sub[h - band:] = 0.12  # the dark board frame, as solid as any stroke

    alpha = mask.astype(np.float32)
    sub[mask == 0] = 0.0  # editors clear the RGB under transparent pixels
    return np.ascontiguousarray(sub), np.ascontiguousarray(alpha)
