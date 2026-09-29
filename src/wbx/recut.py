"""Re-cut hand-made crops from the full-resolution photos they were taken from.

Segmenting a board by hand is the reliable way to decide what counts as one
quote -- a person does in a second what merge radii argue about.  The cost is
that the crops are usually screenshots: the *decision* is perfect, the pixels
are whatever the screen was showing.

Both can be had.  Each crop is located inside the original photo by feature
matching, the crop's own outline is warped into the photo's coordinates, and
that region is cut again at the camera's resolution.  The result is the same
segmentation with three or four times the linear detail, and it goes through the
ordinary pre-cropped path from there.

Matching is on SIFT features with a RANSAC homography rather than correlation:
a whiteboard is mostly blank, so correlation scores are dominated by the blank
and the right photo wins by nothing.  Inlier counts separate cleanly -- on a
56-photo shoot the true source scored hundreds against single digits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import load_image, load_rgba, save_rgba

PHOTO_SCALE = 0.5
"""Photos are matched downscaled; SIFT has ample detail and it is 4x less work."""

MIN_INLIERS = 25
"""Below this the match is not trusted and the crop is reported unmatched."""

RATIO = 0.75
"""Lowe's ratio test."""

MIN_REGION_PIXELS = 1000


@dataclass
class Match:
    crop: Path
    photo: Path | None
    inliers: int
    written: Path | None = None
    width: int = 0
    height: int = 0
    gain: float = 0.0


def _features(gray: np.ndarray, sift):
    return sift.detectAndCompute(gray, None)


def index_photos(photos: list[Path], sift=None, scale: float = PHOTO_SCALE):
    """Detect features once per photo; every crop is matched against this."""
    sift = sift or cv2.SIFT_create(nfeatures=4000)
    out = []
    for path in photos:
        gray = cv2.cvtColor((load_image(path) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        k, d = _features(gray, sift)
        out.append((Path(path), k, d))
    return out


def locate(crop_rgb: np.ndarray, crop_mask: np.ndarray, index, sift=None, matcher=None):
    """Best (inliers, photo, homography) taking the crop into photo coordinates."""
    sift = sift or cv2.SIFT_create(nfeatures=4000)
    matcher = matcher or cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    gray = cv2.cvtColor((crop_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kt, dt = _features(gray, sift)
    if dt is None or len(kt) < 8:
        return 0, None, None

    best = (0, None, None)
    for path, kq, dq in index:
        if dq is None or len(kq) < 8:
            continue
        pairs = matcher.knnMatch(dt, dq, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([kt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        n = int(inliers.sum()) if inliers is not None else 0
        if H is not None and n > best[0]:
            best = (n, path, H)
    return best


def recut(crops: list[Path], photos: list[Path], dest: Path,
          scale: float = PHOTO_SCALE, min_inliers: int = MIN_INLIERS) -> list[Match]:
    """Write a full-resolution RGBA cut-out for every crop that can be located."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    index = index_photos(photos, sift, scale)

    results: list[Match] = []
    for crop_path in crops:
        crop_path = Path(crop_path)
        rgb, mask = load_rgba(crop_path)
        if mask is None:
            mask = np.ones(rgb.shape[:2], bool)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            results.append(Match(crop_path, None, 0))
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub_rgb, sub_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]

        n, photo_path, H = locate(sub_rgb, sub_mask, index, sift, matcher)
        if n < min_inliers or H is None:
            results.append(Match(crop_path, None, n))
            continue

        photo = load_image(photo_path)
        ph, pw = photo.shape[:2]
        full = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ H  # matched small, cut full size
        warped = cv2.warpPerspective(sub_mask.astype(np.uint8), full, (pw, ph),
                                     flags=cv2.INTER_NEAREST)
        wy, wx = np.nonzero(warped)
        if wy.size < MIN_REGION_PIXELS:
            results.append(Match(crop_path, None, n))
            continue

        by0, by1 = int(wy.min()), int(wy.max()) + 1
        bx0, bx1 = int(wx.min()), int(wx.max()) + 1
        region = photo[by0:by1, bx0:bx1].copy()
        keep = warped[by0:by1, bx0:bx1].astype(bool)
        region[~keep] = 0.0  # outside the outline is not board; load_rgba masks it back out

        written = dest / f"{crop_path.stem}.png"
        save_rgba(written, region, keep.astype(np.float32))
        results.append(Match(crop_path, Path(photo_path), n, written,
                             bx1 - bx0, by1 - by0, round((bx1 - bx0) / max(1, x1 - x0), 2)))
    return results
