"""Background estimation and division.

The photo is modelled as ink composited over a board that is itself unevenly
lit.  Estimating that lighting per channel and dividing it out gives an image
where the bare board reads as 1.0 everywhere, which is the precondition for the
unmixing math in :mod:`wbx.unmix`.  Dividing per channel (rather than on
luminance) is also what removes the warm/cool cast of room light.
"""

from __future__ import annotations

import cv2
import numpy as np


def _work_size(shape: tuple[int, ...], work_max: int) -> tuple[int, int]:
    h, w = shape[:2]
    scale = min(1.0, work_max / float(max(h, w)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def estimate_background(
    img: np.ndarray,
    work_max: int = 1200,
    median_k: int = 41,
    method: str = "median",
    close_radius: int = 6,
    blur_sigma: float = 20.0,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the per-channel board illumination behind the strokes.

    ``method="median"`` is the default: a large median blur on a downscaled copy
    rejects strokes as long as they are thin relative to the kernel.
    ``method="close"`` is the fallback for boards written on with a fat marker,
    where strokes start bleeding into the median; a morphological close wipes
    dark structure first, then a wide Gaussian smooths what is left.

    ``exclude`` is an optional boolean mask (True = do not sample here, e.g.
    people in the frame); those pixels are filled with the median of the rest
    before blurring so they cannot drag the estimate down.
    """
    w, h = _work_size(img.shape, work_max)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if exclude is not None and exclude.any():
        ex = cv2.resize(exclude.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not ex.all():
            fill = np.median(small[~ex], axis=0)
            small = small.copy()
            small[ex] = fill

    u8 = np.clip(small * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if method == "median":
        k = int(median_k) | 1
        k = max(3, min(k, (min(h, w) - 1) | 1))
        bg_small = cv2.medianBlur(u8, k).astype(np.float32) / 255.0
    elif method == "close":
        r = max(1, int(close_radius))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(closed.astype(np.float32) / 255.0, (0, 0), blur_sigma)
    else:
        raise ValueError(f"unknown background method: {method!r}")

    bg = cv2.resize(bg_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bg.astype(np.float32))


def normalize(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Divide the illumination out so the bare board sits at 1.0."""
    norm = img / np.maximum(bg, 1e-3)
    return np.ascontiguousarray(np.clip(norm, 0.0, 1.0).astype(np.float32))
