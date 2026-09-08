"""Recover true ink colour from translucent marker over a white board.

Each pixel of a normalized photo is ``observed = a*ink + (1-a)*white``.  With
``a`` from the matte, that inverts to ``ink = 1 - (1-observed)/a``.

The reason this is worth doing rather than just copying dark pixels out: the
recovered colour always lies on the ray from white through the true ink colour,
so the hue is preserved exactly even when ``a`` is misestimated.  A wrong ``a``
costs some saturation; it never turns a pastel into grey.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-3


def unmix(norm: np.ndarray, alpha: np.ndarray, boost_sat: float = 0.0) -> np.ndarray:
    """Solve for straight (non-premultiplied) ink colour."""
    a = np.maximum(alpha, EPS)[..., None]
    ink = np.clip(1.0 - (1.0 - norm) / a, 0.0, 1.0)
    ink = np.where((alpha > EPS)[..., None], ink, 0.0).astype(np.float32)

    if boost_sat > 0.0:
        hsv = cv2.cvtColor(np.ascontiguousarray(ink), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + float(boost_sat)), 0.0, 1.0)
        ink = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return np.ascontiguousarray(ink.astype(np.float32))
