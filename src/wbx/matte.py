"""Soft alpha (ink coverage) estimation."""

from __future__ import annotations

import cv2
import numpy as np


def saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation in [0, 1] of a float32 RGB image."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    return np.ascontiguousarray(hsv[..., 1])


def chroma(rgb: np.ndarray) -> np.ndarray:
    """Colourfulness as ``max(R,G,B) - min(R,G,B)``.

    Preferred over HSV saturation whenever the pixel may be near black: HSV
    divides by the value, so ``(0, 0, 0.03)`` reads as fully saturated even
    though it is, to any eye, black.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.ascontiguousarray(rgb.max(axis=-1) - rgb.min(axis=-1))


def hue(rgb: np.ndarray) -> np.ndarray:
    """HSV hue in degrees [0, 360) of a float32 RGB image."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    return np.ascontiguousarray(hsv[..., 0])


def estimate_alpha(
    norm: np.ndarray,
    sat_weight: float = 0.7,
    alpha_floor: float = 0.06,
) -> np.ndarray:
    """Estimate per-pixel ink coverage from a normalized board image.

    Two cues, combined with a max:

    * darkness, ``1 - min(R, G, B)``, which catches anything that dims the
      board;
    * saturation, which catches the markers that barely darken it at all --
      yellow, orange, and the pale end of every other colour.  Without the
      saturation term a yellow quote comes out as a faint ghost.

    Coverage below ``alpha_floor`` is zeroed: it is sensor noise on bare board,
    and leaving it in would make the unmix divide by near-zero.
    """
    a_dark = 1.0 - norm.min(axis=2)
    a = np.maximum(a_dark, float(sat_weight) * saturation(norm))
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    a[a < float(alpha_floor)] = 0.0
    return a
