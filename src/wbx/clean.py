"""Component-level cleanup: dust, and the grey ghosts of half-erased writing."""

from __future__ import annotations

import cv2
import numpy as np

from .matte import saturation


def _component_stats(labels: np.ndarray, n: int, alpha: np.ndarray, sat: np.ndarray):
    """Per-label peak alpha and alpha-weighted mean saturation.

    The saturation mean is alpha-weighted rather than flat: a component's
    faintest pixels are its noisiest, and letting them vote equally with the
    stroke core makes the ghost test fire on real writing.
    """
    flat = labels.ravel()
    sel = flat > 0
    lab = flat[sel]
    a = alpha.ravel()[sel]
    s = sat.ravel()[sel]

    amax = np.zeros(n, dtype=np.float32)
    np.maximum.at(amax, lab, a)
    asum = np.bincount(lab, weights=a, minlength=n)
    wsat = np.bincount(lab, weights=a * s, minlength=n)
    mean_sat = wsat / np.maximum(asum, 1e-6)
    return amax, mean_sat.astype(np.float32)


def clean(
    alpha: np.ndarray,
    ink: np.ndarray,
    observed: np.ndarray,
    min_area: int = 150,
    ghost_max_alpha: float = 0.35,
    ghost_max_sat: float = 0.10,
    alpha_floor: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop specks and eraser ghosts, returning cleaned ``(alpha, ink)``.

    Two rules, both applied to whole connected components:

    * area below ``min_area`` -- dust, board scuffs, JPEG mosquitoes;
    * peak alpha below ``ghost_max_alpha`` **and** mean saturation below
      ``ghost_max_sat`` -- a smear with no solid core and no colour, i.e. what
      a wipe leaves behind.  A genuinely faint stroke still carries its hue, so
      it survives.

    The saturation is measured on ``observed`` -- the normalized photo -- and
    not on the recovered ink.  Dividing by a small alpha amplifies sensor noise
    into vivid nonsense, so every ghost looks brilliantly coloured after
    unmixing; in the observation it stays the grey it actually is.

    Deliberately not done per pixel: a low-alpha low-saturation pixel test would
    eat the antialiased rim of every black letter and give the text a hard,
    aliased edge.
    """
    a = alpha.copy()
    a[a < float(alpha_floor)] = 0.0
    mask = (a > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return a, np.zeros_like(ink)

    amax, mean_sat = _component_stats(labels, n, a, saturation(observed))
    area = stats[:, cv2.CC_STAT_AREA]

    drop = (area < int(min_area)) | ((amax < float(ghost_max_alpha)) & (mean_sat < float(ghost_max_sat)))
    drop[0] = True  # label 0 is bare board

    killed = drop[labels]
    a[killed] = 0.0
    ink = ink.copy()
    ink[killed] = 0.0
    return a, ink
