"""Component-level cleanup: dust, and the grey ghosts of half-erased writing."""

from __future__ import annotations

import cv2
import numpy as np

from .matte import chroma, saturation


CROP_EDGE_BAND = 0.06
"""Boundary band width, as a fraction of the crop's shorter side."""

CROP_EDGE_SHARE = 0.75
"""Share of a component that must sit in that band for it to be crop litter."""

CROP_EDGE_MAX_CHROMA = 0.30
"""Above this chroma a boundary-hugging component is marker, not litter."""


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


def drop_crop_edge(
    alpha: np.ndarray,
    ink: np.ndarray,
    crop: np.ndarray,
    band: float = CROP_EDGE_BAND,
    share: float = CROP_EDGE_SHARE,
    max_chroma: float = CROP_EDGE_MAX_CHROMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop what a hand-drawn crop boundary clipped in along with the quote.

    Lassoing one quote out of a board photo tends to catch a sliver of the
    board's dark frame, a marker tray or a desk edge wherever the loop ran past
    the edge of the board.  Those are as dark and as solid as ink, so nothing
    upstream rejects them, and they export as a black bar across the sticker.

    Two conditions, and both are needed:

    * the component lies almost entirely in a band just inside the crop
      boundary -- it *runs along* the cut rather than merely touching it, which
      is what separates clipped furniture from a letter the lasso sliced
      through;
    * it is achromatic.  A clipped stroke of coloured marker also hugs the
      boundary, and on these crops several do; the chroma test is what keeps
      them.  Black writing is achromatic too, which is why the geometry has to
      carry the decision and this only guards it.

    The band scales with the crop, not with the photo it came from: it is what
    counts as "against the edge" for a loop drawn by hand around this quote.
    """
    # The edge of the picture bounds the crop as surely as the cut does, and for a
    # board photographed a stretch at a time it is the only bound there is: the mask
    # runs straight off the frame, so without this the board's own trim -- a hairline
    # the length of the photo -- reads as deep interior and survives to bridge every
    # quote into one blob.
    padded = np.pad(crop.astype(np.uint8), 1, constant_values=0)
    d = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    edge = d <= float(band) * min(alpha.shape[:2])

    n, labels, _stats, _c = cv2.connectedComponentsWithStats((alpha > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return alpha, ink

    flat = labels.ravel()
    sel = flat > 0
    lab = flat[sel]
    size = np.bincount(lab, minlength=n).astype(np.float64)
    in_band = np.bincount(lab, weights=edge.ravel()[sel].astype(np.float64), minlength=n)
    chr_sum = np.bincount(lab, weights=chroma(ink).ravel()[sel].astype(np.float64), minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        hugs = in_band / np.maximum(size, 1.0) >= float(share)
        colourless = chr_sum / np.maximum(size, 1.0) < float(max_chroma)
    drop = hugs & colourless
    drop[0] = False  # label 0 is bare board, already transparent

    killed = drop[labels]
    alpha = alpha.copy()
    ink = ink.copy()
    alpha[killed] = 0.0
    ink[killed] = 0.0
    return alpha, ink


SPECULAR_REACH = 0.6
"""Closing radius as a fraction of stroke width when looking for glare gaps."""

SPECULAR_EXCESS = 1.01
"""How much brighter than the estimated board a gap must be to be called glare."""


def fill_specular(
    alpha: np.ndarray,
    ink: np.ndarray,
    observed: np.ndarray,
    background: np.ndarray,
    reach: float = SPECULAR_REACH,
    excess: float = SPECULAR_EXCESS,
    solid: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Close the gaps a reflection burns through a stroke.

    A whiteboard is glossy, so a ceiling light reflects off it *through* the
    ink.  The model every other step relies on -- ``observed = a*ink + (1-a)*
    board`` -- has no term for light added on top, so a glared pixel is read as
    less ink rather than more light, and where the reflection is strong enough
    the coverage collapses to nothing.  The result is a white gash cut clean
    across a letter, which is far more disfiguring than the faintness it comes
    with: a reader forgives pale handwriting and does not forgive a hole.

    Nothing about the ink survives inside such a gash, so this does not try to
    recover it -- it interpolates across from the stroke on either side, which
    is the honest answer to "what was behind the reflection".

    Two conditions keep it off gaps that are really board:

    * the gap closes under a kernel narrower than the stroke, so the space
      between two strokes deliberately set apart is left alone;
    * the pixel is *brighter* than the estimated board underneath it.  Bare
      board divides out to 1.0 by construction; only added light exceeds it.
    """
    mask = alpha >= float(solid)
    if not mask.any():
        return alpha, ink

    dist = cv2.distanceTransform((alpha > 0).astype(np.uint8), cv2.DIST_L2, 5)
    inside = dist[dist > 0]
    if inside.size == 0:
        return alpha, ink
    width = 2.0 * float(np.median(inside))
    r = max(1, int(round(reach * width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    ratio = observed.max(axis=2) / np.maximum(background.max(axis=2), 1e-3)
    holes = closed & ~mask & (ratio >= float(excess))
    if not holes.any():
        return alpha, ink

    fill = holes.astype(np.uint8)
    a8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    i8 = np.clip(ink * 255.0 + 0.5, 0, 255).astype(np.uint8)
    a8 = cv2.inpaint(a8, fill, 3, cv2.INPAINT_TELEA)
    i8 = cv2.inpaint(i8, fill, 3, cv2.INPAINT_TELEA)

    alpha = alpha.copy()
    ink = ink.copy()
    alpha[holes] = (a8.astype(np.float32) / 255.0)[holes]
    ink[holes] = (i8.astype(np.float32) / 255.0)[holes]
    return alpha, np.ascontiguousarray(ink)

