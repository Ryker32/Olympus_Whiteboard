"""Find the board face in a photo that shows more than the board.

:mod:`wbx.rectify` handles the photo that frames the whole board: four corners,
a perspective warp, done.  It deliberately declines when a corner runs off the
picture -- there is nothing to map that corner to -- which is exactly what
happens when someone walks along a long board photographing it a stretch at a
time.  Those frames still need the wall, the aluminium trim and whatever is
visible past the end of the board taken out, because all of it is darker than
the board and mattes as confidently as ink.  Worse, it is *connected*: one strip
of frame running the width of the photo bridges every quote on the board into a
single blob.

So: no quad, just the region.  The board is the large, bright, colourless area;
the writing on it is a hole in that area, and gets filled back in.
"""

from __future__ import annotations

import cv2
import numpy as np

from .matte import chroma

WORK_MAX = 900
"""Detection runs downscaled; the board's outline does not need fine detail."""

BRIGHT_FRACTION = 0.62
"""Of the board's own brightness.  Shadowed board still clears this; trim does not."""

MAX_CHROMA = 0.22
"""A whiteboard is colourless even when the room light is warm."""

MIN_COVERAGE = 0.25
"""Below this the detection is not believable and the whole frame is used."""


def fill_interior(mask: np.ndarray) -> np.ndarray:
    """Add back every gap in ``mask`` that does not reach the frame edge.

    Writing, glare, magnets and tape are holes punched in the board region by
    the brightness test.  A hole that touches the edge of the picture is not a
    hole, it is the world outside the board.
    """
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape[:2]
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            out |= labels == i
    return out


def board_region(img: np.ndarray, work_max: int = WORK_MAX) -> np.ndarray | None:
    """Boolean mask of the board face, or None if nothing convincing is found.

    Returning None rather than a guess matters: a wrong mask deletes writing,
    while no mask only leaves the pipeline where it already was.
    """
    h, w = img.shape[:2]
    scale = min(1.0, work_max / float(max(h, w)))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)

    lum = small.max(axis=2)
    level = float(np.percentile(lum, 90))
    candidate = (lum > BRIGHT_FRACTION * level) & (chroma(small) < MAX_CHROMA)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    mask = fill_interior(labels == biggest)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if mask.mean() < MIN_COVERAGE:
        return None
    full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return None if full.all() else full
