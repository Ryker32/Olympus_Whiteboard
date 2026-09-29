"""Group strokes into quotes, crop them, and write the manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .io import save_rgba
from .matte import chroma, hue


TEXT_HEIGHT_SLACK = 0.7
"""Merge slack as a fraction of letter height.  See :func:`auto_merge_radius`."""

MERGE_RADIUS_BOUNDS = (10, 600)

ACHROMATIC_CHROMA = 0.12
"""Core chroma below which a quote counts as achromatic and reports no hue."""


@dataclass
class Quote:
    file: str
    x: int
    y: int
    w: int
    h: int
    area: int
    color: str
    dominant_hue: float | None


def estimate_stroke_width(mask: np.ndarray) -> float:
    """Median marker stroke width in pixels, via the distance transform."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    inside = dist[dist > 0]
    if inside.size == 0:
        return 0.0
    return float(2.0 * np.median(inside))


def text_height(stats: np.ndarray, min_area: int) -> float:
    """Letter height in pixels: the 75th percentile of component box heights.

    The median sits too low because most components are lowercase letters with
    no ascender; the 75th percentile lands near the real line height without
    being thrown by one tall bracket or a stray descender.
    """
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    heights = heights[areas >= int(min_area)]
    if heights.size == 0:
        return 0.0
    return float(np.percentile(heights, 75))


def auto_merge_radius(stats: np.ndarray, min_area: int) -> int:
    """Pick the merge slack from the handwriting itself.

    Scaled to letter height, not stroke width and not a fixed pixel count.  A
    fixed count cannot be right -- the same board at 2MP and 12MP needs
    different numbers -- and stroke width turned out to be a poor proxy: across
    real photos the slack actually needed ranged from 4.7 to 13.7 stroke widths,
    because a fat marker writing small and a fine marker writing large have the
    same stroke and completely different line spacing.  Letter height is what
    the gap between lines of one quote actually scales with.
    """
    height = text_height(stats, min_area)
    if height <= 0:
        return MERGE_RADIUS_BOUNDS[0]
    return int(np.clip(round(TEXT_HEIGHT_SLACK * height), *MERGE_RADIUS_BOUNDS))


def _group_boxes(boxes: np.ndarray, slack: int) -> np.ndarray:
    """Group boxes whose slack-inflated rectangles touch, transitively.

    Replaces dilating the whole mask with a huge kernel.  Beyond being far
    cheaper, it groups on the structure that actually defines a quote: the lines
    of one quote interleave vertically (ascenders and descenders overlap) while
    separate quotes are set apart in both axes.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    x0 = boxes[:, 0] - slack
    y0 = boxes[:, 1] - slack
    x1 = boxes[:, 0] + boxes[:, 2] + slack
    y1 = boxes[:, 1] + boxes[:, 3] + slack
    touch = (x0[:, None] <= x1[None, :]) & (x1[:, None] >= x0[None, :]) \
        & (y0[:, None] <= y1[None, :]) & (y1[:, None] >= y0[None, :])

    for i, j in np.argwhere(np.triu(touch, 1)):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _label_bboxes(labels: np.ndarray, n: int, mask: np.ndarray):
    """Per dilated-component bbox of the *undilated* stroke pixels it contains."""
    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]

    big = np.iinfo(np.int64).max
    y0 = np.full(n, big, dtype=np.int64)
    x0 = np.full(n, big, dtype=np.int64)
    y1 = np.full(n, -1, dtype=np.int64)
    x1 = np.full(n, -1, dtype=np.int64)
    np.minimum.at(y0, lab, ys)
    np.minimum.at(x0, lab, xs)
    np.maximum.at(y1, lab, ys)
    np.maximum.at(x1, lab, xs)
    area = np.bincount(lab, minlength=n)
    return y0, x0, y1, x1, area


def find_quotes(
    alpha: np.ndarray,
    merge_radius: int | None = None,
    min_quote_area: int = 800,
    pad: int = 12,
    min_area: int = 150,
    single: bool = False,
):
    """Merge nearby strokes into quotes.

    Every stroke component's bounding box is inflated by ``merge_radius`` and
    boxes that then touch are pooled into one quote, transitively.  That joins
    the letters, words and *lines* of one quote -- the hand-width gap we ask
    writers to leave between quotes is what keeps neighbours apart.  The
    kept-or-dropped decision uses real stroke area, so a speck whose inflated
    box is large still fails ``min_quote_area``.

    ``merge_radius`` of None means derive it from the letter height in this
    photo -- see :func:`auto_merge_radius`.

    ``single`` pools every stroke into one quote regardless of distance, for
    input that has already been cropped to a single quote by hand.  Splitting
    such a crop back up can only be wrong, and the gap that separates two
    quotes on a board is exactly the gap a wide-set signature leaves.

    Returns ``(quotes, labels, mask, merge_radius)`` where quotes are
    ``(label, x, y, w, h, area)`` in reading order and ``labels`` maps each
    pixel to its quote.
    """
    mask = alpha > 0
    empty = np.zeros(alpha.shape, np.int32)
    if not mask.any():
        return [], empty, mask, int(merge_radius or 0)

    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return [], empty, mask, int(merge_radius or 0)

    if single:
        r = 0
    elif merge_radius is None:
        r = auto_merge_radius(stats, min_area)
    else:
        r = max(0, int(merge_radius))

    boxes = stats[1:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
    areas = stats[1:, cv2.CC_STAT_AREA]
    groups = np.zeros(len(boxes), dtype=np.int64) if single else _group_boxes(boxes, r)

    # component label -> quote label, 0 meaning "not part of any kept quote"
    remap = np.zeros(n, dtype=np.int32)
    h, w = alpha.shape
    out = []
    for gid in np.unique(groups):
        members = np.flatnonzero(groups == gid)
        area = int(areas[members].sum())
        if area < int(min_quote_area):
            continue
        bx, by = boxes[members, 0], boxes[members, 1]
        bw, bh = boxes[members, 2], boxes[members, 3]
        x0 = max(0, int(bx.min()) - pad)
        y0 = max(0, int(by.min()) - pad)
        x1 = min(w, int((bx + bw).max()) + pad)
        y1 = min(h, int((by + bh).max()) + pad)
        label = len(out) + 1
        remap[members + 1] = label
        out.append((label, x0, y0, x1 - x0, y1 - y0, area))

    out.sort(key=lambda q: (q[2], q[1]))  # reading order: top to bottom, then left to right
    order = {q[0]: i + 1 for i, q in enumerate(out)}
    remap = np.array([order.get(int(v), 0) for v in remap], dtype=np.int32)
    out = [(order[q[0]],) + q[1:] for q in out]
    out.sort(key=lambda q: q[0])

    return out, remap[comp], mask, r


EDGE_MARGIN = 0.01
"""Strokes within this fraction of the short side of the frame count as clipped."""


def pick_main(quotes, labels: np.ndarray, shape, margin: float = EDGE_MARGIN):
    """Keep the one quote a close-up is of, dropping the neighbours at the edges.

    Photographing a board one message at a time still catches the corner of
    whatever is written next to it, and those intruders are indistinguishable
    from the subject by size or colour -- a neighbour's signature can outweigh a
    short quote.  What separates them is the frame: the subject was centred and
    photographed whole, the intruders run off the edge of the picture.

    So quotes clipped by the frame are dropped, and the largest of what remains
    is the subject.  If *everything* touches an edge -- a quote photographed
    tight, filling the frame -- the rule has nothing to say and the largest
    quote wins on area alone.

    Returns ``(quotes, labels)`` renumbered so the kept quote is label 1.
    """
    if not quotes:
        return quotes, labels

    h, w = shape[:2]
    tol = max(1.0, float(margin) * min(h, w))
    framed = [q for q in quotes
              if q[1] > tol and q[2] > tol and q[1] + q[3] < w - tol and q[2] + q[4] < h - tol]
    best = max(framed or quotes, key=lambda q: q[5])

    kept = (1,) + tuple(best[1:])
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[best[0]] = 1
    return [kept], remap[labels]


def bleed_color(ink: np.ndarray, alpha: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Extend ink colour a few pixels into the fully transparent surround.

    Invisible in a correct compositor, but layout tools that scale or blur a
    sticker interpolate RGB across the alpha edge; with black sitting under the
    transparent pixels that shows up as exactly the dark halo we are trying to
    avoid.  Only pixels at alpha == 0 are touched.
    """
    known = (alpha > 0).astype(np.uint8)
    out = ink.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(0, int(iterations))):
        grown = cv2.dilate(known, kernel)
        newly = (grown > 0) & (known == 0)
        if not newly.any():
            break
        blurred = cv2.blur(out * known[..., None].astype(np.float32), (3, 3))
        weight = cv2.blur(known.astype(np.float32), (3, 3))
        filled = blurred / np.maximum(weight, 1e-6)[..., None]
        out[newly] = filled[newly]
        known = grown
    return np.ascontiguousarray(out.astype(np.float32))


def summarize_color(ink: np.ndarray, alpha: np.ndarray) -> tuple[str, float | None]:
    """Alpha-weighted mean colour as hex, plus dominant hue in degrees.

    Hue is read off the solid core of the stroke only -- pixels at half the
    quote's peak coverage or better.  Thin edge pixels are where unmixing
    divides by a small alpha and turns sensor noise into vivid colour, so
    including them would invent a hue for black writing.  Genuinely achromatic
    writing reports None rather than a number read out of noise.
    """

    w = alpha.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return "#000000", None
    mean = (ink.astype(np.float64) * w[..., None]).reshape(-1, 3).sum(axis=0) / total
    r, g, b = (int(round(float(c) * 255)) for c in np.clip(mean, 0, 1))
    hex_color = f"#{r:02x}{g:02x}{b:02x}"

    core = alpha >= 0.5 * float(alpha.max())
    if not core.any():
        return hex_color, None
    chr_ = chroma(ink).astype(np.float64)
    cw = w[core]
    if (chr_[core] * cw).sum() / cw.sum() < ACHROMATIC_CHROMA:
        return hex_color, None
    hw = cw * chr_[core]
    theta = np.deg2rad(hue(ink)[core].astype(np.float64))
    mean_hue = float(np.rad2deg(np.arctan2((hw * np.sin(theta)).sum(), (hw * np.cos(theta)).sum())) % 360.0)
    return hex_color, round(mean_hue, 1)


def export(
    out_dir: Path,
    source: str,
    ink: np.ndarray,
    alpha: np.ndarray,
    quotes,
    labels: np.ndarray,
    rectified: bool,
    params: dict,
    bleed: bool = True,
) -> dict:
    """Write ``board.png``, ``quote_NN.png`` files and ``manifest.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    board_rgb = bleed_color(ink, alpha) if bleed else ink
    save_rgba(out_dir / "board.png", board_rgb, alpha)

    records: list[Quote] = []
    for i, (label, x, y, w, h, area) in enumerate(quotes, start=1):
        sub_alpha = alpha[y : y + h, x : x + w].copy()
        sub_ink = ink[y : y + h, x : x + w].copy()
        # a padded bbox can overlap a neighbouring quote; keep only our own strokes
        foreign = labels[y : y + h, x : x + w] != label
        sub_alpha[foreign] = 0.0
        sub_ink[foreign] = 0.0

        color, dom_hue = summarize_color(sub_ink, sub_alpha)
        name = f"quote_{i:02d}.png"
        save_rgba(out_dir / name, bleed_color(sub_ink, sub_alpha) if bleed else sub_ink, sub_alpha)
        records.append(Quote(name, int(x), int(y), int(w), int(h), int(area), color, dom_hue))

    manifest = {
        "source": source,
        "board": "board.png",
        "rectified": bool(rectified),
        "params": params,
        "quotes": [asdict(q) for q in records],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
