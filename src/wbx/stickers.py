"""Group strokes into quotes, crop them, and write the manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .io import save_rgba
from .matte import chroma, hue


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
    merge_radius: int = 30,
    min_quote_area: int = 800,
    pad: int = 12,
):
    """Merge nearby strokes into quotes.

    Dilating by ``merge_radius`` closes the gaps between the letters and words
    of one quote so they land in a single component, while the hand-width gap
    we ask writers to leave between quotes keeps neighbours apart.  The kept-or-
    dropped decision uses the *undilated* stroke area, so a lone speck that
    dilation inflated into a large blob still fails ``min_quote_area``.

    Returns ``(quotes, labels, mask)`` where quotes are ``(label, x, y, w, h,
    area)`` in reading order.
    """
    mask = alpha > 0
    if not mask.any():
        return [], np.zeros(alpha.shape, np.int32), mask

    r = max(0, int(merge_radius))
    if r > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        dil = cv2.dilate(mask.astype(np.uint8), kernel)
    else:
        dil = mask.astype(np.uint8)

    n, labels = cv2.connectedComponents(dil, connectivity=8)
    y0, x0, y1, x1, area = _label_bboxes(labels, n, mask)

    h, w = alpha.shape
    out = []
    for k in range(1, n):
        if area[k] < int(min_quote_area):
            continue
        ax0 = max(0, int(x0[k]) - pad)
        ay0 = max(0, int(y0[k]) - pad)
        ax1 = min(w, int(x1[k]) + 1 + pad)
        ay1 = min(h, int(y1[k]) + 1 + pad)
        out.append((k, ax0, ay0, ax1 - ax0, ay1 - ay0, int(area[k])))

    out.sort(key=lambda q: (q[2], q[1]))  # reading order: top to bottom, then left to right
    return out, labels, mask


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
