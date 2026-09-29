"""Flatten an extract run into one folder of stickers, ready to hand over.

``wbx extract`` writes per-photo directories, because that is what tracing a
sticker back to its photo needs.  Nobody wants to receive that.  This copies
the stickers out into a flat, sequentially numbered folder with an index that
carries the provenance the directory names were doing.

The PNG bytes are copied, never re-encoded: a handover folder that differs from
what was reviewed is worse than no folder at all.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DUPLICATE_CORRELATION = 0.45
"""Silhouette correlation above which two stickers are flagged as the same quote.

Measured on a 59-crop shoot where eight quotes had been lassoed twice: those
pairs scored 0.58 to 1.00 and the closest unrelated pair scored 0.34, so the
threshold sits in the gap rather than on either cluster.  It is a hint for
whoever culls the folder, not a decision -- nothing is ever dropped.
"""

DUPLICATE_INK_DISTANCE = 0.35
"""RGB distance above which two stickers are different markers, whatever their shape.

Silhouette alone is not enough: two quotes printed in the same hand at the same
size correlate strongly even when they say different things.  Two crops of *one*
quote, though, are the same ink -- on that shoot they sat within 0.24 of each
other while a shape-alike pair of genuinely different quotes was 0.80 apart.
So shape proposes and colour vetoes.
"""

SIGNATURE_SIZE = (48, 32)


@dataclass
class Sticker:
    file: str
    source: str
    width: int
    height: int
    ink: str
    hue: float | None
    looks_like: str = ""


def collect(out_dir: Path) -> list[tuple[Path, dict, str]]:
    """Every exported sticker under OUT, in photo order, with its manifest row."""
    out_dir = Path(out_dir)
    found: list[tuple[Path, dict, str]] = []
    for manifest_path in sorted(out_dir.glob("*/manifest.json")) + sorted(out_dir.glob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for quote in manifest["quotes"]:
            path = manifest_path.parent / quote["file"]
            if path.exists():
                found.append((path, quote, manifest.get("source", manifest_path.parent.name)))
    return found


def _signature(path: Path) -> np.ndarray | None:
    """A z-scored thumbnail of the sticker's silhouette, cropped to the ink.

    Shape only, and scale-free: the same quote lassoed twice gives two crops
    with different margins, which is exactly the difference to ignore.
    """
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[-1] != 4:
        return None
    ys, xs = np.nonzero(rgba[..., 3] > 30)
    if ys.size == 0:
        return None
    tight = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1, 3].astype(np.float32) / 255.0
    thumb = cv2.resize(tight, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    return (thumb - thumb.mean()) / (thumb.std() + 1e-6)


def _ink_mean(path: Path) -> np.ndarray | None:
    """Mean colour of the solid part of the stroke, ignoring the soft rim."""
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[-1] != 4:
        return None
    a = rgba[..., 3].astype(np.float64) / 255.0
    core = a > 0.5
    if not core.any():
        return None
    rgb = rgba[..., [2, 1, 0]].astype(np.float64) / 255.0  # BGRA -> RGB
    w = a[core]
    return (rgb[core] * w[:, None]).sum(axis=0) / w.sum()


def find_duplicates(paths: list[Path]) -> dict[int, int]:
    """Map each sticker index to the earlier index it appears to duplicate.

    A match needs both a matching silhouette and matching ink; see the two
    thresholds above for why neither is enough on its own.  Each sticker points
    at most at one earlier sticker, and one already flagged is never used as a
    target, so a quote cropped three times chains back to the first copy.
    """
    sigs = [_signature(p) for p in paths]
    inks = [_ink_mean(p) for p in paths]
    dupe: dict[int, int] = {}
    for i in range(len(sigs)):
        if sigs[i] is None or inks[i] is None:
            continue
        for j in range(i):
            if sigs[j] is None or inks[j] is None or j in dupe:
                continue
            if float((sigs[i] * sigs[j]).mean()) < DUPLICATE_CORRELATION:
                continue
            if float(np.linalg.norm(inks[i] - inks[j])) <= DUPLICATE_INK_DISTANCE:
                dupe[i] = j
                break
    return dupe


def build_album(out_dir: Path, dest: Path, prefix: str = "sticker") -> list[Sticker]:
    """Copy every sticker under OUT into ``dest`` and write ``index.csv``."""
    found = collect(Path(out_dir))
    if not found:
        raise FileNotFoundError(f"no stickers found under {out_dir}")

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(found))))
    dupes = find_duplicates([p for p, _q, _s in found])

    rows: list[Sticker] = []
    for i, (path, quote, source) in enumerate(found):
        name = f"{prefix}_{i + 1:0{width}d}.png"
        shutil.copyfile(path, dest / name)
        rows.append(Sticker(name, source, int(quote["w"]), int(quote["h"]),
                            quote["color"], quote["dominant_hue"]))
    for i, j in dupes.items():
        rows[i].looks_like = rows[j].file

    with (dest / "index.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sticker", "source photo", "width", "height", "ink", "hue", "looks like"])
        for r in rows:
            writer.writerow([r.file, r.source, r.width, r.height, r.ink,
                             "" if r.hue is None else r.hue, r.looks_like])
    return rows
