"""QA contact sheet.

Every sticker is shown twice: over a light checkerboard and over a dark one.
The dark half is the point.  Naive extraction leaves a grey fringe of
half-transparent board colour around every stroke, which is invisible on white
and screams on black; correct unmixing gives a clean coloured edge.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io import save_rgb

TILE_W, TILE_H = 260, 180
LABEL_H = 22
COLUMNS = 3
GAP = 10
CHECK = 16


def checkerboard(h: int, w: int, a: float, b: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // CHECK) + (xx // CHECK)) % 2 == 0, a, b).astype(np.float32)
    return np.dstack([board] * 3)


def _fit(rgba: np.ndarray, tw: int, th: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgba.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = small[..., :3].astype(np.float32) / 255.0
    alpha = small[..., 3].astype(np.float32) / 255.0
    return rgb, alpha


def _tile(rgba: np.ndarray, light: bool) -> np.ndarray:
    bg = checkerboard(TILE_H, TILE_W, 1.0, 0.92) if light else checkerboard(TILE_H, TILE_W, 0.10, 0.18)
    rgb, alpha = _fit(rgba, TILE_W - 8, TILE_H - 8)
    h, w = alpha.shape
    y = (TILE_H - h) // 2
    x = (TILE_W - w) // 2
    region = bg[y : y + h, x : x + w]
    bg[y : y + h, x : x + w] = rgb * alpha[..., None] + region * (1.0 - alpha[..., None])
    return bg


def find_stickers(out_dir: Path) -> list[Path]:
    """All quote PNGs under OUT, including per-photo subdirectories."""
    files = sorted(out_dir.glob("quote_*.png")) + sorted(out_dir.glob("*/quote_*.png"))
    return files


def build_sheet(out_dir: Path, dest: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    files = find_stickers(out_dir)
    if not files:
        raise FileNotFoundError(f"no quote_*.png found under {out_dir}")

    cell_w = TILE_W * 2 + GAP
    cell_h = TILE_H + LABEL_H
    rows = (len(files) + COLUMNS - 1) // COLUMNS
    sheet = np.full(
        (rows * (cell_h + GAP) + GAP, COLUMNS * (cell_w + GAP) + GAP, 3),
        0.55,
        dtype=np.float32,
    )

    labels: list[tuple[str, int, int]] = []
    for i, path in enumerate(files):
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            continue
        if rgba.shape[2] == 4:
            rgba = rgba[..., [2, 1, 0, 3]]  # BGRA -> RGBA
        else:  # pragma: no cover - stickers are always RGBA
            rgba = np.dstack([rgba[..., ::-1], np.full(rgba.shape[:2], 255, np.uint8)])

        r, c = divmod(i, COLUMNS)
        y = GAP + r * (cell_h + GAP)
        x = GAP + c * (cell_w + GAP)
        sheet[y : y + TILE_H, x : x + TILE_W] = _tile(rgba, light=True)
        sheet[y : y + TILE_H, x + TILE_W + GAP : x + TILE_W + GAP + TILE_W] = _tile(rgba, light=False)

        labels.append((str(path.relative_to(out_dir)), x + 2, y + TILE_H + 15))

    u8 = np.clip(sheet * 255.0 + 0.5, 0, 255).astype(np.uint8)  # OpenCV draws on 8-bit only
    for text, tx, ty in labels:
        cv2.putText(u8, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    dest = Path(dest) if dest else out_dir / "sheet.png"
    save_rgb(dest, u8.astype(np.float32) / 255.0)
    return dest
