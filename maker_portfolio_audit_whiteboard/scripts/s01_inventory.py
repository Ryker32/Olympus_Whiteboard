"""Recover run-level metrics from the existing output tree (read-only).

Sources: out/cropped, out/hires (per-photo manifests + board.png), out/stickers,
out/stickers_hires (album index.csv), out/send (handover set).

Writes maker_portfolio_audit/data/inventory.json and per_crop_resolution.csv.

    python maker_portfolio_audit/scripts/s01_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
DATA = ROOT / "maker_portfolio_audit" / "data"


def index(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    inv: dict = {}

    crops = sorted((ROOT / "samples" / "cropped").glob("*.png"))
    photos = sorted((ROOT / "photos").glob("*.jpg"))
    inv["n_photos"] = len(photos)
    inv["n_crops"] = len(crops)
    inv["photo_size"] = sorted({Image.open(p).size for p in photos})

    # per crop: screenshot board.png size vs re-cut board.png size (board.png is full input size)
    rows = []
    for c in crops:
        lo = OUT / "cropped" / c.stem
        hi = OUT / "hires" / c.stem
        lo_m = json.loads((lo / "manifest.json").read_text())
        r = {"crop": c.name, "lo_board_w": Image.open(lo / "board.png").size[0],
             "lo_quotes": len(lo_m["quotes"]), "hi_present": hi.exists()}
        if lo_m["quotes"]:
            r["lo_sticker_w"] = lo_m["quotes"][0]["w"]
        if hi.exists():
            hi_m = json.loads((hi / "manifest.json").read_text())
            r["hi_board_w"] = Image.open(hi / "board.png").size[0]
            r["board_width_ratio"] = round(r["hi_board_w"] / r["lo_board_w"], 3)
            if hi_m["quotes"]:
                r["hi_sticker_w"] = hi_m["quotes"][0]["w"]
        rows.append(r)
    keys = ["crop", "lo_board_w", "hi_board_w", "board_width_ratio", "lo_quotes", "hi_present",
            "lo_sticker_w", "hi_sticker_w"]
    with (DATA / "per_crop_resolution.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)

    ratios = [r["board_width_ratio"] for r in rows if "board_width_ratio" in r]
    inv["hires_dirs"] = sum(r["hi_present"] for r in rows)
    inv["crops_missing_from_hires"] = [r["crop"] for r in rows if not r["hi_present"]]
    inv["crops_with_zero_stickers_lowres"] = [r["crop"] for r in rows if r["lo_quotes"] == 0]
    inv["board_width_ratio_median"] = st.median(ratios)
    inv["board_width_ratio_min_max"] = [min(ratios), max(ratios)]

    lo = index(OUT / "stickers" / "index.csv")
    hi = index(OUT / "stickers_hires" / "index.csv")
    for name, rows_ in (("album_lowres", lo), ("album_hires", hi)):
        ws = [int(r["width"]) for r in rows_]
        inv[name] = {
            "n": len(rows_),
            "median_width_px": st.median(ws),
            "median_width_in_at_300dpi": round(st.median(ws) / 300.0, 2),
            "flagged_duplicates": [(r["sticker"], r["looks like"]) for r in rows_ if r["looks like"]],
            "achromatic": sum(1 for r in rows_ if r["hue"] == ""),
        }

    # handover set: which files are byte-identical to the hires album
    send = sorted((OUT / "send").glob("sticker_*.png"))
    alt = sorted((OUT / "send" / "alt").glob("*.png"))
    modified = [p.name for p in send if md5(p) != md5(OUT / "stickers_hires" / p.name)]
    inv["handover"] = {
        "n_delivered": len(send),
        "n_byte_identical_to_pipeline": len(send) - len(modified),
        "modified_after_pipeline": modified,
        "alt_dupes": [p.name for p in alt if "dupe" in p.name],
        "alt_originals": [p.name for p in alt if "original" in p.name or "unrotated" in p.name],
    }

    (DATA / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
