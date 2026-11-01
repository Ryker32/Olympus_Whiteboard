"""Score the album duplicate detector on the real shoot, against the handover cull.

Uses wbx.album's own signature and ink functions (unchanged) to compute, for every
pair of stickers, the silhouette correlation and the core-ink RGB distance -- the
two quantities album.find_duplicates thresholds at 0.45 and 0.35.

Reference labels: the 9 stickers placed in out/send/alt/ as "*_dupe.png" (the
culled copies of the handover set, hires album numbering).  Each is paired with
the kept copy of the same source quote.  Eight partners are the album's own
flags; the ninth (sticker_20 <-> sticker_38, crops 010643 / 010925) was flagged
by the low-res album run but not the hires run.  WHO made the cull is not
recorded in the repository -- treat these labels as INFERRED ground truth.

    PYTHONPATH=src python maker_portfolio_audit/scripts/s02_duplicates.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wbx.album import (DUPLICATE_CORRELATION, DUPLICATE_INK_DISTANCE,  # noqa: E402
                       _ink_mean, _signature, find_duplicates)

DATA = ROOT / "maker_portfolio_audit" / "data"
FIG = ROOT / "maker_portfolio_audit" / "figures"

# culled copy -> kept copy, by SOURCE CROP (stable across both album numberings)
CULL_PAIRS = [
    ("010428", "010433"), ("010555", "010659"), ("010559", "010602"), ("010643", "010925"),
    ("010731", "011218"), ("010748", "010751"), ("010817", "011132"), ("011054", "011058"),
    ("010711", "011206"),
]


def album(name: str):
    folder = ROOT / "out" / name
    with (folder / "index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    paths = [folder / r["sticker"] for r in rows]
    stamps = [r["source photo"].split()[-1].removesuffix(".png") for r in rows]
    return rows, paths, stamps


def score(name: str) -> dict:
    rows, paths, stamps = album(name)
    sigs = [_signature(p) for p in paths]
    inks = [_ink_mean(p) for p in paths]
    truth = {frozenset(p) for p in CULL_PAIRS}
    pairs = []
    for i, j in itertools.combinations(range(len(paths)), 2):
        if sigs[i] is None or sigs[j] is None or inks[i] is None or inks[j] is None:
            continue
        pairs.append({
            "a": rows[i]["sticker"], "b": rows[j]["sticker"],
            "corr": float((sigs[i] * sigs[j]).mean()),
            "ink_dist": float(np.linalg.norm(inks[i] - inks[j])),
            "same_quote": frozenset((stamps[i], stamps[j])) in truth,
        })
    flagged = find_duplicates(paths)  # the shipped decision rule
    flagged_pairs = {frozenset((stamps[i], stamps[j])) for i, j in flagged.items()}
    tp = len(flagged_pairs & truth)
    present = {t for t in truth if all(s in stamps for s in t)}

    dup = [p for p in pairs if p["same_quote"]]
    non = [p for p in pairs if not p["same_quote"]]
    shape_alike = [p for p in non if p["corr"] >= DUPLICATE_CORRELATION]
    summary = {
        "album": name, "n_stickers": len(paths), "n_pairs": len(pairs),
        "reference_pairs_present": len(present), "flagged": len(flagged_pairs), "true_positives": tp,
        "precision": tp / max(1, len(flagged_pairs)), "recall": tp / max(1, len(present)),
        "dup_corr_range": [min(p["corr"] for p in dup), max(p["corr"] for p in dup)],
        "dup_ink_max": max(p["ink_dist"] for p in dup),
        "max_nondup_corr": max(p["corr"] for p in non),
        "nondup_above_corr_threshold": len(shape_alike),
        "min_ink_of_shape_alike_nondup": min((p["ink_dist"] for p in shape_alike), default=None),
        "missed": [sorted(t) for t in present - flagged_pairs],
    }
    with (DATA / f"duplicate_pairs_{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["a", "b", "corr", "ink_dist", "same_quote"])
        w.writeheader()
        w.writerows(pairs)
    return summary | {"_pairs": pairs}


def plot(results: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True, facecolor="#fcfcfb")
    for ax, res in zip(axes, results):
        ax.set_facecolor("#fcfcfb")
        pairs = res["_pairs"]
        non = [p for p in pairs if not p["same_quote"]]
        dup = [p for p in pairs if p["same_quote"]]
        ax.axvspan(DUPLICATE_CORRELATION, 1.05, ymin=0, ymax=DUPLICATE_INK_DISTANCE / 1.2,
                   color="#2a78d6", alpha=0.07, lw=0)
        ax.scatter([p["corr"] for p in non], [p["ink_dist"] for p in non], s=6, c="#9a9893",
                   alpha=0.45, lw=0, label=f"different quotes (n={len(non)})")
        ax.scatter([p["corr"] for p in dup], [p["ink_dist"] for p in dup], s=46, c="#eb6834",
                   edgecolors="#fcfcfb", linewidths=1.2, zorder=3,
                   label=f"same quote, culled copy (n={len(dup)})")
        ax.axvline(DUPLICATE_CORRELATION, color="#52514e", lw=1, ls="--")
        ax.axhline(DUPLICATE_INK_DISTANCE, color="#52514e", lw=1, ls="--")
        ax.text(DUPLICATE_CORRELATION + 0.01, 1.13, "corr ≥ 0.45", color="#52514e", fontsize=8)
        ax.text(-0.38, DUPLICATE_INK_DISTANCE + 0.02, "ink ≤ 0.35", color="#52514e", fontsize=8)
        ax.text(0.98, 0.03, "flag region", color="#2a78d6", fontsize=8, ha="right")
        for p in dup:
            if p["corr"] < DUPLICATE_CORRELATION or p["ink_dist"] > DUPLICATE_INK_DISTANCE:
                ax.annotate("missed", (p["corr"], p["ink_dist"]), xytext=(8, 8),
                            textcoords="offset points", fontsize=8, color="#0b0b0b")
        ax.set_xlim(-0.4, 1.05)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("silhouette correlation (z-scored 48×32 alpha thumbnail)")
        ax.set_title(f"{res['album']}: {res['n_stickers']} stickers, {res['n_pairs']} pairs\n"
                     f"flagged {res['flagged']}, precision {res['precision']:.2f}, "
                     f"recall {res['true_positives']}/{res['reference_pairs_present']}",
                     fontsize=9, loc="left", color="#0b0b0b")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RGB distance of core ink colour (0–√3)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Duplicate-crop detector on the real 59-crop shoot (labels: handover cull in out/send/alt)",
                 fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_duplicate_detector.{ext}", dpi=220, facecolor="#fcfcfb")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    results = [score("stickers"), score("stickers_hires")]
    plot(results)
    clean = [{k: v for k, v in r.items() if k != "_pairs"} for r in results]
    (DATA / "duplicate_summary.json").write_text(json.dumps(clean, indent=2) + "\n")
    print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
