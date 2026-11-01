"""Plot best vs runner-up RANSAC inlier count per crop, from s04's matrix.

Tests the recut.py claim that "the true source scored hundreds against single
digits", using the matrix s04 measured (no new matching here).

    python maker_portfolio_audit/scripts/s06_plot_recut.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "maker_portfolio_audit" / "data"
FIG = ROOT / "maker_portfolio_audit" / "figures"
MIN_INLIERS = 25  # wbx.recut.MIN_INLIERS


def main() -> None:
    with (DATA / "recut_inlier_matrix.csv").open() as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    recs = []
    for r in body:
        vals = sorted(((int(v), header[i + 1]) for i, v in enumerate(r[1:])), reverse=True)
        recs.append({"crop": r[0], "best": vals[0][0], "best_photo": vals[0][1],
                     "second": vals[1][0], "second_photo": vals[1][1]})
    matched = [r for r in recs if r["best"] >= MIN_INLIERS]
    summary = {
        "n_crops": len(recs), "n_matched": len(matched),
        "best_min_median_max": [min(r["best"] for r in matched),
                                sorted(r["best"] for r in matched)[len(matched) // 2],
                                max(r["best"] for r in matched)],
        "runner_up_max": max(r["second"] for r in matched),
        "runner_up_median": sorted(r["second"] for r in matched)[len(matched) // 2],
        "min_ratio_best_over_second": min(r["best"] / max(1, r["second"]) for r in matched),
        "crops_where_runner_up_ge_25": [r for r in matched if r["second"] >= MIN_INLIERS],
    }
    (DATA / "recut_separation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    recs.sort(key=lambda r: r["best"])
    xs = range(len(recs))
    fig, ax = plt.subplots(figsize=(9.0, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.vlines(list(xs), [max(1, r["second"]) for r in recs], [max(1, r["best"]) for r in recs],
              color="#d6d4cf", lw=1.5, zorder=1)
    ax.scatter(list(xs), [max(1, r["second"]) for r in recs], s=22, c="#9a9893", zorder=2,
               label="runner-up photo (best of the other 55)")
    ax.scatter(list(xs), [max(1, r["best"]) for r in recs], s=26, c="#2a78d6", zorder=3,
               edgecolors="#fcfcfb", linewidths=0.8, label="best-matching photo")
    ax.axhline(MIN_INLIERS, color="#eb6834", lw=1.2, ls="--")
    ax.text(4, MIN_INLIERS * 0.72, "accept threshold: 25 inliers", ha="left",
            fontsize=8, color="#0b0b0b")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1200)
    ax.set_xlabel("hand-made crop (59), sorted by best inlier count", color="#0b0b0b")
    ax.set_ylabel("RANSAC inliers (log scale; 0 drawn at 1)", color="#0b0b0b")
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title(f"Locating each hand-made crop among 56 overlapping originals (SIFT + RANSAC): "
                 f"{summary['n_matched']}/{summary['n_crops']} matched.\n"
                 f"The runner-up is often high too (median {summary['runner_up_median']}): "
                 f"neighbouring frames overlap and contain the same quote; recut takes the maximum.",
                 fontsize=9, loc="left", color="#0b0b0b")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_recut_inlier_separation.{ext}", dpi=220, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
