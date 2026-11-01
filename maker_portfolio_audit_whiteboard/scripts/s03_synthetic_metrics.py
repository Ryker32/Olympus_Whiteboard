"""Measure the actual values behind the synthetic test suite, plus flag-level ablations.

tests/test_synthetic.py only asserts thresholds (hue <= 8 deg, IoU > 0.7, ...).
This records the numbers themselves, on the same fixtures, with the same metric
definitions, and toggles only switches that already exist in wbx.pipeline.Params.
Nothing here is real-photo evidence: every value is on the generated board in
tests/synthetic.py (1600x1000, seed 0).

    python maker_portfolio_audit/scripts/s03_synthetic_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synthetic as syn  # noqa: E402
from wbx import matte, rectify as rectify_mod, unmix as unmix_mod  # noqa: E402
from wbx.pipeline import Params, process_image  # noqa: E402

DATA = ROOT / "maker_portfolio_audit" / "data"
FIG = ROOT / "maker_portfolio_audit" / "figures"

INK, SEC, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#9a9893", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


# metric definitions copied from tests/test_synthetic.py so the numbers mean the same thing
def hue_of(rgb):
    return float(matte.hue(np.asarray(rgb, np.float32).reshape(1, 1, 3))[0, 0])


def hue_delta(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def dominant_hue(ink, alpha, sel):
    w = (alpha * matte.saturation(ink))[sel].astype(np.float64)
    th = np.deg2rad(matte.hue(ink)[sel].astype(np.float64))
    return float(np.rad2deg(np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())) % 360.0)


def run(photo, **kw):
    return process_image(photo, Params(rectify=False, **kw))


def fat_core(cov):
    d = cv2.distanceTransform((cov > 0.5 * cov.max()).astype(np.uint8), cv2.DIST_L2, 5)
    return d >= 6


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    photo, strokes = syn.make_board()
    ink, alpha, quotes, _labels, _r, merge_r = run(photo)

    total = np.zeros(alpha.shape, np.float32)
    for s in strokes:
        total = np.maximum(total, s.coverage)
    out["bare_board_leak_fraction"] = float((alpha[total < 0.02] > 0.05).mean())
    out["n_quotes"] = len(quotes)
    out["n_words"] = len(strokes)
    out["auto_merge_radius_px"] = int(merge_r)

    per = {}
    got = alpha > 0
    for s in strokes:
        truth = s.coverage > 0.1
        ys, xs = np.nonzero(truth)
        box = np.zeros(alpha.shape, bool)
        box[max(0, ys.min() - 30):ys.max() + 31, max(0, xs.min() - 30):xs.max() + 31] = True
        near = got & box
        rec = {"iou": float((near & truth).sum() / (near | truth).sum())}
        if s.name not in syn.ACHROMATIC:
            core = (s.coverage > 0.5 * s.coverage.max()) & (alpha > 0)
            rec["hue_error_deg"] = hue_delta(dominant_hue(ink, alpha, core), hue_of(s.ink))
        per[s.name] = rec
    out["per_word"] = per

    # rectification: corner error vs yaw (the test checks 25 deg only)
    sweep = {}
    for ang in (0, 10, 20, 25, 30, 40, 50):
        frame, corners = syn.place_on_wall(photo, angle_deg=float(ang))
        found = rectify_mod.find_board_quad(frame)
        sweep[ang] = None if found is None else float(
            np.abs(found - rectify_mod.order_corners(corners)).max())
    out["rectify_max_corner_error_px_by_yaw_deg"] = sweep

    # ---- ablations, each with the shipped default vs one existing flag flipped
    abl = {}
    fat_photo, fat_cov = syn.add_fat_word(photo)
    core = fat_core(fat_cov)
    for refine in (True, False):
        _i, a, *_ = run(fat_photo, bg_refine=refine)
        abl[f"fat_stroke_core_filled__bg_refine={refine}"] = float((a[core] > 0.5).mean())

    yellow = next(s for s in strokes if s.name == "yellow")
    ycore = yellow.coverage > 0.6 * yellow.coverage.max()
    for sw in (0.7, 0.0):
        _i, a, *_ = run(photo, sat_weight=sw)
        abl[f"yellow_core_retained__sat_weight={sw}"] = float((a[ycore] > 0).mean())

    h, w = photo.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - 0.30 * w) ** 2 + (yy - 0.72 * h) ** 2) / (2 * (0.05 * w) ** 2)))
    glared = np.clip(photo + 0.30 * blob[..., None], 0, 1).astype(np.float32)
    spot = blob > 0.5
    for dg in (True, False):
        _i, a, q, *_ = run(glared, deglare=dg)
        abl[f"glare_spot_matted_fraction__deglare={dg}"] = float((a[spot] > 0).mean())

    red = next(s for s in strokes if s.name == "red")
    rcore = red.coverage > 0.6 * red.coverage.max()
    abl["red_core_true_coverage"] = float(red.coverage[rcore].mean())
    for label, kw in (("default", {}), ("no_chroma_mask", {"bg_ink_chroma": 10.0}),
                      ("no_refit", {"bg_refine": False})):
        _i, a, *_ = run(photo, **kw)
        abl[f"red_core_recovered_coverage__{label}"] = float(a[rcore].mean())
    out["ablations"] = abl

    # ---- unmixing under a wrong alpha: hue vs chroma, the README's central claim
    yel = np.float32([0.98, 0.84, 0.24])
    a_true = 0.7
    obs = (a_true * yel + (1 - a_true)).reshape(1, 1, 3).astype(np.float32)
    guesses = np.round(np.linspace(0.3, 1.0, 29), 3)
    curve = []
    for g in guesses:
        rec = unmix_mod.unmix(obs, np.float32([[g]]))[0, 0]
        curve.append({"alpha_guess": float(g), "hue_error_deg": hue_delta(hue_of(rec), hue_of(yel)),
                      "chroma": float(rec.max() - rec.min())})
    out["unmix_wrong_alpha_curve"] = curve
    out["unmix_true_chroma"] = float(yel.max() - yel.min())

    (DATA / "synthetic_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "unmix_wrong_alpha_curve"}, indent=2))
    plot_ablation(abl)
    plot_unmix(curve, out["unmix_true_chroma"], a_true)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(SEC)
    ax.spines["bottom"].set_color(SEC)
    ax.tick_params(colors=SEC)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_ablation(abl):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    # Only the toggles that changed the outcome on this fixture are drawn.  The
    # sat_weight and deglare toggles produced identical results (see
    # synthetic_metrics.json) and are reported as null results in the text,
    # not plotted as if they showed something.
    abl = dict(abl)
    abl["red_rel__on"] = abl["red_core_recovered_coverage__default"] / abl["red_core_true_coverage"]
    abl["red_rel__off"] = abl["red_core_recovered_coverage__no_refit"] / abl["red_core_true_coverage"]
    rows = [
        ("Fat chisel stroke: core pixels filled (α > 0.5)", "fat_stroke_core_filled__bg_refine=True",
         "fat_stroke_core_filled__bg_refine=False", "two-pass refit", "single pass", True),
        ("Dense red word: recovered / true coverage in stroke core", "red_rel__on",
         "red_rel__off", "two-pass refit", "single pass", True),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 2.7), facecolor=SURF)
    for ax, (title, on, off, lon, loff, higher_better) in zip(axes, rows):
        style(ax)
        vals = [abl[off] * 100, abl[on] * 100]
        ax.barh([0, 1], vals, height=0.62, color=[MUTED, BLUE])
        for y, v, lab in zip([0, 1], vals, [loff, lon]):
            ax.text(v + 1.2, y, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
        ax.set_yticks([0, 1], [loff, lon], fontsize=8.5)
        ax.set_xlim(0, 112)
        ax.set_title(title + ("  (higher is better)" if higher_better else "  (lower is better)"),
                     fontsize=9, loc="left", color=INK)
    axes[-1].set_xlabel("% of fixture pixels  (synthetic board, ground truth known; one flag toggled per pair)",
                        color=INK)
    fig.tight_layout(h_pad=1.0)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_ablations_synthetic.{ext}", dpi=220, facecolor=SURF)


def plot_unmix(curve, true_chroma, a_true):
    import matplotlib.pyplot as plt

    g = [c["alpha_guess"] for c in curve]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.0), facecolor=SURF)
    for ax in (a1, a2):
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.axvline(a_true, color=SEC, lw=1, ls="--")
        ax.set_xlabel("alpha used for unmixing (true α = 0.7)", color=INK)
    a1.plot(g, [c["hue_error_deg"] for c in curve], color=BLUE, lw=2, marker="o", ms=4)
    a1.set_ylabel("hue error (degrees)", color=INK)
    a1.set_ylim(0, 10)
    a1.axhline(8, color=ORANGE, lw=1)
    a1.text(0.72, 8.25, "test tolerance 8°", fontsize=8, color=INK)
    a1.text(0.43, 4.6, "← α underestimated:\nray leaves the gamut,\nchannel clips", fontsize=7.5, color=SEC)
    a1.set_title("Hue of recovered ink", fontsize=9, loc="left", color=INK)
    a2.plot(g, [c["chroma"] for c in curve], color=BLUE, lw=2, marker="o", ms=4)
    a2.axhline(true_chroma, color=ORANGE, lw=1)
    a2.text(0.31, true_chroma + 0.02, "true chroma", fontsize=8, color=INK)
    a2.set_ylabel("chroma  max(RGB) − min(RGB)", color=INK)
    a2.set_title("Saturation of recovered ink", fontsize=9, loc="left", color=INK)
    fig.suptitle("ink = 1 − (1 − observed)/α on one yellow pixel: a wrong α costs saturation, not hue",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_unmix_wrong_alpha.{ext}", dpi=220, facecolor=SURF)


if __name__ == "__main__":
    main()
