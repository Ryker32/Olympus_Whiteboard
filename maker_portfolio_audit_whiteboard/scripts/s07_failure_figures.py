"""Before/after figures for documented failures, re-run from the original code.

F1/F2: v1 (commit b4bde13) on samples/exampletext.jpeg -- the hallucinated board
       quad and the "Hello" split -- vs the current code.  v1 is exported with
       `git archive b4bde13` into a scratch dir (read-only on .git) and imported
       from there; the working tree is never touched.
F3:    a real lasso crop read alpha-blind (black under alpha 0) vs its mask.
F4:    synthetic fat marker, alpha with and without the two-pass refit.
F7:    one screenshot crop vs its SIFT re-cut, at the same print scale.

    python maker_portfolio_audit/scripts/s07_failure_figures.py <scratch_dir>
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "maker_portfolio_audit" / "figures"
DATA = ROOT / "maker_portfolio_audit" / "data"
SURF, INK, SEC = "#fcfcfb", "#0b0b0b", "#52514e"


def export_commit(rev: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(["git", "-C", str(ROOT), "archive", rev, "src"], check=True,
                          capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest)
    return dest / "src"


def load_wbx(src: Path):
    for k in [k for k in sys.modules if k == "wbx" or k.startswith("wbx.")]:
        del sys.modules[k]
    sys.path.insert(0, str(src))
    mods = {n: importlib.import_module(f"wbx.{n}") for n in ("pipeline", "rectify", "io")}
    sys.path.remove(str(src))
    return mods


def thumb(img, max_w=900):
    s = min(1.0, max_w / img.shape[1])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def style(ax, title):
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.axis("off")


def f1_f2(scratch: Path, plt):
    photo_path = ROOT / "samples" / "exampletext.jpeg"
    v1 = load_wbx(export_commit("b4bde13", scratch / "v1"))
    img = v1["io"].load_image(photo_path)
    quad_v1 = v1["rectify"].find_board_quad(img)
    _i, a1, q1, *_ = v1["pipeline"].process_image(img, v1["pipeline"].Params())

    cur = load_wbx(ROOT / "src")
    quad_now = cur["rectify"].find_board_quad(img)
    _i, a2, q2, _l, _r, r2 = cur["pipeline"].process_image(img, cur["pipeline"].Params())
    res = {"v1_quad": None if quad_v1 is None else quad_v1.tolist(), "v1_n_stickers": len(q1),
           "current_quad": None if quad_now is None else quad_now.tolist(),
           "current_n_stickers": len(q2), "current_merge_radius_px": int(r2)}
    (DATA / "v1_vs_current_exampletext.json").write_text(json.dumps(res, indent=2) + "\n")
    print(res)

    small, s = thumb(img)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURF)
    axes[0].imshow(small)
    if quad_v1 is not None:
        q = np.vstack([quad_v1, quad_v1[:1]]) * s
        axes[0].plot(q[:, 0], q[:, 1], color="#eb6834", lw=2.2)
    style(axes[0], "F1 · v1 (b4bde13): 'board' quad found on a\nframe-filling board (orange) — would be warped")

    def boxes(ax, alpha, quotes, colour):
        sm, sc = thumb(1.0 - np.clip(alpha, 0, 1))
        ax.imshow(sm, cmap="gray", vmin=0, vmax=1)
        for (_lab, x, y, w, h, _a) in quotes:
            ax.add_patch(plt.Rectangle((x * sc, y * sc), w * sc, h * sc, fill=False, ec=colour, lw=1.8))

    _i, a1n, q1n, *_ = v1["pipeline"].process_image(img, v1["pipeline"].Params(rectify=False))
    boxes(axes[1], a1n, q1n, "#eb6834")
    style(axes[1], f"F2 · v1, fixed 30 px merge radius:\n'Hello' → {len(q1n)} stickers")
    boxes(axes[2], a2, q2, "#2a78d6")
    style(axes[2], f"current (0a999a4): no quad (correct), radius {r2} px\n= 0.7 × letter height → {len(q2)} sticker")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_v1_failures_exampletext.{ext}", dpi=170, facecolor=SURF)


def f3_f7(scratch: Path, plt):
    from PIL import Image

    name = "Screenshot 2026-09-26 010508"
    shot = np.asarray(Image.open(ROOT / "samples" / "cropped" / f"{name}.png").convert("RGBA"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), facecolor=SURF)
    axes[0].imshow(shot[..., :3])
    style(axes[0], "F3 · lasso crop read alpha-blind: editors write\nRGB = 0 under α = 0 (the black is not board)")
    axes[1].imshow(shot[..., 3], cmap="gray")
    style(axes[1], "the same file's alpha: the operator's segmentation,\nused as the crop mask (eroded 3 px)")
    cleared = float((shot[..., 3] < 255).mean())
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / "fig_cleared_alpha_crop.{}".format(ext), dpi=170, facecolor=SURF)

    lo = cv2.imread(str(ROOT / "out" / "cropped" / name / "quote_01.png"), cv2.IMREAD_UNCHANGED)
    hi = cv2.imread(str(ROOT / "out" / "hires" / name / "quote_01.png"), cv2.IMREAD_UNCHANGED)
    zx, zy = int(0.25 * lo.shape[1]), int(0.30 * lo.shape[0])  # same physical patch in both
    zw, zh = int(0.22 * lo.shape[1]), int(0.30 * lo.shape[0])
    g = hi.shape[1] / lo.shape[1]

    def on_white(rgba):
        rgb = rgba[..., [2, 1, 0]].astype(np.float32) / 255
        a = rgba[..., 3:].astype(np.float32) / 255
        return rgb * a + (1 - a)

    lo_patch = on_white(lo[zy:zy + zh, zx:zx + zw])
    hi_patch = on_white(hi[int(zy * g):int((zy + zh) * g), int(zx * g):int((zx + zw) * g)])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURF)
    axes[0].imshow(cv2.resize(lo_patch, (hi_patch.shape[1], hi_patch.shape[0]), interpolation=cv2.INTER_NEAREST))
    style(axes[0], f"F7 · from the screenshot crop: {lo.shape[1]} px wide\n= {lo.shape[1] / 300:.1f} in at 300 dpi")
    axes[1].imshow(hi_patch)
    style(axes[1], f"from the SIFT/RANSAC re-cut original: {hi.shape[1]} px\n= {hi.shape[1] / 300:.1f} in at 300 dpi  (×{g:.2f})")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_recut_before_after.{ext}", dpi=170, facecolor=SURF)
    return {"cleared_fraction_010508": cleared}


def f4(plt):
    sys.path.insert(0, str(ROOT / "tests"))
    import synthetic as syn

    cur = load_wbx(ROOT / "src")
    photo, _ = syn.make_board()
    fat, cov = syn.add_fat_word(photo)
    P = cur["pipeline"]
    _i, a_on, *_ = P.process_image(fat, P.Params(rectify=False))
    _i, a_off, *_ = P.process_image(fat, P.Params(rectify=False, bg_refine=False))
    ys, xs = np.nonzero(cov > 0.1)
    sl = (slice(ys.min() - 20, ys.max() + 20), slice(xs.min() - 20, xs.max() + 20))
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6), facecolor=SURF)
    axes[0].imshow(fat[sl])
    style(axes[0], "F4 · synthetic chisel-tip word\n(~10 px median / 18 px peak stroke)")
    axes[1].imshow(1 - a_off[sl], cmap="gray", vmin=0, vmax=1)
    style(axes[1], "coverage α, single background pass:\nmedian sinks into the ink → hollow")
    axes[2].imshow(1 - a_on[sl], cmap="gray", vmin=0, vmax=1)
    style(axes[2], "coverage α, two-pass refit\n(ink masked, Telea-inpainted, refit)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_fat_marker_refit.{ext}", dpi=170, facecolor=SURF)


def main(scratch: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    f1_f2(scratch, plt)
    extra = f3_f7(scratch, plt)
    f4(plt)
    print(extra)

    # F3 claim check: "20-40% of a typical crop" is cleared (alpha 0) -- measure all 59
    from PIL import Image
    fr = []
    for p in sorted((ROOT / "samples" / "cropped").glob("*.png")):
        a = np.asarray(Image.open(p).convert("RGBA"))[..., 3]
        fr.append(float((a == 0).mean()))
    fr.sort()
    stats = {"n": len(fr), "min": fr[0], "p25": fr[len(fr) // 4], "median": fr[len(fr) // 2],
             "p75": fr[3 * len(fr) // 4], "max": fr[-1],
             "share_in_20_40pct": sum(0.2 <= f <= 0.4 for f in fr) / len(fr)}
    (DATA / "crop_cleared_fraction.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
