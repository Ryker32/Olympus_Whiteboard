"""Wall-clock runtime of wbx.pipeline.process_image on the real photos, and a
stage-by-stage figure built from the pipeline's own --debug intermediates.

No runtime figure exists anywhere in the repository; these are new
measurements on this machine and should be quoted with the hardware attached.

    python maker_portfolio_audit/scripts/s05_runtime_and_stages.py <scratch_dir>
"""

from __future__ import annotations

import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wbx.io import load_rgba  # noqa: E402
from wbx.pipeline import Params, process_image, process_photo  # noqa: E402

DATA = ROOT / "maker_portfolio_audit" / "data"
FIG = ROOT / "maker_portfolio_audit" / "figures"
REPEATS = 3


def timed(path: Path) -> dict:
    img, crop = load_rgba(path)
    process_image(img, Params(), crop=crop)  # warm-up
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        res = process_image(img, Params(), crop=crop)
        ts.append(time.perf_counter() - t0)
    return {"file": path.name, "w": img.shape[1], "h": img.shape[0],
            "megapixels": round(img.shape[0] * img.shape[1] / 1e6, 2),
            "pre_cropped": crop is not None, "n_quotes": len(res[2]),
            "median_s": round(st.median(ts), 3), "min_s": round(min(ts), 3), "max_s": round(max(ts), 3)}


def main(scratch: Path) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    photos = sorted((ROOT / "photos").glob("*.jpg"))
    targets = [ROOT / "samples" / n for n in
               ("nomarker.jpeg", "exampletext.jpeg", "IMG_0877.jpeg", "IMG_0878.jpeg")]
    targets += [photos[0], photos[len(photos) // 2], photos[-1]]
    recut = sorted((scratch / "recut").glob("*.png"))[:4]
    targets += recut

    rows = [timed(p) for p in targets]
    meta = {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
            "machine": platform.platform(), "processor": platform.processor(),
            "repeats_after_warmup": REPEATS}
    (DATA / "runtime.json").write_text(json.dumps({"env": meta, "runs": rows}, indent=2) + "\n")
    for r in rows:
        print(r)

    # stage figure: the pipeline's own debug intermediates for one iPad sample
    src = ROOT / "samples" / "IMG_0877.jpeg"
    res = process_photo(src, scratch / "debug_run", Params(), debug=True)
    dbg = res.out_dir / "debug"
    q = res.manifest["quotes"][0]
    stage_figure(dbg, res.out_dir, q)


def stage_figure(dbg: Path, out_dir: Path, q: dict) -> None:
    import matplotlib.pyplot as plt

    x, y, w, h = q["x"], q["y"], q["w"], q["h"]
    m = int(0.25 * max(w, h))

    def window(path, gray=False):
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        H, W = im.shape[:2]
        im = im[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
        if gray:
            return im
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    sticker = cv2.imread(str(out_dir / q["file"]), cv2.IMREAD_UNCHANGED)
    rgb = sticker[..., [2, 1, 0]].astype(np.float32) / 255
    a = sticker[..., 3:].astype(np.float32) / 255

    def on(bg):
        return rgb * a + bg * (1 - a)

    yy, xx = np.mgrid[0:sticker.shape[0], 0:sticker.shape[1]]
    chk = (((yy // 24) + (xx // 24)) % 2)[..., None]
    dark = np.where(chk, 0.18, 0.10).astype(np.float32)

    panels = [
        (window(dbg / "01_rectified.png"), "1  input photo (iPad, 3264×2448)", None),
        (window(dbg / "02_background.png"), "2  per-channel board lighting\n(2-pass, ink masked & inpainted)", None),
        (window(dbg / "03_normalized.png"), "3  photo ÷ lighting:\nbare board = 1.0", None),
        (window(dbg / "04_alpha.png", gray=True), "4  coverage α =\nmax(1−min RGB, 0.7·sat)", "gray_r"),
        (on(dark * np.ones(3, np.float32)), "5  exported sticker,\nover a dark checkerboard", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), facecolor="#fcfcfb")
    for ax, (im, title, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(title, fontsize=8.5, loc="left", color="#0b0b0b")
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"fig_pipeline_stages_IMG_0877.{ext}", dpi=200, facecolor="#fcfcfb")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
