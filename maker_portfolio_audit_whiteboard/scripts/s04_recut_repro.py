"""Independently re-run `wbx recut` on the real shoot and record the full inlier matrix.

Reproduces the README claim "56 of 59 crops at 2.2x linear" and the recut.py
docstring claim "the true source scored hundreds against single digits".

Read-only with respect to the repository: re-cut PNGs go to a scratch directory
given on the command line (never into the repo), and only CSVs are written into
maker_portfolio_audit/data/.

    PYTHONPATH=src python maker_portfolio_audit/scripts/s04_recut_repro.py <scratch_dir>
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wbx.io import load_rgba  # noqa: E402
from wbx.recut import PHOTO_SCALE, RATIO, index_photos, recut  # noqa: E402

DATA = ROOT / "maker_portfolio_audit" / "data"


def inlier_row(crop_path: Path, index, sift, matcher) -> list[int]:
    """Inliers against every photo -- the same scoring as wbx.recut.locate, all kept."""
    rgb, mask = load_rgba(crop_path)
    if mask is None:
        mask = np.ones(rgb.shape[:2], bool)
    ys, xs = np.nonzero(mask)
    sub = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gray = cv2.cvtColor((sub * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kt, dt = sift.detectAndCompute(gray, None)
    row = []
    for _path, kq, dq in index:
        if dt is None or len(kt) < 8 or dq is None or len(kq) < 8:
            row.append(0)
            continue
        pairs = matcher.knnMatch(dt, dq, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
        if len(good) < 8:
            row.append(0)
            continue
        src = np.float32([kt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        row.append(int(inl.sum()) if (H is not None and inl is not None) else 0)
    return row


def main(scratch: Path) -> None:
    crops = sorted((ROOT / "samples" / "cropped").glob("*.png"))
    photos = sorted((ROOT / "photos").glob("*.jpg"))
    DATA.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = recut(crops, photos, scratch / "recut")  # the shipped code path, untouched
    t_recut = time.perf_counter() - t0
    with (DATA / "recut_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "matched_photo", "best_inliers", "written", "width", "height", "gain"])
        for r in results:
            w.writerow([r.crop.name, r.photo.name if r.photo else "", r.inliers,
                        bool(r.written), r.width, r.height, r.gain])

    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    index = index_photos(photos, sift, PHOTO_SCALE)
    with (DATA / "recut_inlier_matrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop"] + [p.name for p in photos])
        for c in crops:
            w.writerow([c.name] + inlier_row(c, index, sift, matcher))

    found = [r for r in results if r.written]
    gains = sorted(r.gain for r in found)
    print(f"recut wall time {t_recut:.1f}s for {len(crops)} crops x {len(photos)} photos")
    print(f"{len(found)}/{len(results)} re-cut; median gain {gains[len(gains) // 2]:.2f}x")
    (DATA / "recut_timing.txt").write_text(
        f"wbx.recut.recut wall time: {t_recut:.1f} s, {len(crops)} crops x {len(photos)} photos\n"
        f"re-cut: {len(found)}/{len(results)}; median gain {gains[len(gains) // 2]:.2f}x\n"
        f"python {sys.version.split()[0]}, opencv {cv2.__version__}, numpy {np.__version__}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
