# Reproducing the audit numbers and figures

Run from the repository root. Every script is read-only with respect to the repo:
outputs go to `maker_portfolio_audit/data/` and `maker_portfolio_audit/figures/`,
or to the scratch directory you pass in. Audit environment: Python 3.13.5,
OpenCV 5.0.0, NumPy 2.1.3, matplotlib 3.10.0 (Anaconda). The project's own
`uv.lock` pins typer 0.27.2 and NumPy 2.4/2.5, which may shift numbers slightly.

| script | needs | produces |
|---|---|---|
| `s01_inventory.py` | `out/`, `samples/cropped`, `photos/` | `data/inventory.json`, `data/per_crop_resolution.csv` |
| `s02_duplicates.py` | `out/stickers*`, `src/` | `data/duplicate_*.{csv,json}`, `figures/fig_duplicate_detector.*` |
| `s03_synthetic_metrics.py` | `src/`, `tests/synthetic.py` | `data/synthetic_metrics.json`, `figures/fig_ablations_synthetic.*`, `figures/fig_unmix_wrong_alpha.*` |
| `s04_recut_repro.py SCRATCH` | `samples/cropped`, `photos/` (~5 min) | `data/recut_results.csv`, `data/recut_inlier_matrix.csv`, `data/recut_timing.txt`, re-cut PNGs in SCRATCH/recut |
| `s05_runtime_and_stages.py SCRATCH` | s04's SCRATCH/recut | `data/runtime.json`, `figures/fig_pipeline_stages_IMG_0877.*` |
| `s06_plot_recut.py` | s04 matrix | `data/recut_separation.json`, `figures/fig_recut_inlier_separation.*` |
| `s07_failure_figures.py SCRATCH` | git (runs `git archive b4bde13`, read-only) | `figures/fig_v1_failures_exampletext.*`, `fig_cleared_alpha_crop.*`, `fig_recut_before_after.*`, `fig_fat_marker_refit.*`, `data/v1_vs_current_exampletext.json`, `data/crop_cleared_fraction.json` |

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
S=/path/to/scratch
python maker_portfolio_audit/scripts/s01_inventory.py
python maker_portfolio_audit/scripts/s02_duplicates.py
python maker_portfolio_audit/scripts/s03_synthetic_metrics.py
python maker_portfolio_audit/scripts/s04_recut_repro.py $S
python maker_portfolio_audit/scripts/s05_runtime_and_stages.py $S
python maker_portfolio_audit/scripts/s06_plot_recut.py
python maker_portfolio_audit/scripts/s07_failure_figures.py $S
# test suite, without touching .pytest_cache:
PYTHONPATH=src python -m pytest -p no:cacheprovider -q --basetemp=$S/pt
```
RANSAC is stochastic: inlier counts can vary by a few between runs. The 56/59
outcome and 2.24× gain were stable across the two runs made here.
