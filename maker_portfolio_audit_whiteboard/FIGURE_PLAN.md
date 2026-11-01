# Figure Plan

Ranked by how directly each figure shows an engineering *decision*.
"Generated" figures are in `figures/` as PNG + SVG and are rebuilt by the
named script. **Privacy flag ⚠** = shows third parties' names or handwriting
(see `QUESTIONS_FOR_ME.md` Q11) and must not be published without consent or blurring.

| # | Title | Source asset | Decision demonstrated | Annotations / callouts | Claim supported | Status |
|---|---|---|---|---|---|---|
| 1 | **Failure → fix on the photo that broke v1** | `figures/fig_v1_failures_exampletext.png` (`s07`) | D9 quad rejection; D10 handwriting-scaled merge radius | Orange quad's top edge = image border; "5 stickers → 1"; "30 px → 0.7 × letter height" | Real photos exposed two v1 bugs; both reproduced from `b4bde13` and fixed | **Generated** ✅ (own "Hello" sample; confirm Q11a) |
| 2 | **Pipeline, stage by stage** | `figures/fig_pipeline_stages_IMG_0877.png` (`s05`, from the pipeline's own `--debug` output) | D1/D2/D3: lighting model, normalize, α estimator, unmix | Put the equations on the arrows: `I/B`, `α = max(1−min RGB, 0.7·S)`, `k = 1−(1−Î)/α` | The system is a physical-model inversion, not thresholding | **Generated** ✅ (confirm Q11a) |
| 3 | **A wrong α costs saturation, not hue (until the gamut clips)** | `figures/fig_unmix_wrong_alpha.png` (`s03`) | D1, and its limit | Shade α < 0.53 as "clipping region" | Why unmixing beats thresholding, and the precise condition where that stops holding | **Generated** ✅ |
| 4 | **Fat marker: hollow → solid** | `figures/fig_fat_marker_refit.png` + `fig_ablations_synthetic.png` (`s03`, `s07`) | D6 two-pass refit with inpainted exclusion | "56.6% → 100% core filled"; "median window 41 px vs stroke 18 px" | The refit fixes a mechanism-level failure; a paired test proves the failure exists without it | **Generated** ✅ |
| 5 | **Raw shoot frame: what the pipeline is up against** | `photos/20260925_113754.jpg` | Motivation for D8 (glare), D13 (board region), F6 (trim) | Arrows: ceiling-light reflections; aluminium trim; board running off frame; "no quad visible" | Real input breaks the model's assumptions in 3 named ways | Exists; **must annotate** ⚠ (blur names) |
| 6 | **Re-cut: screenshot vs camera pixels** | `figures/fig_recut_before_after.png` (`s07`) | D14 SIFT/RANSAC re-cut | "731 px → 1607 px (×2.20)"; also mark the glare patch the hi-res reveals (honest) | 2.24× median gain; print size 1.86 → 4.25 in | **Generated** ✅ ⚠ (fragment of a message; low risk, check) |
| 7 | **Locating 59 crops among 56 overlapping frames** | `figures/fig_recut_inlier_separation.png` (`s06`) | D14 accept threshold | "56/59"; the three misses at left; "runner-up often also contains the quote" | Matching works, and the threshold sits in a continuum (an honest limitation) | **Generated** ✅ |
| 8 | **Duplicate detector: shape proposes, colour vetoes** | `figures/fig_duplicate_detector.png` (`s02`) | D15 | flag region; the 2 misses | 8/8 precision, 8/9 recall, 1,653 pairs | **Generated** ✅ |
| 9 | **Lasso crop read alpha-blind** | `figures/fig_cleared_alpha_crop.png` (`s07`) | F3 / D11–D12 | "median 22.8% of a crop is cleared to black" | Why the alpha mask has to be read and eroded | **Generated** ✅ ⚠ (shows a message) |
| 10 | **Light/dark contact sheet (QA instrument)** | `out/send/sheet.png` (crop 2–3 tiles) or `out/stickers_hires/sheet.png` | D16/D17; dark-background halo test | "halo = grey, follows the stroke; translucency = darker but on-hue" | QA designed around how the failure shows up | Exists ⚠ (needs crop + blur, or regenerate on synthetic) |
| 11 | **Darkest-channel glare test** | must generate: bar chart of R/G/B for board, yellow ink and reflection (synthetic values) | D8 | "yellow is brighter than the board in R, G; darker in B" | Why brightest-channel tests delete yellow | **Must generate** (diagram, from `tests/synthetic.py` colours) |
| 12 | **System block diagram** | must generate from ENGINEERING_REPORT §2 | architecture | branch points: opaque vs RGBA, quad vs board-region | Scope and modularity | **Must generate** |
| 13 | **Iteration timeline** | must generate from `DEVELOPMENT_TIMELINE.md` | 3 merge rules; 4 versions | commit hashes on the axis | Iteration from real-world failure | **Must generate** |
| 14 | Synthetic test board with ground truth | must generate (`tests/synthetic.make_board()`) | testing strategy | glare blob, vignette, 6 inks | Ground-truth validation without real labels | **Must generate** (trivial) |
| 15 | Final delivered artefact (card/yearbook page) | **not in repo** (Q9) | outcome | — | That it was used | **Missing** — ask |

## Recommended card set (6 figures)
1 (failure → fix) · 2 (pipeline) · 4 (fat marker) · 3 (unmix math) · 6 or 7 (re-cut) · 5 (annotated raw frame, blurred).
Swap 8 in for 7 if the card needs a quantitative real-data plot.

## Figures NOT to make
* A "hue accuracy on real photos" bar chart: no real ground truth exists.
* A sat-weight or deglare ablation: both were null on the synthetic fixture.
  Plotting them would imply an effect that was never measured.
* Anything from the 14 hand-modified stickers presented as pipeline output.
