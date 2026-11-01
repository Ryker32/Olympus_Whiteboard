# Engineering Report: `wbx`, Whiteboard Quote Extractor (repo "Olympus_Whiteboard")

Audit date 2026-10-26. Repository state: branch `hires-stickers-and-glare-fixes`
@ `0a999a4`, clean apart from untracked `samples/` files. Nothing in the
repository was modified. All new artefacts are in `maker_portfolio_audit/`.

Labels: **VERIFIED** = checked against code/data/execution in this audit ·
**INFERRED** = reasoned from indirect evidence · **UNVERIFIED** = claimed in the
repo, no data found · **CONTRADICTED** = the repository's own data or execution disagrees.

---

## 0. What this project is (and is not)

Despite the name, this is **not** a collaborative whiteboard app. There is no
UI, server, persistence layer, concurrency or deployment. It is a **Python CLI
for classical computer vision** (`wbx`, package `whiteboard-extract`). It turns
phone photos of a handwritten whiteboard into transparent RGBA PNG "stickers"
of each message, keeping the marker colours, for print layout. VERIFIED:
`pyproject.toml:1-18`, `README.md:1-5`, `src/wbx/cli.py`.

* **Scale:** 2,836 lines (src 1,830 + tests 896), 14 modules, 37 tests. VERIFIED (`wc -l`).
* **Interfaces:** four CLI commands: `extract`, `recut`, `album`, `sheet`
  (`src/wbx/cli.py:542-657`). No GUI. A drag-and-drop web UI is listed as a
  stretch goal (`README.md:296`).
* **Only learned model:** optional MediaPipe person segmentation (`src/wbx/people.py`).
  Everything else is deterministic image processing.

So the card should present this as **applied imaging and computer-vision
engineering driven by real-world failure cases**. It is not product/UX work, and
it is not a novel algorithm.

## 1. Project objective and constraints

**Stated objective (VERIFIED, `README.md:3-5`):** "Students write memories on a
whiteboard in coloured marker; we photograph the board; this turns each photo
into transparent PNG stickers … ready to drop into the senior spread layout."

**Observed use (VERIFIED from photo content, INFERRED as purpose):** the one
real shoot (`photos/`, 56 frames) is a board of get-well messages to a staff
member. That differs from "senior memories". See Q8.

**Constraints that shaped the design (all VERIFIED in code or comments):**
1. **Colour fidelity on a print layout.** Thresholding gives grey halos and
   muddy yellows (`README.md:40-44`), and the halo is "invisible on white and
   screams on black" (`src/wbx/sheet.py:1-6`).
2. **Uncontrolled capture:** a phone, room lighting with a gradient, a glossy
   board throwing ceiling-light reflections, eraser ghosts, a long board shot a
   stretch at a time, so there are no board corners in frame (`src/wbx/board.py:1-15`).
3. **Input from a human operator:** lasso crops saved as screenshots, with
   transparency, black under alpha 0, and low resolution (`src/wbx/io.py:30-46`, `src/wbx/recut.py:1-15`).
4. **No ground truth on real photos.** Hence a synthetic board generated from
   the same physical model (`tests/synthetic.py:1-8`).
5. **Handover to a non-technical designer:** flat folder, provenance CSV,
   byte-identical copies (`src/wbx/album.py:1-10`).

## 2. System architecture

```
photo(s) ─▶ io.load_rgba ─┬─(opaque)──▶ rectify (quad?) ──no quad──▶ board.board_region (mask)
                          └─(RGBA crop)─▶ crop mask (eroded 3 px)            │
                                                                             ▼
             normalize.estimate_background (pass 1, median 41 px @ ≤1200 px, excluded px Telea-inpainted)
                                       ▼
             matte.estimate_alpha  α = max(1 − min(R,G,B), 0.7·S_HSV),  α<0.06 → 0
             deglare: α=0 where min_c(img) > 1.01·min_c(bg)
                                       ▼
             refit: mask (α>0 ∪ chroma>0.08) ∩ (darker than bg on darkest channel), dilate 2 px
             → normalize pass 2 → matte again
                                       ▼
             unmix  ink = clip(1 − (1 − obs)/α)       (straight alpha)
                                       ▼
             clean: per-component — area<150 px OR (peak α<0.35 AND mean S<0.10) → drop
             fill_specular: close glare gaps narrower than 0.6× stroke, inpaint α & ink
             drop_crop_edge: ≥75% in boundary band (6% short side) AND chroma<0.30 → drop
                                       ▼
             stickers.find_quotes: union-find on component bboxes inflated by 0.7× letter height
             (or 1 sticker for a hand crop; or `pick_main` for a close-up) → bleed RGB 3 px → RGBA PNG + manifest.json
side tools:  recut (SIFT 4000 feats, FLANN, Lowe 0.75, RANSAC 5 px, ≥25 inliers) · album (dupe flags) · sheet (QA)
```
Sources: `src/wbx/pipeline.py:92-255`, constants in `pipeline.py:26-50`,
`clean.py:11-18,154-158`, `stickers.py:236-243`, `recut.py:112-121`,
`board.py:24-34`. VERIFIED.

**Data structures:** float32 RGB in [0,1] throughout (`io.py:1-7`). Boolean
masks for crop/board/people/ink. `cv2` label images. Quotes as tuples
`(label,x,y,w,h,area)`. A `Params` dataclass serialised into each
`manifest.json` so every run records its own parameters
(`pipeline.py:26-56, 272-273`). The audit used this to identify a code version
that was never committed (see timeline row 8).

## 3. Engineering decisions (evidence-backed)

| # | Decision | Alternative rejected | Stated reason | Evidence | Status |
|---|---|---|---|---|---|
| D1 | Invert a compositing model per pixel, `obs = α·ink + (1−α)·board` | Threshold dark pixels and copy them | Thresholding grey-fringes pastels. Unmixing puts the recovered colour on the white→ink ray, so a wrong α changes saturation, not hue. | `unmix.py:1-11`, `README.md:40-59` | VERIFIED as implemented. "Hue preserved exactly" is **CONTRADICTED in part**: exact only while the ray stays in gamut. Underestimating α by >~25% clips a channel, giving up to 9.2° hue error (`figures/fig_unmix_wrong_alpha.png`) |
| D2 | Divide out the lighting **per channel** | Luminance-only correction | Also removes the warm/cool cast of room light | `normalize.py:1-9` | VERIFIED (code). No measurement |
| D3 | α from darkness **OR** saturation | Darkness only | Yellow and pastels barely darken the board | `matte.py:40-56` | VERIFIED (code). **Not demonstrated by tests:** yellow core retention is 100% with `sat_weight=0` too, because yellow's blue channel is dark (`data/synthetic_metrics.json` → ablations) |
| D4 | Clean whole **connected components**, never per pixel | Per-pixel "faint & grey" test | A per-pixel test eats the antialiased rim of black text | `clean.py:42-68` | VERIFIED: black text area change <3% asserted (`tests/test_synthetic.py:150-163`) |
| D5 | Ghost test on the **observed** saturation, not the unmixed ink | Saturation of the unmixed ink | Dividing by a small α inflates noise into vivid colour | `clean.py:61-64` | VERIFIED (code) |
| D6 | **Two-pass background**: mask the first pass's ink and refit, inpainting (Telea) rather than flat-filling | Larger median kernel; flat fill | A bigger kernel only moves the failure width. A flat fill over a large area "invents a quote out of glare" | `pipeline.py:186-217`, `normalize.py:206-217` | **VERIFIED, strongest result:** fat-stroke core fill **56.6% → 100%** (`fig_ablations_synthetic.png`). Bare-board refit invents no ink (test passes) |
| D7 | Chroma (max−min, no background needed) added to the refit mask | Trust the first matte alone | Where the estimate sank into dense colour, the matte sees no ink | `pipeline.py:195-210` | Real-photo claim (red quote bg `[0.78 0.55 0.54]→[0.74 0.70 0.67]`, `README.md:185-188`) **UNVERIFIED**: no script or data. Synthetic: no measurable effect (0.6249 vs 0.6244 coverage) |
| D8 | All glare tests on the **darkest** channel | Brightest channel or luminance | Yellow is brighter than the board in R and G. A brightest-channel test deletes yellow. "Added light raises every channel; ink lowers at least one." | `pipeline.py:166-177`, `README.md:196-200` | Reasoning VERIFIED (sound). Guard test passes. **Not demonstrated by the synthetic glare test:** the spot does not matte even with `deglare=False`. The benefit rests on real photos, with no measurement |
| D9 | Reject a board quad with a corner on the frame edge, or edge contrast <0.06 | Accept the largest convex quad | The Otsu fallback "invented a board" from the lighting gradient on a real photo | `rectify.py:24-77`, commit `6be22e6` | VERIFIED: real-photo regression passes. Corner error ≤1.07 px for yaw 0–40°. **No quad found at 50°** (`data/synthetic_metrics.json`) |
| D10 | Merge radius scaled to the **photo's own handwriting**, as 0.7× letter height (75th-percentile component height), with union-find on inflated boxes | Fixed 30 px (v1), then 5× stroke width (v1.1), then mask dilation | A fixed px count is wrong across resolutions. Stroke width "turned out to be a poor proxy … 4.7 to 13.7 stroke widths" | `stickers.py:266-326`, `6be22e6` msg | Mechanism VERIFIED. "4.7–13.7" **UNVERIFIED** (no data). **README:238 CONTRADICTED** (still documents 5× stroke width) |
| D11 | A hand crop is **always one sticker** | Re-split with the merge radius | "Lassoing is the operator saying this is one quote" | `pipeline.py:113-115`, `stickers.py:367-370` | VERIFIED (tests) |
| D12 | Crop-edge litter rule: geometry (≥75% within a 6% boundary band) **and** achromatic | Drop anything touching the edge | Clipped coloured strokes legitimately hug the edge | `clean.py:91-151` | VERIFIED (synthetic: frame removal costs <5% of the writing) |
| D13 | Board **region** (bright ∧ colourless, holes refilled) when no quad is visible | Leave it unmasked | One strip of trim is a single component that bridges every quote | `board.py` | VERIFIED (code). No dedicated test |
| D14 | Re-cut by **SIFT + RANSAC homography** | Template correlation | "A whiteboard is mostly blank, so correlation picks the wrong photo by a hair" | `recut.py:1-15` | VERIFIED: 56/59 reproduced, separation in §8. The correlation comparison itself is **UNVERIFIED** (no data) |
| D15 | Duplicate flag needs shape **AND** ink colour; never deletes | Shape only | "Same-hand writing correlates in shape whatever it says" | `album.py:249-266` | Detector VERIFIED (§8). **On real data the colour veto never fires**: no unrelated pair reached the 0.45 shape threshold |
| D16 | Bleed RGB 3 px under α=0 | Leave black under transparency | Layout tools interpolate RGB across the edge, giving a dark halo | `stickers.py:459-480` | VERIFIED (code) |
| D17 | Straight (non-premultiplied) alpha; album copies bytes, never re-encodes | — | "What is sent is what was reviewed" | `io.py:389-392`, `album.py:233-234` | VERIFIED (byte-identity test passes) |

## 4. Development timeline
See `DEVELOPMENT_TIMELINE.md`. In short: v1 full pipeline (`b4bde13`, 09-08) →
v1.1 two real-photo fixes (`6be22e6`, 09-08) → shoot (09-25, 56 frames) → 59 hand
crops plus an uncommitted intermediate version (09-26) → v2 real-photo hardening,
re-cut, handover (`0a999a4`, 09-29).

## 5. Failures and debugging (the strongest material)

| Failure (observed on real data) | Root cause given | Fix | Evidence |
|---|---|---|---|
| F1. Board "invented" on `exampletext.jpeg`: quad's top = image border, bottom cut diagonally, ¼ of board cropped | Otsu fallback segmented the **lighting gradient** | Corner-on-frame and edge-contrast rejection | `6be22e6`; test `test_frame_filling_board_is_not_rectified` passes |
| F2. "Hello" split into 5 stickers on an iPad photo | Fixed 30 px merge radius vs ~3264 px-wide photo | Auto radius; later letter-height rule | `6be22e6`; `stickers.py:281-295` |
| F3. Pre-cropped PNGs matte as a solid black blob | Editors write RGB=0 under α=0, 20–40% of a crop | Read and erode the alpha mask; exclude it from sampling | `0a999a4`, `io.py:327-358`. "20–40%" **VERIFIED** in this audit: median 22.8%, 41/59 crops in range |
| F4. Fat marker exports hollow (bright rim, empty middle) | Median rejects only ink that is thin relative to its 41 px window | Two-pass refit (D6) | Synthetic 56.6% → 100%; **paired regression test proves the failure exists without the fix** (`test_fat_marker_is_hollow_without_the_refit`) |
| F5. Reflections become pale film; dense colour bleaches itself; yellow deleted by a glare rule | Model has no additive-light term | Darkest-channel tests, chroma-masked refit, `fill_specular` | `0a999a4`, `README.md:172-205`. Real-photo effect UNVERIFIED |
| F6. Trim/frame bridges every quote into one blob | Long board shot in stretches, no quad | `board.py` region mask + crop-edge rule on the picture edge | `0a999a4`; `clean.py:120-127` |
| F7. Screenshot crops too low-res for print (1.86 in at 300 dpi) | Crops made from the screen | SIFT re-cut from originals → 4.25 in | §8 |
| F8. **Unresolved:** 2 stickers with glare-bleached strokes; sticker_10 "signature unrecoverable" | Specular saturation destroys the ink signal | None; documented as a capture problem | `out/send/NOTES.txt` |

Unsuccessful attempts recorded: larger median kernels (rejected, `README.md:80-82`),
flat fill of excluded pixels (rejected, `normalize.py:211-217`), stroke-width
merge radius (replaced, `stickers.py:284-291`), brightest-channel glare test
(rejected, `README.md:196-200`), correlation for re-cutting (rejected, `recut.py:96-99`).
These are claimed in comments. Only the stroke-width rule survives in git
history (`6be22e6`).

## 6. Mathematical content

1. **Image formation (normalized):** `I/B = α·k + (1−α)·1`, where B is the per-channel board estimate (`normalize.py:244-247`).
2. **Unmixing:** `k = 1 − (1 − Î)/α`, so `1 − k = (1 − Î)/α`. A wrong α scales the vector `1 − k` but keeps its direction, so hue is kept and chroma scales by `α_true/α_guess`, **until a component of k leaves [0,1] and is clipped** (`unmix.py:321-325`; verified numerically in `fig_unmix_wrong_alpha`).
3. **Alpha estimator:** `α = max(1 − min_c Î_c, w_s · S_HSV(Î))`, w_s = 0.7, floored at 0.06 (`matte.py:279-301`).
4. **Specular test:** `min_c I_c > (1+ε)·min_c B_c`, ε = 0.01 → α = 0 (`pipeline.py:177`).
5. **Stroke width:** `2·median(DT(mask))` over interior pixels, from the L2 distance transform (`stickers.py:257-263`).
6. **Letter height:** P75 of component bbox heights with area ≥150 px. Merge slack `r = clip(0.7·h, 10, 600)` (`stickers.py:266-295`).
7. **Grouping:** two boxes join if their r-inflated rectangles overlap. Transitive closure by union-find with path halving (`stickers.py:298-326`). O(n²) in component count for the touch matrix.
8. **Dominant hue:** circular mean `atan2(Σw·sinθ, Σw·cosθ)` over core pixels (α ≥ 0.5·max) with w = α·chroma. Reported as `None` if the weighted chroma is <0.12 (`stickers.py:483-511`).
9. **Homography:** RANSAC (5 px reprojection) on Lowe-ratio (0.75) SIFT matches. Scaled back to full size with `diag(1/s, 1/s, 1)·H`, s = 0.5 (`recut.py:207`).
10. **Duplicate score:** mean product of z-scored 48×32 alpha thumbnails (a Pearson correlation), plus the L2 distance of α-weighted core RGB (`album.py:295-348`).
11. **Synthetic perspective:** a pinhole projection of a yawed plane, `x = f·X·cosθ/(D − X·sinθ)` (`tests/synthetic.py:173-184`).

## 7. Validation: what was actually tested

| Kind | What | Result in this audit |
|---|---|---|
| Synthetic, ground truth | 1600×1000 board from the forward model: gradient 235→255, vignette, Gaussian glare, 6 words in known RGB at known coverage, σ=1.5/255 noise (`tests/synthetic.py`) | see §8 |
| Unit/regression suite | 37 tests (`tests/test_synthetic.py`) | **35 pass, 2 fail** (Anaconda Py 3.13.5, OpenCV 5.0.0, NumPy 2.1.3, 64.9 s) |
| — failure 1 | `test_real_handwriting_is_one_sticker`: radius 319 px vs expected 40–120 | **Genuine stale test.** Also in your `.pytest_cache/lastfailed`. The sticker count (1) is still correct |
| — failure 2 | `test_cli_extract_and_sheet`: click `TypeError` | **Audit environment only** (typer 0.9 here vs 0.27.2 in `uv.lock`). Not a code defect |
| Real-photo regressions | bare board → 0 stickers; frame-filling board not rectified; teal hue 160–195° | pass (4 tests use `samples/`, 2 iPad photos) |
| Real shoot, end-to-end | 56 photos / 59 crops → 56 hi-res stickers → 47 delivered | VERIFIED counts. **No quantitative quality metric exists on real photos** (no ground truth) |
| Never tested | `--remove-people` (MediaPipe not installed), HEIC path, `--bg-method close`, `board.py` (no direct test), 50° yaw, two overlapping quotes in different colours (known limitation, `README.md:292-296`) | — |

## 8. Results (strongest, with conditions)

Full table with sources: `METRICS.csv`.

**Real data, VERIFIED by re-running the shipped code:**
* **Re-cut:** 56/59 crops matched among 56 originals. Median linear gain **2.24×**
  (range 2.19–2.24). Misses had best scores of 8, 0 and 7 inliers (threshold 25).
  Best-photo inliers for matched crops: 67–585 (median 191.5).
  `scripts/s04_recut_repro.py` → `data/recut_results.csv`. Wall time 276 s for
  59 crops × 56 photos on this machine.
  **The README/docstring claim "true source scored hundreds against single digits" is CONTRADICTED as stated.**
  The full 59×56 inlier matrix (`data/recut_inlier_matrix.csv`,
  `figures/fig_recut_inlier_separation.png`) shows that the shoot's frames
  overlap. A median of 5 photos per crop clear the 25-inlier threshold. The
  runner-up photo scores a median of 105 inliers (max 308), and in the closest
  case it is within 3% of the best. Scores form a continuum through the threshold
  with no gap. What *is* true: 87% of the 3,304 crop×photo scores are under 10,
  and all 56 accepted homographies give a consistent 2.19–2.24× scale, which is
  evidence that the matches are geometrically sane, not lucky. RANSAC is
  stochastic: one miss scored 7 in the shipped-path run and 0 in the matrix run.
* **Lasso crops:** the fraction cleared to α = 0 has a median of 22.8%
  (IQR 20.2–28.1%, range 13–80%, n = 59). 69% of crops fall in the README's
  "20–40%" range. VERIFIED (`data/crop_cleared_fraction.json`).
* **v1 failures reproduced** by re-running `b4bde13` code (`git archive`, read-only)
  on `exampletext.jpeg`. v1 returns quad `[[0,0],[3263,0],[3247,1383],[39,1779]]`:
  the top edge *is* the image border and the bottom cuts diagonally, exactly as the
  `6be22e6` message says. Unrectified, it splits "Hello" into **5** stickers.
  Current code: no quad, 1 sticker, radius 319 px (`data/v1_vs_current_exampletext.json`,
  `figures/fig_v1_failures_exampletext.png`). VERIFIED.
* **Print size:** median sticker width 557.5 px → 1,275.5 px = **1.86 → 4.25 in at
  300 dpi** (`out/stickers/index.csv`, `out/stickers_hires/index.csv`).
  README says 1.9 → 4.1. That is close but not identical (minor discrepancy, CONTRADICTED on the second figure).
* **Duplicate detector** (`scripts/s02_duplicates.py`): 1,653 and 1,540 pairs.
  8 flagged in each run, **precision 8/8, recall 8/9**, against the handover cull.
  Each run misses a different pair. No non-duplicate pair exceeded correlation 0.33.
* **Real-photo regressions:** bare iPad board → 0 ink pixels. Teal recovered
  at 177.3° (`out/exampletext/manifest.json`).

**Synthetic (ground truth known), VERIFIED** (`data/synthetic_metrics.json`):
* Bare-board leak 0.0% (test bound 0.5%). 6/6 words → 6 stickers.
* Per-word stroke IoU **0.950–0.974**. Hue error **0.45–2.41°** (test bound 8°).
* Rectification max corner error **≤1.07 px** for yaw 0–40°. **Fails (no quad) at 50°.**
* Fat stroke filled **56.6% → 100%** with the refit. Dense red coverage
  recovered 68.1% → 79.5% of true.

**Runtime** (new measurement, `scripts/s05_runtime_and_stages.py` → `data/runtime.json`;
Windows 11 laptop, Python 3.13.5, OpenCV 5.0.0; median of 3 runs after warm-up):
4.8–7.0 s per full photo (7.3–8.0 MP, n = 7 photos) and 1.1–2.7 s per re-cut crop
(1.3–3.1 MP, n = 4). No runtime figure exists anywhere in the repository. Before
quoting these numbers, state the hardware and note that no optimisation was
attempted. Running all 37 tests takes 64.9 s.

## 9. Limitations and contradictions

1. **CONTRADICTED:** `README.md:238` says the merge radius is "5× median stroke
   width". Code is 0.7× letter height (`stickers.py:16`). The test at
   `tests/test_synthetic.py:319` encodes the old rule and fails. Commit `6be22e6` says
   "On the sample it picks 60px"; it now picks 319–320 px.
2. **CONTRADICTED (partial):** "hue is preserved exactly" (`README.md:57`): only in gamut (§6.2).
3. **Provenance gap:** 14/47 delivered stickers come from an uncommitted process
   ("film-removal pass", rotation, coverage lift). The `out/cropped` run came from
   an uncommitted code version.
4. **UNVERIFIED claims with no data in the repo:** red-quote background values;
   4.7–13.7 stroke widths; correlation "wins by nothing"; "a shape-alike pair …
   0.80 apart" (no non-duplicate pair even reaches the shape threshold in either album run).
   **CONTRADICTED:** "hundreds against single digits" for re-cut matching (§8).
   The duplicate docstring's "0.58–1.00 / unrelated 0.34" holds only if crops
   010559/010602 are different quotes, but the handover cull treats them as the same (Q7).
5. **Untested features:** the saturation cue and deglare show no effect on the
   synthetic fixtures; `board.py` has no direct test; people masking and HEIC were never run.
6. **No real-photo accuracy metric.** Quality on the shoot was judged by eye
   (contact sheet). 2 stickers have unrecovered glare damage (NOTES.txt).
7. **Known v1 limitation:** overlapping quotes in different colours merge (`README.md:292-296`).
8. **Rectification envelope:** fails at 50° yaw (synthetic). The real shoot never used rectification (all `rectified: false`).
9. **`main` lacks v2.** v2 lives only on an unmerged branch whose remote tracking ref is gone.
10. **Privacy:** real photos and stickers contain third parties' names and handwriting.

## 10. Is this "novel"?
No new algorithm. The engineering value is in **diagnosing physically grounded
failure modes on real photos** (glare as additive light, fat strokes poisoning
their own background, editor-cleared alpha, frame trim bridging components) and
fixing each with a reason that can be stated and tested. A paired test shows the
failure without the fix (F4). Present it that way.
