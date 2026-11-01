# Project Card Draft: Whiteboard Ink Extraction (`wbx`)

> Everything here is limited to what the repository supports. Phrases in
> **[brackets]** depend on your answers in `QUESTIONS_FOR_ME.md` (mainly Q1, Q4,
> Q8, Q11). Do not drop the brackets until you have answered them.

## A. Title options
1. **Un-compositing Marker Ink: Recovering True-Colour Handwriting from Glossy-Whiteboard Photos**
2. Inverting `obs = α·ink + (1−α)·board`: A Classical-Vision Pipeline from Phone Photos to Print-Ready Stickers
3. Whiteboard → Sticker: Physically-Modelled Ink Matting, Glare Rejection and SIFT Re-Cutting
4. When Real Photos Break the Model: Hardening an Ink-Unmixing Pipeline Against Glare, Fat Markers and Hand Crops
5. `wbx`: Turning 56 Overlapping Board Photos into 47 Print-Resolution Handwriting Stickers

## B. Five strongest contributions (ranked by significance × evidence)
1. **Physical-model ink unmixing instead of thresholding.** Per-channel lighting
   division, then `k = 1 − (1−Î)/α`. A wrong α changes saturation, not hue, as
   long as the colour stays in gamut. Synthetic hue error 0.45–2.41°, stroke IoU
   0.95–0.97. *Evidence: strong (synthetic). The method is standard compositing
   algebra applied well, not a new algorithm.*
2. **Two-pass, self-masking background estimate.** The first matte masks the ink,
   Telea inpainting fills the gap, and the lighting is fitted again. Fat-marker
   core fill goes from **56.6% to 100%**. A paired test proves the failure
   exists without the fix. *Evidence: strongest single result.*
3. **Real-photo failure diagnosis and regression.** Two v1 bugs (a hallucinated
   board quad; "Hello" split into 5) were reproduced in this audit from the
   original commit, fixed, and locked with real-photo tests.
4. **Resolution recovery by feature matching.** 59 hand-lassoed screenshot crops
   were located among 56 overlapping originals (SIFT + RANSAC homography):
   **56/59 matched, 2.24× linear gain**, so median print size went from 1.86 in
   to 4.25 in at 300 dpi. Reproduced in this audit.
5. **Operator-aware input handling and handover:** alpha-mask-aware loading (the
   median crop is 22.8% cleared to black), the crop-edge litter rule, board-region
   masking, and a duplicate detector that needs both shape and colour to match
   (8/8 precision, 8/9 recall on the real shoot).

## C. Three most revealing challenges
1. **A fat marker poisons its own background.** A median filter only rejects ink
   that is thin relative to its window. Inside a chisel stroke the "board"
   estimate becomes the ink, so the letter exports hollow. A bigger kernel only
   moves the width at which this happens. *Fix:* mask the first pass's ink, and
   inpaint rather than flat-fill. A flat fill invented quotes on a bare board.
   *Result:* 56.6% → 100%.
2. **Glare is light added, and the model has no term for it.** A reflection reads
   as *less ink*. Every glare test compares the **darkest** channel, because yellow
   marker is brighter than the board in R and G and only gives itself away in B.
   *Honest status:* the reasoning is sound and a guard test keeps yellow intact,
   but the synthetic glare fixture does not measure any benefit. The real-photo
   evidence is visual only.
3. **The real shoot didn't look like the design input.** 56 overlapping close-ups
   of a long board meant no corners, trim in frame, hand crops with black under
   transparency, and screenshot resolution. *Fix:* board-region masking, alpha-aware
   loading, and SIFT re-cut back to camera pixels. *Honest status:* 14 of the 47
   delivered stickers still needed hand correction outside the code.

## D. Technical description (≈300 words)
**[For a get-well / senior-memory board, Q8]**, classmates wrote messages on a
whiteboard in coloured marker. The goal was to turn phone photos of that board
into transparent PNG "stickers" of each message for a print layout, with the
marker colours intact. Thresholding fails here: photographed marker is
translucent and washed out, so thresholded edges carry a grey fringe that shows
on dark backgrounds and turns yellows muddy.

The pipeline (Python, OpenCV, NumPy; 1,830 lines plus 896 lines of tests) models
each pixel as translucent ink over an unevenly lit board. It estimates the
lighting per channel, divides it out, and estimates coverage α from darkness or
saturation. It then inverts the compositing equation, `ink = 1 − (1 − observed)/α`,
so the recovered colour stays on the line from white through the true marker
colour. Cleanup works on whole connected components, so it never erodes
antialiased letter edges.

Real photos broke the first version twice: it warped a frame-filling board to a
quad hallucinated from the lighting gradient, and it split "Hello" into five
stickers. Both were fixed and locked in with regression tests. The real shoot
(56 overlapping photos, 59 hand-lassoed crops) exposed more. Fat chisel strokes
dragged their own background estimate into the ink and came out hollow; a
two-pass, ink-masked refit raised core fill from 57% to 100% on a synthetic
test. Ceiling glare breaks the model's assumptions, so glare tests compare the
darkest channel to avoid deleting yellow ink. Screenshot crops were too small
to print, so each was relocated in its original photo by SIFT feature matching
and a RANSAC homography: 56/59 recovered at 2.24× resolution.

Validation used a synthetic board generated from the same physical model:
hue error ≤2.4°, stroke IoU ≥0.95, 35/37 tests passing. 47 stickers were
delivered; 14 needed manual correction for unrecoverable glare. **[Built with
Claude Code as an AI pair-programmer; I … (Q1, Q3)].**

## E. Layout (6 figures)
```
┌───────────────────────── title + 1-line objective ─────────────────────────┐
│ [Fig 2] pipeline stages, full width, equations on arrows                   │
├──────────────────────────────┬─────────────────────────────────────────────┤
│ [Fig 1] v1 failure → fix     │ [Fig 4] fat marker: hollow → solid + bars   │
├──────────────────────────────┼─────────────────────────────────────────────┤
│ [Fig 3] wrong-α: hue vs chroma│ [Fig 6] re-cut before/after (or Fig 7 plot) │
├──────────────────────────────┴─────────────────────────────────────────────┤
│ [Fig 5] annotated raw frame (blurred) │ key numbers │ "what I did" box      │
└────────────────────────────────────────────────────────────────────────────┘
```
Captions:
* **Fig 2.** "Each pixel is modelled as translucent ink over an unevenly lit
  board. Divide out the lighting, estimate coverage α, solve for the ink. Shown
  on the pipeline's own debug output."
* **Fig 1.** "Two bugs found by the first real photo, reproduced from commit
  `b4bde13`. A board 'found' in the lighting gradient (left); a fixed 30 px merge
  radius splitting one word into five stickers (centre). Fixed by rejecting quads
  with corners on the frame and scaling the radius to letter height."
* **Fig 4.** "A median background rejects only ink thinner than its window.
  Inside a chisel stroke it becomes the ink, and the letter goes hollow. Masking
  the first pass's ink and refitting fills 100% of the core, against 57% with one pass."
* **Fig 3.** "Unmixing keeps hue exact while the recovered colour is in gamut. A
  wrong α trades saturation instead, which is why α estimation was tuned for
  clean edges and the colour left to the algebra."
* **Fig 6.** "Hand-made crops were screenshots. Each was found in its original
  photo by SIFT features and a RANSAC homography and cut again: 56/59, 2.24×."
* **Fig 5.** "The real input: ceiling reflections, aluminium trim, no visible
  corners. Each one breaks an assumption the model makes."

## F. Five talking points (reasoning, not vocabulary)
1. "Glare is light *added*; ink takes light *away*. So I ask whether a pixel is
   brighter than the board on its **darkest** channel. Yellow marker is brighter
   than the board in red and green, and a brightest-channel test deletes it."
2. "Making the median kernel bigger doesn't fix hollow fat strokes. It just moves
   the stroke width at which they go hollow. The fix had to remove the cause,
   which was the stroke contaminating its own background estimate."
3. "A test that a fix works proves little without a test that the failure exists
   without it. The fat-marker test comes as a pair."
4. "A whiteboard is mostly blank, so correlation scores are dominated by blank
   board. I matched features instead. Then the audit showed something I hadn't
   accounted for: overlapping frames mean several photos legitimately contain
   each quote."
5. "Fixed pixel thresholds don't transfer between a 2 MP screenshot and a 12 MP
   photo. The merge radius scales with the handwriting in each photo. It went
   through three versions before letter height worked."

## G. Hard reviewer questions, with evidence-backed answers
**Q: How do you know the colours are right on real photos, with no ground truth?**
A: On real photos I don't, quantitatively. Hue accuracy (≤2.4°) was measured on
a synthetic board built from the same forward model, which is circular for
model-mismatch errors like glare. On real data the evidence is a teal marker
recovered at 177° (test band 160–195°) and visual QA on the dark-background
contact sheet. A calibration card in frame would be the honest next step.

**Q: Isn't `ink = 1 − (1−obs)/α` just alpha-matting algebra? What's new?**
A: Yes, it's the "over" compositing operator inverted. The engineering is
everything that makes its assumptions hold on real photos: a board that reads
1.0 everywhere (two-pass, inpainted refit), α that excludes glare (darkest-channel
test), and α that sees yellow. I also know where it breaks: if α is
underestimated by more than ~25%, the colour leaves the gamut and hue drifts up to 9°.

**Q: Your re-cut threshold is 25 inliers. How did you choose it, and how close are the alternatives?**
A: The full 59×56 inlier matrix shows the frames overlap. A median of 5 photos
per crop clear 25, and the runner-up is a median 105 inliers. So the threshold
filters out crops that appear in no photo; it does not separate right from wrong
photos. All 56 accepted matches gave a consistent 2.19–2.24× scale, which is the
best evidence the homographies are correct. The three misses were fragments too
small to carry features.

## H. Claims I must verify before submission
- [ ] **Authorship of each idea** in B/C/F, from the two Claude session transcripts (Q1–Q3). Rephrase anything you didn't originate as "directed and validated".
- [ ] **Who drew the 59 crops, made the cull, and made the 14 hand edits** (Q4–Q6).
- [ ] **Purpose and outcome.** Get-well board vs senior spread; was the output used? (Q8, Q9)
- [ ] **Consent** for any handwriting or name shown (Q11). Default to "life of pi" / "Hello" figures only.
- [ ] **Hardware** behind "5–7 s per photo". Re-time on the machine you'll name.
- [ ] Don't quote "hundreds against single digits" (contradicted) or "4.1 in" (measured 4.25).
- [ ] Don't quote "hue preserved exactly" without "in gamut".
- [ ] Don't claim the saturation cue or deglare were *measured* to help. Synthetic ablations were null.
- [ ] Fix or disclose: README:238 (stale merge-radius text) and the failing test at `tests/test_synthetic.py:319`.
- [ ] Merge/push v2 if reviewers will see GitHub; `main` lacks it (Q12).
- [ ] Credit OpenCV (SIFT, RANSAC, Telea inpainting) and Claude Code on the card.
