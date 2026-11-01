# Whiteboard Quote Extractor (`wbx`)

Students write their own thoughts and quotes on a whiteboard in colored marker; I photographed the board; then processed each photo via this program to turn each message into a transparent PNG "sticker" of the handwritten note. Each of the notes still contained their original colors and this was made because I felt like it would be more personal if people wrote messages by hand rather than with just text in Adobe.

```bash
uv sync
uv run wbx extract photos/*.jpg -o out
uv run wbx sheet out          # QA contact sheet
```

Quotes could be extracted by below:

```bash
uv run wbx extract crops/*.png -o out
```

I used Screen Snip to lasso crop a bunch of the quotes so sometimes the segmentations were great, but the image resolution suffered immensely.

```bash
uv run wbx recut crops/*.png -p photos -o recut
uv run wbx extract recut/*.png -o out
```

Per photo you get `out/<photo>/`: one `quote_NN.png` per quote, a full-board
`board.png`, and a `manifest.json`.  All PNGs are RGBA with **straight
(non-premultiplied) alpha**.

## How it works

When I tried thresholding dark pixels and copying them out, the pastel color of markers were ruined and washed out. You get all sorts of artifacts and noise so I implemented a fix where every pixel is modeled as a transluscent ink that is composited over the board.


```
observed = a * ink + (1 - a) * white
```

Normalize, then solve per pixel for coverage `a` and the actual  ink color:

```
ink = 1 - (1 - observed) / a
```

A misestimate in the `a` sometimes costs saturation quality, but it never makes a pastel gray which is great. That is why I tuned the alpha estimation for clean edges and the color is left for algos.

Runtime order:

| step | module | what it does |
| --- | --- | --- |
| 0 | `people.py` | optional person mask (`--remove-people`) |
| 1 | `rectify.py` | find the board quad, perspective-warp to frontal |
| 2 | `normalize.py` | estimate per-channel board lighting, divide it out (twice: see below) |
| 3 | `matte.py` | soft alpha from darkness **and** saturation |
| 4 | `unmix.py` | solve for ink color |
| 5 | `clean.py` | alpha floor, despeckle, eraser-ghost removal |
| 6 | `stickers.py` | merge strokes into quotes, crop, export RGBA + manifest |
| 7 | `sheet.py` | QA contact sheet on light and dark checkerboards |
| - | `board.py` | mask the board face when the photo shows more than the board |
| - | `recut.py` | re-cut hand-made crops from the originals, at full resolution |
| - | `album.py` | flatten a run into one folder of copies for handover |

I double ran step two because a median blur only rejected ink that is thin relative to its window, so when I rerun it it allows for a better final result. Explanation: A median blur only rejects ink that is thin relative to its
window, so inside a fat marker the "board" estimate sinks to the ink itself,
the stroke normalizes to 1.0 and mattes as a hole: the letter exports as a
bright outline around nothing. Kernel sizing only increases marker width. So I decided that the first pass is to create a mask of where to not sample, then the lighting is refitted, excluded pixels are inpainted, allows a clearer image on second pass.

The saturation term in step 3 is what catches the tricky buggers: yellow, orange and pale markers
which barely darken the board at all. The cleanup in step 5 works on whole
connected components.

## Pre-cropped quotes

A photo can also arrive already lassoed down to one quote, saved as a PNG with
everything outside the loop transparent.  That input is detected from the alpha
channel and handled differently in three ways, no flags needed:

* **the cleared pixels are not board.**  Editors write pure black under alpha 0,
  which is 20-40% of a typical crop; read alpha-blind it would drag the
  background estimate dark and matte as one solid stroke.  The mask is eroded a
  few pixels first, because a boundary pixel that is half board and half
  cleared-to-black mattes as ink and draws the outline of the crop around the
  sticker.
* **no rectification.**  A crop is not a board photo; there is no quad to find.
* **one sticker out, always.**  Lassoing *is* the operator saying "this is one
  quote", so the merge step is skipped rather than given a large radius.  The
  gap that separates two quotes on a board is the same gap a wide-set signature
  leaves, and re-splitting a crop can only be wrong.  `--no-single-quote` turns
  that off for a crop that really does hold several.

One extra cleanup runs on cropped input: a loop drawn around a quote near the
edge of the board catches a sliver of the board's own frame, a marker tray or a
desk edge, and that is as dark and as solid as ink, so nothing else rejects it.
A component is dropped when it lies almost entirely in a band just inside the
crop boundary **and** is achromatic -- it runs along the cut rather than merely
touching it, which is what tells clipped furniture from a letter the lasso
sliced through, and the chroma test keeps the clipped colored strokes that
genuinely do hug the edge.  See `CROP_EDGE_*` in `clean.py`.

## Photos that show more than the board

`rectify.py` wants four visible corners and declines without them, which is the
right answer for a long board photographed a stretch at a time -- there is no
quad.  Those frames still arrive with the wall, the aluminium trim and whatever
is past the end of the board in shot, all of it darker than the board and all of
it mattes as confidently as ink.  The damage is not cosmetic: one strip of trim
running the width of the photo is a single connected component that bridges
every quote into one blob, and no merge radius can undo that.

So `board.py` finds the board *region* instead of its corners -- the large,
bright, colorless area, with the writing filled back in as holes -- and the
rest is masked exactly like the cleared surround of a hand-made crop.  What the
mask cannot resolve, a hairline of trim a few pixels wide, is caught by the same
crop-edge rule, which counts the edge of the picture as a boundary and not just
the edge of the mask.

## Re-cutting hand-made crops

Deciding where one quote ends is the part a person does in a second and a merge
radius argues about.  When that work has already been done by hand, `wbx recut`
keeps it: each crop is located in its original photo by SIFT features and a
RANSAC homography, the crop's outline is warped into the photo, and the region
is cut again at full resolution.  On a real shoot that recovered 56 of 59 crops
at 2.2x linear -- the three misses were fragments too small to carry features --
and stickers went from 1.9 to 4.1 inches at 300dpi.

Feature matching rather than correlation, because a whiteboard is mostly blank:
correlation scores are dominated by the blank and the true source photo wins by
nothing, while inlier counts separated hundreds against single digits.

## Handing the stickers over

`wbx album OUT -o DIR` copies every sticker into one flat folder as
`sticker_01.png`, `sticker_02.png`, ... with an `index.csv` giving each one's
source photo, size and ink color -- the provenance the per-photo directory
names were carrying.  The PNG bytes are copied, never re-encoded, so what is
sent is what was reviewed.  `wbx sheet DIR` works on that folder too.

The index has one more column.  Lassoing the same quote twice is the normal
accident on a shoot, so stickers that look like an earlier copy are flagged in
`looks like` -- a match needs both a matching silhouette **and** matching ink,
because same-hand writing correlates in shape whatever it says.  Nothing is
ever dropped; it is a hint for whoever culls the folder.

## Glare

A whiteboard is glossy, so the ceiling lights reflect off it.  That breaks the
model everything else rests on: `observed = a*ink + (1-a)*board` has no term for
light added on top, so a reflection is read as *less ink* rather than more light.

Three things follow, and all three are handled where the background is
estimated rather than patched up afterwards:

* **The ink must not be sampled as board.**  The refit masks the first matte's
  ink, but that matte came from the estimate being corrected -- where the
  estimate has sunk into a dense patch of color, the matte sees no ink there
  and the refit samples it again.  Chroma needs no background at all, so
  colored marker is masked whatever the estimate believes.  On a red quote
  this moved the estimated "board" under the strokes from `[0.78 0.55 0.54]`
  -- visibly red -- to `[0.74 0.70 0.67]`, and the recovered ink from washed-out
  pink to the red in the photograph.
* **A reflection must not be masked as ink.**  It is board, brightly lit; mask
  it and the estimate under it gets interpolated from the glare around it, and
  the ordinary board beside it then mattes as a gray cloud.
* **A reflection must not matte.**  The darkness cue already reads zero on it,
  but the saturation cue does not, and a highlight carries enough tint to come
  out as a pale film.

All three tests ask whether a pixel is brighter than the board, and all three
ask it of the **darkest** channel.  Yellow marker is brighter than the board in
red and green and gives itself away only in blue; asked of the brightest
channel, every one of these rules deletes yellow writing entirely.  Added light
raises every channel, ink lowers at least one.

What none of this recovers is a reflection strong enough to bleach a stroke to
the board's own brightness.  Nothing about the ink survives there to recover,
and `fill_specular` interpolates across the gap rather than inventing color for
it -- a reader forgives pale handwriting and does not forgive a hole.

## Reading the contact sheet

`wbx sheet OUT` renders every sticker twice, on a light checkerboard and a dark
one.  **The dark half is the point.**  Naive extraction leaves a gray fringe of
half-transparent board color that is invisible on white and screams on black.

Markers really are translucent, so a sticker composited over a dark background
is genuinely darker than the same sticker on white.  That is the ink, not a
halo; a halo is gray and desaturated, and follows the stroke as an outline.

## Tuning

Start from `--debug`, which writes `OUT/<photo>/debug/`: `01_rectified`,
`02_background`, `03_normalized`, `04_alpha`, `05_ink`,
`06_components_overlay`.  Then:

| symptom | flag |
| --- | --- |
| pale strokes missing | raise `--sat-weight` (default 0.7) |
| board noise surviving | raise `--alpha-floor` (0.06) or `--min-area` (150) |
| thick letters hollow, bright rim only | `--bg-method close` (the refit is already on) |
| two quotes merged into one sticker | lower `--merge-radius` |
| one quote split into several stickers | raise `--merge-radius` |
| stray marks exported as quotes | raise `--min-quote-area` (800) |
| a crop really does hold two quotes | `--no-single-quote` |
| eraser ghosts surviving | raise `--ghost-max-alpha` (0.35) / `--ghost-max-sat` (0.10) |
| faint real strokes eaten as ghosts | lower those two |
| strokes bleeding into `02_background` | `--bg-method close` (fat-marker fallback) |
| board quad missed or wrong | `--corners x1,y1,...` or `--no-rectify` |
| design wants punchier color | `--boost-sat 0.3` (off by default, on purpose) |

`--merge-radius` defaults to **5x the median stroke width measured in that
photo**, not a fixed pixel count.  A fixed count cannot be right: the same board
at 2MP and 12MP needs different numbers, as do a fine-tip and a chisel-tip
marker.  The value actually used is recorded in each `manifest.json`.

`--no-bleed` turns off the few-pixel extension of ink color underneath fully
transparent pixels.  Leave it on: it changes nothing a correct compositor draws,
but layout tools that scale a sticker interpolate RGB across the alpha edge, and
black underneath the transparent pixels shows up as exactly the dark halo we are
avoiding.

## Optional extras

```bash
uv sync --extra heic     # iPhone HEIC/HEIF photos
uv sync --extra people   # --remove-people, MediaPipe selfie segmentation
```

Neither is needed for JPEG/PNG photos of an empty board; the tool exits with an
instruction rather than a traceback when one is missing.

## Tests

```bash
uv run pytest
```

Photos in `samples/` are used as regressions too, and skipped when absent.

`tests/synthetic.py` generates a whiteboard photo in code -- lighting gradient,
vignette, glare blob, six words in known ink colors at known coverage, sensor
noise -- so the pipeline can be checked against ground truth without waiting for
shoot day.  The suite asserts background cleanliness, hue preservation within 8
degrees, stroke IoU, ghost and speck removal, that cleaning does not erode black
text, one sticker per quote, no gray halos, and rectification of a board yawed
25 degrees to within 3 px.  Keep it green through every tuning change.

Against the real photos it asserts that the bare board yields zero stickers,
that a board with no visible corners is left unrectified rather than warped to a
guessed quad, that real handwriting comes out as one sticker, and that the teal
marker comes back teal.

## Capture tips for shoot day

This matters as much as the code.

- Shoot straight on, board filling the frame, no flash, angled away from windows.
  If the board fills the frame completely its corners are not visible, so the
  tool skips rectification and says so -- that is correct, not a failure.
- Fresh markers, chisel tip, write big.
- Leave a hand-width gap between quotes -- sticker merging assumes gaps.
- Wipe the board properly first.  Ghosts are removable, but clean is cleaner.
- One photo per board is fine; the tool batches multiple photos.

## Known v1 limitation

Two overlapping quotes in different colors merge into a single sticker.
Splitting them by hue clustering inside the merged component is a stretch goal,
along with `--group-by-color` and a small drag-and-drop web UI.
