# Whiteboard Quote Extractor (`wbx`)

Students write memories on a whiteboard in coloured marker; we photograph the
board; this turns each photo into transparent PNG "stickers" of the handwriting
with the marker colours intact, ready to drop into the senior spread layout.

```bash
uv sync
uv run wbx extract photos/*.jpg -o out
uv run wbx sheet out          # QA contact sheet
```

Per photo you get `out/<photo>/`: one `quote_NN.png` per quote, a full-board
`board.png`, and a `manifest.json`.  All PNGs are RGBA with **straight
(non-premultiplied) alpha**.

## How it works

Thresholding the dark pixels and copying them out ruins pastels: photographed
marker is washed out, edges blend into the board, and you get grey halos and
muddy yellows.  Instead every pixel is modelled as translucent ink composited
over the board:

```
observed = a * ink + (1 - a) * white
```

Normalize so the bare board reads 1.0, then solve per pixel for coverage `a` and
the true ink colour:

```
ink = 1 - (1 - observed) / a
```

The recovered colour always lands on the ray from white through the real marker
colour, so **the hue is preserved exactly**.  A misestimated `a` costs some
saturation; it never drags a pastel toward grey.  That is why alpha estimation
is tuned for clean edges and the colour is left to the arithmetic.

Runtime order:

| step | module | what it does |
| --- | --- | --- |
| 0 | `people.py` | optional person mask (`--remove-people`) |
| 1 | `rectify.py` | find the board quad, perspective-warp to frontal |
| 2 | `normalize.py` | estimate per-channel board lighting, divide it out |
| 3 | `matte.py` | soft alpha from darkness **and** saturation |
| 4 | `unmix.py` | solve for ink colour |
| 5 | `clean.py` | alpha floor, despeckle, eraser-ghost removal |
| 6 | `stickers.py` | merge strokes into quotes, crop, export RGBA + manifest |
| 7 | `sheet.py` | QA contact sheet on light and dark checkerboards |

The saturation term in step 3 is what catches yellow, orange and pale markers,
which barely darken the board at all.  The cleanup in step 5 works on whole
connected components, never per pixel: a per-pixel "faint and grey" test would
eat the antialiased rim of every black letter.

## Reading the contact sheet

`wbx sheet OUT` renders every sticker twice, on a light checkerboard and a dark
one.  **The dark half is the point.**  Naive extraction leaves a grey fringe of
half-transparent board colour that is invisible on white and screams on black.

Markers really are translucent, so a sticker composited over a dark background
is genuinely darker than the same sticker on white.  That is the ink, not a
halo; a halo is grey and desaturated, and follows the stroke as an outline.

## Tuning

Start from `--debug`, which writes `OUT/<photo>/debug/`: `01_rectified`,
`02_background`, `03_normalized`, `04_alpha`, `05_ink`,
`06_components_overlay`.  Then:

| symptom | flag |
| --- | --- |
| pale strokes missing | raise `--sat-weight` (default 0.7) |
| board noise surviving | raise `--alpha-floor` (0.06) or `--min-area` (150) |
| two quotes merged into one sticker | lower `--merge-radius` |
| one quote split into several stickers | raise `--merge-radius` |
| stray marks exported as quotes | raise `--min-quote-area` (800) |
| eraser ghosts surviving | raise `--ghost-max-alpha` (0.35) / `--ghost-max-sat` (0.10) |
| faint real strokes eaten as ghosts | lower those two |
| strokes bleeding into `02_background` | `--bg-method close` (fat-marker fallback) |
| board quad missed or wrong | `--corners x1,y1,...` or `--no-rectify` |
| design wants punchier colour | `--boost-sat 0.3` (off by default, on purpose) |

`--merge-radius` defaults to **5x the median stroke width measured in that
photo**, not a fixed pixel count.  A fixed count cannot be right: the same board
at 2MP and 12MP needs different numbers, as do a fine-tip and a chisel-tip
marker.  The value actually used is recorded in each `manifest.json`.

`--no-bleed` turns off the few-pixel extension of ink colour underneath fully
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
vignette, glare blob, six words in known ink colours at known coverage, sensor
noise -- so the pipeline can be checked against ground truth without waiting for
shoot day.  The suite asserts background cleanliness, hue preservation within 8
degrees, stroke IoU, ghost and speck removal, that cleaning does not erode black
text, one sticker per quote, no grey halos, and rectification of a board yawed
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

Two overlapping quotes in different colours merge into a single sticker.
Splitting them by hue clustering inside the merged component is a stretch goal,
along with `--group-by-color` and a small drag-and-drop web UI.
