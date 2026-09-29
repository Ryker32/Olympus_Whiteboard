"""The honesty check for the whole pipeline.

Every assertion here is about the synthetic board in ``tests/synthetic.py``,
where the true ink colours and stroke shapes are known exactly.  Keep it green
through every tuning change.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from wbx import clean as clean_mod
from wbx import matte, normalize as norm_mod, rectify as rectify_mod, unmix as unmix_mod
from wbx.io import load_image, load_rgba, save_rgb, save_rgba
from wbx.pipeline import Keep, Params, process_image, process_photo
from wbx.album import build_album
from wbx.recut import recut
from wbx.sheet import build_sheet
from wbx.stickers import find_quotes, summarize_color

import synthetic as syn


# --------------------------------------------------------------------------- helpers

def run(photo, **kw):
    p = Params(rectify=False, **kw)
    return process_image(photo, p)


def hue_of(rgb: np.ndarray) -> float:
    return float(matte.hue(np.asarray(rgb, np.float32).reshape(1, 1, 3))[0, 0])


def hue_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def dominant_hue(ink: np.ndarray, alpha: np.ndarray, sel: np.ndarray) -> float:
    w = (alpha * matte.saturation(ink))[sel].astype(np.float64)
    th = np.deg2rad(matte.hue(ink)[sel].astype(np.float64))
    return float(np.rad2deg(np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())) % 360.0)


@pytest.fixture(scope="module")
def board():
    return syn.make_board()


@pytest.fixture(scope="module")
def extracted(board):
    photo, strokes = board
    ink, alpha, quotes, labels, _rect, _merge = run(photo)
    return photo, strokes, ink, alpha, quotes, labels


# --------------------------------------------------------------------------- M2

def test_background_is_clean(extracted):
    """Bare board must come out fully transparent, glare and vignette included."""
    _photo, strokes, _ink, alpha, _q, _l = extracted
    total = np.zeros(alpha.shape, np.float32)
    for s in strokes:
        total = np.maximum(total, s.coverage)
    bare = total < 0.02
    leaked = float((alpha[bare] > 0.05).mean())
    assert leaked < 0.005, f"{leaked:.4%} of bare board survived as ink"


def test_hue_is_preserved(extracted):
    """The whole point of unmixing: recovered ink keeps the marker's hue."""
    _photo, strokes, ink, alpha, _q, _l = extracted
    for s in strokes:
        if s.name in syn.ACHROMATIC:
            continue  # black has no meaningful hue
        core = (s.coverage > 0.5 * s.coverage.max()) & (alpha > 0)
        assert core.sum() > 500
        delta = hue_delta(dominant_hue(ink, alpha, core), hue_of(s.ink))
        assert delta <= 8.0, f"{s.name}: hue drifted {delta:.1f} degrees"


def test_stroke_coverage_iou(extracted):
    """Recovered strokes must land on the real ones, not near them."""
    _photo, strokes, _ink, alpha, _q, _l = extracted
    got = alpha > 0
    union = np.zeros(alpha.shape, bool)
    for s in strokes:
        truth = s.coverage > 0.1
        union |= truth
        ys, xs = np.nonzero(truth)
        # score each word inside its own neighbourhood; the other five words are
        # not misses of this one
        box = np.zeros(alpha.shape, bool)
        box[max(0, ys.min() - 30):ys.max() + 31, max(0, xs.min() - 30):xs.max() + 31] = True
        near = got & box
        iou = float((near & truth).sum()) / float((near | truth).sum())
        assert iou > 0.7, f"{s.name}: IoU {iou:.3f}"

    overall = float((got & union).sum()) / float((got | union).sum())
    assert overall > 0.7, f"whole board IoU {overall:.3f}"


def test_unmix_preserves_hue_under_wrong_alpha():
    """Even a badly misestimated alpha only costs saturation, never hue."""
    ink = np.float32([[[0.98, 0.84, 0.24]]])
    a_true = 0.7
    norm = a_true * ink + (1 - a_true)
    for a_guess in (0.4, 0.55, 0.7, 0.9, 1.0):
        got = unmix_mod.unmix(norm, np.float32([[a_guess]]))
        # the recovered colour rides the white-to-ink ray, so the only thing that
        # can move the hue at all is clipping where that ray leaves the gamut
        assert hue_delta(hue_of(got[0, 0]), hue_of(ink[0, 0])) < 8.0
    exact = unmix_mod.unmix(norm, np.float32([[a_true]]))
    assert hue_delta(hue_of(exact[0, 0]), hue_of(ink[0, 0])) < 0.5


# --------------------------------------------------------------------------- M3

def test_specks_and_ghosts_are_removed():
    photo, strokes = syn.make_board()
    speckled = syn.add_specks(photo)
    dirty, ghost_cov = syn.add_ghost(speckled)
    ghost_region = ghost_cov > 0.15

    norm = norm_mod.normalize(dirty, norm_mod.estimate_background(dirty))
    raw = matte.estimate_alpha(norm)
    seen = float((raw[ghost_region] > 0).mean())
    assert seen > 0.5, f"the ghost fixture is invisible to the matte ({seen:.2%}); test proves nothing"

    speck_region = (matte.estimate_alpha(
        norm_mod.normalize(speckled, norm_mod.estimate_background(speckled))) > 0)
    for s in strokes:
        speck_region &= s.coverage <= 0.05
    assert speck_region.sum() > 100, "the speck fixture is invisible to the matte"

    _ink, alpha_clean, _q, _l, _r, _m = run(dirty)
    assert float((alpha_clean[ghost_region] > 0).mean()) < 0.02, "eraser ghost survived"
    assert float((alpha_clean[speck_region] > 0).mean()) < 0.02, "specks survived"

    for s in strokes:
        core = s.coverage > 0.5 * s.coverage.max()
        assert float((alpha_clean[core] > 0).mean()) > 0.97, f"{s.name} was eaten by cleaning"


def test_cleaning_does_not_erode_black_text(board):
    """Component-level cleaning must leave the antialiased rim of real writing alone."""
    photo, strokes = board
    norm = norm_mod.normalize(photo, norm_mod.estimate_background(photo))
    alpha = matte.estimate_alpha(norm)
    ink = unmix_mod.unmix(norm, alpha)
    black = next(s for s in strokes if s.name == "black").coverage > 0.1

    before = int((alpha[black] > 0).sum())
    alpha2, _ = clean_mod.clean(alpha, ink, norm)
    after = int((alpha2[black] > 0).sum())
    assert before > 0
    change = abs(before - after) / before
    assert change < 0.03, f"black text area changed {change:.2%} during cleaning"


# --------------------------------------------------------------------------- M4

def test_one_sticker_per_quote(extracted):
    _photo, strokes, _ink, _alpha, quotes, _l = extracted
    assert len(quotes) == len(strokes)


def test_quotes_are_not_split_or_merged(extracted):
    """Each sticker's bbox should contain exactly one ground-truth word."""
    _photo, strokes, _ink, _alpha, quotes, _l = extracted
    for _label, x, y, w, h, _area in quotes:
        hits = [s.name for s in strokes if (s.coverage[y:y + h, x:x + w] > 0.1).sum() > 200]
        assert len(hits) == 1, f"sticker at {(x, y, w, h)} covers {hits}"


def test_no_grey_halo_on_dark_background(tmp_path, board):
    """The dark half of the contact sheet is the halo detector; check it numerically."""
    photo, strokes = board
    src = tmp_path / "board_photo.png"
    save_rgb(src, photo)
    result = process_photo(src, tmp_path / "out", Params(rectify=False))
    manifest = json.loads((result.out_dir / "manifest.json").read_text())
    assert len(manifest["quotes"]) == len(strokes)

    chromatic = {hue_of(s.ink) for s in strokes if s.name not in syn.ACHROMATIC}
    checked = 0
    for entry in manifest["quotes"]:
        rgba = load_image_rgba(result.out_dir / entry["file"])
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        core = alpha > 0.6 * alpha.max()
        edge = (alpha > 0.10) & (alpha < 0.40)
        if core.sum() < 200 or edge.sum() < 200:
            continue
        sat = matte.saturation(rgb)
        if sat[core].mean() < 0.15:
            continue  # the black quote has nothing to fringe
        checked += 1
        ratio = float(sat[edge].mean() / sat[core].mean())
        assert ratio > 0.5, f"{entry['file']}: edges are {1 - ratio:.0%} less saturated -- grey halo"
        assert hue_delta(dominant_hue(rgb, alpha, edge), dominant_hue(rgb, alpha, core)) < 15.0
    assert checked >= len(chromatic) - 1

    dest = build_sheet(result.out_dir)
    assert dest.exists()


def load_image_rgba(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0


def test_manifest_colors_match_the_markers(tmp_path, board):
    photo, strokes = board
    src = tmp_path / "board_photo.png"
    save_rgb(src, photo)
    result = process_photo(src, tmp_path / "out", Params(rectify=False))

    truth = {s.name: hue_of(s.ink) for s in strokes if s.name not in syn.ACHROMATIC}
    got = [q["dominant_hue"] for q in result.manifest["quotes"] if q["dominant_hue"] is not None]
    for want in truth.values():
        assert any(hue_delta(g, want) <= 8.0 for g in got), f"no sticker reported hue near {want:.0f}"


# --------------------------------------------------------------------------- M5

def test_rectify_finds_the_board_corners():
    photo, _strokes = syn.make_board()
    frame, corners = syn.place_on_wall(photo, angle_deg=25.0)
    found = rectify_mod.find_board_quad(frame)
    assert found is not None, "board quad not detected"
    err = np.abs(found - rectify_mod.order_corners(corners)).max()
    assert err <= 3.0, f"corner error {err:.2f}px"


def test_rectified_board_still_yields_every_quote():
    photo, strokes = syn.make_board()
    frame, _corners = syn.place_on_wall(photo, angle_deg=25.0)
    _ink, _alpha, quotes, _labels, rectified, _merge = process_image(frame, Params())
    assert rectified
    assert len(quotes) == len(strokes)


def test_rectify_passes_through_when_no_board_found():
    flat = np.full((300, 400, 3), 0.9, np.float32)
    out, corners = rectify_mod.rectify(flat)
    assert corners is None
    assert out.shape == flat.shape


# --------------------------------------------------------------------------- CLI

def test_cli_extract_and_sheet(tmp_path, board):
    from typer.testing import CliRunner

    from wbx.cli import app

    photo, strokes = board
    src = tmp_path / "IMG_0001.png"
    save_rgb(src, photo)
    out = tmp_path / "out"

    runner = CliRunner()
    res = runner.invoke(app, ["extract", str(src), "-o", str(out), "--debug", "--no-rectify"])
    assert res.exit_code == 0, res.output

    quote_dir = out / "IMG_0001"
    assert len(list(quote_dir.glob("quote_*.png"))) == len(strokes)
    assert (quote_dir / "board.png").exists()
    for name in ("01_rectified", "02_background", "03_normalized", "04_alpha", "05_ink",
                 "06_components_overlay"):
        assert (quote_dir / "debug" / f"{name}.png").exists()

    res = runner.invoke(app, ["sheet", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "sheet.png").exists()


# --------------------------------------------------------------------------- real photos

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def sample(name: str) -> Path:
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} not in samples/")
    return path


def test_bare_board_yields_nothing():
    """The empty-board calibration shot: real glare and scuffs, zero stickers."""
    img = load_image(sample("nomarker.jpeg"))
    _ink, alpha, quotes, _labels, _rect, _merge = process_image(img, Params())
    assert quotes == []
    assert int((alpha > 0).sum()) == 0, "bare board left ink behind"


def test_frame_filling_board_is_not_rectified():
    """A board with no visible corners must not be warped to a hallucinated quad."""
    for name in ("nomarker.jpeg", "exampletext.jpeg"):
        img = load_image(sample(name))
        out, corners = rectify_mod.rectify(img)
        assert corners is None, f"{name}: invented a board quad"
        assert out.shape == img.shape


def test_real_handwriting_is_one_sticker():
    """Letter gaps scale with the photo; a fixed 30px radius split this into five."""
    img = load_image(sample("exampletext.jpeg"))
    _ink, _alpha, quotes, _labels, _rect, merge_radius = process_image(img, Params())
    assert len(quotes) == 1, f"'Hello' came out as {len(quotes)} stickers"
    assert 40 <= merge_radius <= 120, f"auto merge radius {merge_radius}px looks wrong"


def test_real_marker_colour_is_recovered():
    """The teal marker must come back teal, not grey and not blue."""
    img = load_image(sample("exampletext.jpeg"))
    ink, alpha, quotes, labels, _rect, _merge = process_image(img, Params())
    (label, x, y, w, h, _area) = quotes[0]
    sub_a = alpha[y:y + h, x:x + w]
    sub_ink = ink[y:y + h, x:x + w]
    color, dom_hue = summarize_color(sub_ink, sub_a)
    assert dom_hue is not None, "teal marker reported as achromatic"
    assert 160.0 <= dom_hue <= 195.0, f"hue {dom_hue} is not teal"


# --------------------------------------------------------------------------- pre-cropped input

def _lassoed(tmp_path, photo, strokes, names, frame=0.0, pad=60, name="crop.png"):
    """Save a lasso cut-out around the named words, and say where they were."""
    want = [s for s in strokes if s.name in names]
    ys, xs = np.nonzero(sum(s.coverage for s in want) > 0.1)
    box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(photo.shape[1], int(xs.max()) + pad), min(photo.shape[0], int(ys.max()) + pad))
    rgb, alpha = syn.lasso_crop(photo, box, frame=frame)
    src = tmp_path / name
    save_rgba(src, rgb, alpha)
    return src, box, want


def test_crop_mask_is_read_back(tmp_path, board):
    photo, strokes = board
    src, box, _ = _lassoed(tmp_path, photo, strokes, {"black"})
    rgb, crop = load_rgba(src)
    assert crop is not None and not crop.all(), "lasso cut-out read back as fully opaque"
    assert rgb.shape[:2] == crop.shape == (box[3] - box[1], box[2] - box[0])

    plain = load_image(src)
    assert plain[~crop].min() > 0.9, "transparency composited over black, not board white"


def test_cleared_surround_is_not_ink(tmp_path, board):
    """The black under the transparent pixels must not matte as a solid blob."""
    photo, strokes = board
    src, _box, want = _lassoed(tmp_path, photo, strokes, {"black"})
    result = process_photo(src, tmp_path / "out", Params())

    rgba = load_image_rgba(result.out_dir / "board.png")
    alpha = rgba[..., 3]
    _rgb, crop = load_rgba(src)
    assert float(alpha[~crop].max()) == 0.0, "cleared surround came out opaque"

    truth = int((want[0].coverage > 0.1).sum())
    got = int((alpha > 0).sum())
    assert 0.6 * truth < got < 1.6 * truth, f"ink area {got} vs {truth} written pixels"


def test_one_sticker_per_crop(tmp_path, board):
    """A lasso is the operator saying 'this is one quote'; never split it again."""
    photo, strokes = board
    src, box, want = _lassoed(tmp_path, photo, strokes, {"black", "yellow"})
    result = process_photo(src, tmp_path / "out", Params())

    assert len(result.manifest["quotes"]) == 1, "hand-cropped quote came out as several stickers"
    assert result.rectified is False, "tried to find a board quad in a crop"

    q = result.manifest["quotes"][0]
    for s in want:  # the one sticker has to hold both words, not just the bigger
        sub = s.coverage[box[1]:box[3], box[0]:box[2]][q["y"]:q["y"] + q["h"], q["x"]:q["x"] + q["w"]]
        assert (sub > 0.1).sum() > 200, f"{s.name} missing from the sticker"


def test_clipped_board_frame_is_dropped(tmp_path, board):
    """The dark bar a lasso catches at the board edge is not part of the quote."""
    photo, strokes = board
    plain, _b, _w = _lassoed(tmp_path, photo, strokes, {"black"}, name="plain.png")
    framed, _b2, _w2 = _lassoed(tmp_path, photo, strokes, {"black"}, frame=0.08, name="framed.png")

    a = process_photo(plain, tmp_path / "out_plain", Params())
    b = process_photo(framed, tmp_path / "out_framed", Params())
    assert len(b.manifest["quotes"]) == 1

    before = a.manifest["quotes"][0]["area"]
    after = b.manifest["quotes"][0]["area"]
    assert abs(after - before) < 0.05 * before, f"frame removal cost {before - after} px of writing"


def test_single_quote_can_be_forced_off(tmp_path, board):
    """The auto behaviour is a default, not a rule: a crop can hold two quotes."""
    photo, strokes = board
    src, _box, _want = _lassoed(tmp_path, photo, strokes, {"black", "yellow"})
    result = process_photo(src, tmp_path / "out", Params(keep=Keep.ALL))
    assert len(result.manifest["quotes"]) == 2


# --------------------------------------------------------------------------- fat marker

def _fat_core(cov):
    """Deep inside the fat strokes, where the background estimate goes wrong."""
    d = cv2.distanceTransform((cov > 0.5 * cov.max()).astype(np.uint8), cv2.DIST_L2, 5)
    return d >= 6


def test_fat_marker_comes_out_solid(board):
    """A chisel-tip stroke must not export as an outline with a hole in it."""
    photo, _strokes = board
    photo, cov = syn.add_fat_word(photo)
    core = _fat_core(cov)
    assert core.sum() > 2000, "the test's own fat word is too thin to prove anything"

    _ink, alpha, _q, _l, _r, _m = run(photo)
    filled = float((alpha[core] > 0.5).mean())
    assert filled > 0.97, f"{1 - filled:.0%} of the fat stroke came out hollow"


def test_fat_marker_is_hollow_without_the_refit(board):
    """Why the second background pass exists; keeps the test above honest."""
    photo, _strokes = board
    photo, cov = syn.add_fat_word(photo)
    core = _fat_core(cov)

    _ink, alpha, _q, _l, _r, _m = run(photo, bg_refine=False)
    filled = float((alpha[core] > 0.5).mean())
    assert filled < 0.75, "one pass now fills fat strokes; the refit may be redundant"


def test_refit_does_not_invent_ink_on_a_bare_board():
    """Excluding the first pass's ink must not leave a flat patch that mattes."""
    photo = syn.board_lighting()
    _ink, alpha, quotes, _l, _r, _m = run(photo)
    assert quotes == []
    assert float(alpha.mean()) < 0.001, "the refit manufactured ink out of lighting"


# --------------------------------------------------------------------------- album

def test_album_is_a_flat_folder_of_copies(tmp_path, board):
    """Handover folder: every sticker, renamed, byte-identical, plus provenance."""
    import csv

    photo, strokes = board
    src = tmp_path / "IMG_0042.png"
    save_rgb(src, photo)
    out = tmp_path / "out"
    result = process_photo(src, out, Params(rectify=False))

    dest = tmp_path / "album"
    rows = build_album(out, dest)
    assert len(rows) == len(strokes)

    files = sorted(dest.glob("sticker_*.png"))
    assert [f.name for f in files] == [f"sticker_{i:02d}.png" for i in range(1, len(strokes) + 1)]
    originals = sorted(result.out_dir.glob("quote_*.png"))
    for copy, original in zip(files, originals):
        assert copy.read_bytes() == original.read_bytes(), f"{copy.name} was re-encoded"
    assert not (dest / "board.png").exists(), "the whole-board render is not a sticker"

    with (dest / "index.csv").open() as fh:
        index = list(csv.DictReader(fh))
    assert [r["sticker"] for r in index] == [f.name for f in files]
    assert {r["source photo"] for r in index} == {"IMG_0042.png"}

    assert build_sheet(dest).exists()  # the flat layout stays reviewable


def test_album_flags_the_same_quote_cropped_twice(tmp_path, board):
    """Lassoing one quote twice is the normal accident; say so, drop nothing."""
    photo, strokes = board
    out = tmp_path / "out"
    for name in ("first", "second"):
        src, _box, _want = _lassoed(tmp_path, photo, strokes, {"red"}, name=f"{name}.png")
        process_photo(src, out, Params())

    rows = build_album(out, tmp_path / "album")
    assert len(rows) == 2, "a flagged duplicate must still be copied"
    assert rows[1].looks_like == rows[0].file
    assert rows[0].looks_like == "", "the first copy is the one to keep"


def test_album_does_not_flag_different_quotes(tmp_path, board):
    photo, strokes = board
    out = tmp_path / "out"
    src = tmp_path / "IMG_0043.png"
    save_rgb(src, photo)
    process_photo(src, out, Params(rectify=False))

    rows = build_album(out, tmp_path / "album")
    assert len(rows) == len(strokes)
    assert [r.looks_like for r in rows] == [""] * len(rows), "distinct quotes flagged as copies"


# --------------------------------------------------------------------------- close-ups

def _subject_of(photo, **kw):
    _ink, _a, quotes, _l, _r, _m = run(photo, **kw)
    return quotes


def test_close_up_keeps_the_subject_not_the_neighbours():
    """One message per photo still catches the corner of the next one."""
    H, W = 1600, 2200
    subject = syn.word("THANKS", rgb=(30, 90, 200))

    def half_off(text, rgb, edge, along):
        """Place a word so half of it hangs off the named edge of the frame."""
        w = syn.word(text, rgb=rgb)
        ih, iw = w[0].shape[:2]
        pos = {"left": (-iw // 2, along), "right": (W - iw // 2, along),
               "top": (along, -ih // 2), "bottom": (along, H - ih // 2)}[edge]
        return (w, pos[0], pos[1])

    cases = [
        [half_off("NEIGHBOUR LONG", (200, 40, 60), "right", 200)],
        [half_off("LEFT", (40, 150, 80), "left", 180),
         half_off("RIGHT", (140, 80, 190), "right", 1150)],
        [half_off("ABOVE", (200, 40, 60), "top", 700)],
        [half_off("BELOW WIDE", (40, 150, 80), "bottom", 150)],
    ]
    for i, intruders in enumerate(cases):
        photo, (sx, sy) = syn.close_up(subject, intruders, size=(H, W))
        found = _subject_of(photo, keep=Keep.ALL)
        assert len(found) > 1, f"case {i}: the intruders did not even register"

        kept = _subject_of(photo, keep=Keep.MAIN)
        assert len(kept) == 1, f"case {i}: kept {len(kept)} quotes, want the subject only"
        _label, x, y, _w, _h, _area = kept[0]
        assert abs(x - sx) < 90 and abs(y - sy) < 90, \
            f"case {i}: kept the quote at {(x, y)}, subject is at {(sx, sy)}"


def test_close_up_with_nothing_clipped_keeps_the_largest():
    """Rule has nothing to say when every quote is whole; fall back to size."""
    subject = syn.word("THANKS SO MUCH", rgb=(30, 90, 200))
    photo, (sx, sy) = syn.close_up(subject, [(syn.word("HI", rgb=(200, 40, 60)), 1700, 200)])
    kept = _subject_of(photo, keep=Keep.MAIN)
    assert len(kept) == 1
    assert abs(kept[0][1] - sx) < 90 and abs(kept[0][2] - sy) < 90


def test_keep_all_is_still_the_default_for_a_board_shot(board):
    photo, strokes = board
    assert len(_subject_of(photo)) == len(strokes)


# --------------------------------------------------------------------------- recut

def test_recut_finds_the_source_photo_and_gains_resolution(tmp_path, board):
    """A screenshot-scale crop is located in the full photo and cut again from it."""
    photo, strokes = board
    original = tmp_path / "20260101_120000.jpg"
    save_rgb(original, photo)

    # a crop at a fraction of the photo's scale: what a screenshot of it is.  The
    # real shoot came off the screen at about 1/2.2, which is the ratio to model.
    src, box, want = _lassoed(tmp_path, photo, strokes, {"red", "green"}, name="lasso.png")
    small = load_image_rgba(src)
    shrunk = cv2.resize(small, None, fx=1 / 2.2, fy=1 / 2.2, interpolation=cv2.INTER_AREA)
    crop = tmp_path / "crop.png"
    save_rgba(crop, shrunk[..., :3], shrunk[..., 3])

    dest = tmp_path / "recut"
    (match,) = recut([crop], [original], dest)
    assert match.photo is not None, f"no match (best {match.inliers} inliers)"
    assert match.photo.name == original.name
    assert match.written is not None and match.written.exists()
    assert match.gain > 1.8, f"only gained {match.gain}x; the point is resolution"

    rgb, mask = load_rgba(match.written)
    assert mask is not None and not mask.all(), "re-cut lost the outline"
    truth = sum(x.coverage for x in want)[box[1]:box[3], box[0]:box[2]]
    assert mask.sum() > 0.5 * (truth > 0.1).sum()


def test_recut_reports_a_crop_that_is_not_in_any_photo(tmp_path, board):
    """A crop from a different shoot must be reported, never matched anyway."""
    photo, strokes = board
    original = tmp_path / "board.jpg"
    save_rgb(original, photo)

    other = syn.board_lighting()
    other, _cov = syn.add_fat_word(other, word="ELSEWHERE", rgb=(200, 40, 60))
    stranger = tmp_path / "stranger.png"
    save_rgba(stranger, other, np.ones(other.shape[:2], np.float32))

    (match,) = recut([stranger], [original], tmp_path / "recut")
    assert match.photo is None and match.written is None


# --------------------------------------------------------------------------- glare

def test_pale_marker_survives_the_specular_test(board):
    """Yellow is brighter than the board in red and green; only blue gives it away.

    The glare rules ask whether a pixel is brighter than the board, and asking it
    of the brightest channel throws yellow away entirely.  This is the guard.
    """
    photo, strokes = board
    yellow = next(s for s in strokes if s.name == "yellow")
    core = yellow.coverage > 0.6 * yellow.coverage.max()

    _ink, alpha, _q, _l, _r, _m = run(photo)
    kept = float((alpha[core] > 0).mean())
    assert kept > 0.97, f"the glare rules deleted {1 - kept:.0%} of the yellow marker"


def test_reflection_is_not_matted_as_ink(board):
    """A highlight on a glossy board is light added, not ink; it must not matte."""
    photo, _strokes = board
    h, w = photo.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - 0.30 * w) ** 2 + (yy - 0.72 * h) ** 2) / (2 * (0.05 * w) ** 2)))
    glared = np.clip(photo + 0.30 * blob[..., None], 0, 1).astype(np.float32)
    spot = blob > 0.5

    _ink, plain, _q, _l, _r, _m = run(photo)
    _ink, after, _q, _l, _r, _m = run(glared)
    assert float((plain[spot] > 0).mean()) < 0.02, "the fixture landed on existing ink"
    assert float((after[spot] > 0).mean()) < 0.05, "the reflection came out as a sticker"


def test_dense_colour_does_not_bleach_itself(board):
    """A patch of solid colour must not drag its own background estimate into it."""
    photo, strokes = board
    red = next(s for s in strokes if s.name == "red")
    core = red.coverage > 0.6 * red.coverage.max()

    _ink, alpha, _q, _l, _r, _m = run(photo)
    recovered = float(alpha[core].mean())
    truth = float(red.coverage[core].mean())
    assert recovered > 0.75 * truth, \
        f"recovered {recovered:.2f} coverage where the truth is {truth:.2f}"
