"""The honesty check for the whole pipeline.

Every assertion here is about the synthetic board in ``tests/synthetic.py``,
where the true ink colours and stroke shapes are known exactly.  Keep it green
through every tuning change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wbx import clean as clean_mod
from wbx import matte, normalize as norm_mod, rectify as rectify_mod, unmix as unmix_mod
from wbx.io import load_image, save_rgb
from wbx.pipeline import Params, process_image, process_photo
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
