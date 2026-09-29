"""The runtime order: rectify -> normalize -> matte -> unmix -> clean -> stickers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import clean as clean_mod
from . import matte, normalize as norm_mod, rectify as rectify_mod, stickers, unmix as unmix_mod
from .io import load_rgba, save_gray, save_rgb


class Keep(str, Enum):
    """Which of the quotes found in one photo to export."""

    ALL = "all"        # a board shot: every quote is a sticker
    MAIN = "main"      # a close-up: the centred quote, not the neighbours at the edges
    MERGED = "merged"  # a hand-cropped quote: all strokes are one sticker


@dataclass
class Params:
    alpha_floor: float = 0.06
    sat_weight: float = 0.7
    min_area: int = 150
    min_quote_area: int = 800
    merge_radius: int | None = None  # None = derive from stroke width
    boost_sat: float = 0.0
    deglare: bool = True
    specular_excess: float = 0.01
    ghost_max_alpha: float = 0.35
    ghost_max_sat: float = 0.10
    pad: int = 12
    bg_method: str = "median"
    bg_median_k: int = 41
    bg_refine: bool = True
    bg_ink_slack: int = 2
    bg_ink_chroma: float = 0.08
    rectify: bool = True
    board_mask: bool = True
    corners: list[float] | None = None
    keep: Keep | None = None  # None = MERGED for pre-cropped input, else ALL
    crop_erode: int = 3
    remove_people: bool = False
    bleed: bool = True

    def public(self) -> dict:
        d = asdict(self)
        d.pop("corners", None)
        d["keep"] = None if self.keep is None else Keep(self.keep).value
        return d


@dataclass
class Result:
    source: Path
    out_dir: Path
    manifest: dict
    rectified: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def n_quotes(self) -> int:
        return len(self.manifest["quotes"])


def _overlay(alpha: np.ndarray, quotes, labels: np.ndarray) -> np.ndarray:
    """Kept quote components tinted and boxed, for eyeballing merge_radius."""
    base = np.dstack([1.0 - np.clip(alpha, 0, 1)] * 3).astype(np.float32)
    rng = np.random.default_rng(0)
    colors = []
    for label, _x, _y, _w, _h, _area in quotes:
        color = rng.uniform(0.1, 0.9, size=3).astype(np.float32)
        colors.append(color)
        sel = (labels == label) & (alpha > 0)
        base[sel] = base[sel] * 0.35 + color * 0.65

    u8 = np.clip(base * 255.0 + 0.5, 0, 255).astype(np.uint8)  # OpenCV draws on 8-bit only
    for i, ((_label, x, y, w, h, _area), color) in enumerate(zip(quotes, colors), start=1):
        bgr = [float(c * 255) for c in color]
        cv2.rectangle(u8, (x, y), (x + w - 1, y + h - 1), bgr, 2)
        cv2.putText(u8, str(i), (x + 4, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, bgr, 2, cv2.LINE_AA)
    return u8.astype(np.float32) / 255.0


def process_image(
    img: np.ndarray,
    p: Params,
    debug_dir: Path | None = None,
    warnings: list[str] | None = None,
    crop: np.ndarray | None = None,
):
    """Run the full pipeline on a loaded photo.

    ``crop`` marks which pixels are real board, for input that was lassoed out
    of a larger photo (see :func:`wbx.io.load_rgba`).  It is eroded before use:
    the lasso edge is antialiased and usually shaves the board itself, and a
    boundary pixel that is half board and half cleared-to-black mattes as a
    solid stroke, drawing the outline of the crop around the sticker.

    A cropped input is also, by construction, not a board photo: there is no
    quad to rectify, and it holds exactly one quote.

    Returns ``(ink, alpha, quotes, labels, rectified, merge_radius)``.
    """
    warnings = warnings if warnings is not None else []
    # only a mask that came with the file says "this is one quote someone cut out";
    # a board region detected below is just geometry and says nothing about count
    keep = p.keep if p.keep is not None else (Keep.MERGED if crop is not None else Keep.ALL)
    hand_cropped = crop is not None

    people = None
    if p.remove_people:
        from .people import person_mask

        people = person_mask(img)
        if debug_dir:
            save_gray(debug_dir / "00_person_mask.png", people.astype(np.float32))

    rectified = False
    if p.rectify and not hand_cropped:
        corners = np.array(p.corners, dtype=np.float32).reshape(4, 2) if p.corners else None
        warped, used = rectify_mod.rectify(img, corners)
        if used is None:
            warnings.append("no board quad found; continuing unrectified")
        else:
            # the person mask belongs to the original frame; warp it the same way
            if people is not None:
                people = rectify_mod.warp_to_quad(people.astype(np.float32)[..., None].repeat(3, 2), used)[..., 0] > 0.5
            img = warped
            rectified = True
    if debug_dir:
        save_rgb(debug_dir / "01_rectified.png", img)

    if crop is None and p.board_mask and not rectified:
        # no quad means the board runs off the picture, so the frame, the wall and
        # whatever is past the end of the board are still in shot -- and connected
        crop = board_mod.board_region(img)

    if crop is not None:
        r = max(0, int(p.crop_erode))
        if r:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            crop = cv2.erode(crop.astype(np.uint8), k).astype(bool)
        if not crop.any():
            warnings.append("crop mask is empty after erosion; ignoring it")
            crop = None
    if debug_dir and crop is not None:
        save_gray(debug_dir / "01b_board_mask.png", crop.astype(np.float32))

    # everything outside the crop is cleared-to-black, not board: never sample it
    exclude = people if crop is None else (~crop if people is None else (~crop | people))

    def board(sample_exclude):
        bg = norm_mod.estimate_background(
            img, median_k=p.bg_median_k, method=p.bg_method, exclude=sample_exclude
        )
        norm = norm_mod.normalize(img, bg)
        a = matte.estimate_alpha(norm, sat_weight=p.sat_weight, alpha_floor=p.alpha_floor)
        if p.deglare:
            # A pixel brighter than the board is light added on top of it, not ink
            # taken out of it.  The darkness cue already reads zero there -- the
            # normalized value clips at 1.0 -- but the saturation cue does not, and
            # a reflection off a glossy board carries enough tint to matte as a pale
            # cloud sitting over the glare.
            #
            # Measured on the *darkest* channel, never the brightest: yellow marker
            # is brighter than the board in red and green and only gives itself away
            # in blue, so a brightest-channel test reads it as a reflection and
            # deletes it.  Added light raises every channel; ink lowers at least one.
            a[img.min(axis=2) > bg.min(axis=2) * (1.0 + p.specular_excess)] = 0.0
        if people is not None:
            a[people] = 0.0
        if crop is not None:
            a[~crop] = 0.0
        return bg, norm, a

    bg, norm, alpha = board(exclude)

    if p.bg_refine and alpha.any():
        # The worst contaminant of a stroke's background estimate is the stroke.
        # The median only rejects ink that is thin relative to its window, so
        # inside a fat marker the estimate sinks to the ink itself, the board
        # there normalizes to 1.0, and the stroke mattes as a hole: letters come
        # out hollow, with a bright rim and nothing in the middle.  Sizing the
        # kernel up merely moves the marker width at which that happens.  So
        # take the first matte, exclude everything it called ink, and re-fit --
        # the same treatment people in the frame get, for the same reason.
        # The first matte is not enough on its own to say where the ink is: it was
        # computed from the very background estimate being corrected, so where that
        # estimate has sunk into a dense patch of colour the matte sees no ink there
        # and the refit dutifully samples the ink again.  Chroma needs no background
        # -- it is max(R,G,B) - min(R,G,B) on the raw photo -- so coloured marker is
        # masked whatever the estimate believes, and the matte still carries black,
        # which has no chroma to find.  Bare board sits near 0.03.
        # ...and only where the pixel is no brighter than the board already looks,
        # on its darkest channel (see the specular note above: yellow fails a
        # brightest-channel test).  Ink darkens the board; a specular highlight
        # brightens it, and a glossy board throws plenty of those.  Without this the
        # reflection is masked out as if it were ink, the estimate under it is
        # interpolated from the glare around it, and the ordinary board beside it
        # then mattes as a grey cloud.
        darker = img.min(axis=2) <= bg.min(axis=2)
        ink_mask = ((alpha > 0) | (matte.chroma(img) > float(p.bg_ink_chroma))) & darker
        r = max(0, int(p.bg_ink_slack))
        if r:  # a little slack for the soft rim that fell under the alpha floor
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            ink_mask = cv2.dilate(ink_mask.astype(np.uint8), k).astype(bool)
        refit = ink_mask if exclude is None else (ink_mask | exclude)
        if not refit.all():
            bg, norm, alpha = board(refit)

    if debug_dir:
        save_rgb(debug_dir / "02_background.png", bg)
        save_rgb(debug_dir / "03_normalized.png", norm)
    ink = unmix_mod.unmix(norm, alpha, boost_sat=p.boost_sat)
    if debug_dir:
        save_gray(debug_dir / "04_alpha.png", alpha)

    alpha, ink = clean_mod.clean(
        alpha,
        ink,
        norm,
        min_area=p.min_area,
        ghost_max_alpha=p.ghost_max_alpha,
        ghost_max_sat=p.ghost_max_sat,
        alpha_floor=p.alpha_floor,
    )
    if p.deglare:
        alpha, ink = clean_mod.fill_specular(alpha, ink, img, bg)
    if crop is not None:
        alpha, ink = clean_mod.drop_crop_edge(alpha, ink, crop)
    if debug_dir:
        save_rgb(debug_dir / "05_ink.png", ink)

    quotes, labels, _mask, merge_radius = stickers.find_quotes(
        alpha,
        merge_radius=p.merge_radius,
        min_quote_area=p.min_quote_area,
        pad=p.pad,
        min_area=p.min_area,
        single=keep == Keep.MERGED,
    )
    if keep == Keep.MAIN:
        quotes, labels = stickers.pick_main(quotes, labels, alpha.shape)
    if debug_dir:
        save_rgb(debug_dir / "06_components_overlay.png", _overlay(alpha, quotes, labels))

    return ink, alpha, quotes, labels, rectified, merge_radius


def process_photo(path: Path, out_root: Path, p: Params, debug: bool = False) -> Result:
    """Process one photo into ``OUT/<stem>/``."""
    path = Path(path)
    out_dir = Path(out_root) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    img, crop = load_rgba(path)
    ink, alpha, quotes, labels, rectified, merge_radius = process_image(
        img, p, debug_dir, warnings, crop=crop
    )
    params = p.public()
    params["merge_radius"] = merge_radius  # record what was actually used

    manifest = stickers.export(
        out_dir,
        source=path.name,
        ink=ink,
        alpha=alpha,
        quotes=quotes,
        labels=labels,
        rectified=rectified,
        params=params,
        bleed=p.bleed,
    )
    return Result(path, out_dir, manifest, rectified, warnings)
