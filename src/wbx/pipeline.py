"""The runtime order: rectify -> normalize -> matte -> unmix -> clean -> stickers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from . import clean as clean_mod
from . import matte, normalize as norm_mod, rectify as rectify_mod, stickers, unmix as unmix_mod
from .io import load_image, save_gray, save_rgb


@dataclass
class Params:
    alpha_floor: float = 0.06
    sat_weight: float = 0.7
    min_area: int = 150
    min_quote_area: int = 800
    merge_radius: int | None = None  # None = derive from stroke width
    boost_sat: float = 0.0
    ghost_max_alpha: float = 0.35
    ghost_max_sat: float = 0.10
    pad: int = 12
    bg_method: str = "median"
    bg_median_k: int = 41
    rectify: bool = True
    corners: list[float] | None = None
    remove_people: bool = False
    bleed: bool = True

    def public(self) -> dict:
        d = asdict(self)
        d.pop("corners", None)
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
):
    """Run the full pipeline on a loaded photo.

    Returns ``(ink, alpha, quotes, labels, rectified)``.
    """
    warnings = warnings if warnings is not None else []
    people = None
    if p.remove_people:
        from .people import person_mask

        people = person_mask(img)
        if debug_dir:
            save_gray(debug_dir / "00_person_mask.png", people.astype(np.float32))

    rectified = False
    if p.rectify:
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

    bg = norm_mod.estimate_background(
        img, median_k=p.bg_median_k, method=p.bg_method, exclude=people
    )
    norm = norm_mod.normalize(img, bg)
    if debug_dir:
        save_rgb(debug_dir / "02_background.png", bg)
        save_rgb(debug_dir / "03_normalized.png", norm)

    alpha = matte.estimate_alpha(norm, sat_weight=p.sat_weight, alpha_floor=p.alpha_floor)
    if people is not None:
        alpha[people] = 0.0
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
    if debug_dir:
        save_rgb(debug_dir / "05_ink.png", ink)

    quotes, labels, _mask, merge_radius = stickers.find_quotes(
        alpha,
        merge_radius=p.merge_radius,
        min_quote_area=p.min_quote_area,
        pad=p.pad,
    )
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
    img = load_image(path)
    ink, alpha, quotes, labels, rectified, merge_radius = process_image(
        img, p, debug_dir, warnings
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
