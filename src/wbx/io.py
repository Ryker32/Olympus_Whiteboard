"""Image loading and saving.

Everything inside the pipeline is float32 RGB in [0, 1].  Loading goes through
Pillow rather than ``cv2.imread`` for two reasons: it honours the EXIF
orientation tag that phone cameras write, and it is the hook that ``pillow-heif``
registers into for iPhone HEIC files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HEIC_SUFFIXES = {".heic", ".heif"}

_heif_registered = False


class HeicSupportMissing(RuntimeError):
    """Raised when a HEIC photo is handed in without the ``heic`` extra."""


def _register_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise HeicSupportMissing(
            "HEIC photos need the optional dependency: install with "
            "`uv sync --extra heic` (or `pip install pillow-heif`), or convert "
            "the photos to JPEG first."
        ) from exc
    pillow_heif.register_heif_opener()
    _heif_registered = True


def load_rgba(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a photo as float32 RGB in [0, 1] plus its crop mask.

    The mask is ``None`` for an ordinary opaque photo.  It is a boolean array
    when the input is a *pre-cropped* quote: someone lassoed one quote out of a
    board photo and saved it with transparency, so the file carries both the
    pixels and which of them are real board.

    That mask cannot be thrown away.  Editors clear the RGB under transparent
    pixels to pure black, which is 20-40% of a typical lasso crop, so an
    alpha-blind load hands the pipeline a huge black blob: the background
    estimate is dragged dark and the blob mattes as solid ink.  Partially
    transparent pixels along the lasso edge keep their real colour (the alpha
    is straight, not premultiplied), but they still sit on the boundary, so
    only fully opaque pixels count as board.
    """
    path = Path(path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        _register_heif()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        im = im.convert("RGBA")
        arr = np.asarray(im, dtype=np.float32) / 255.0

    rgb = np.ascontiguousarray(arr[..., :3])
    if not has_alpha:
        return rgb, None
    crop = arr[..., 3] >= 1.0
    if crop.all():
        return rgb, None
    return rgb, np.ascontiguousarray(crop)


def load_image(path: str | Path) -> np.ndarray:
    """Load a photo as float32 RGB in [0, 1], EXIF rotation applied.

    Any transparency is composited over white -- the colour of bare board --
    rather than dropped, which would leave black where the pixels were cleared.
    Callers that need to know which pixels were transparent want
    :func:`load_rgba` instead.
    """
    rgb, crop = load_rgba(path)
    if crop is None:
        return rgb
    out = rgb.copy()
    out[~crop] = 1.0
    return out


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Image.fromarray(_u8(rgb), mode="RGB").save(Path(path))


def save_gray(path: str | Path, a: np.ndarray) -> None:
    Image.fromarray(_u8(a), mode="L").save(Path(path))


def save_rgba(path: str | Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    """Write straight (non-premultiplied) alpha RGBA PNG."""
    rgba = np.dstack([_u8(rgb), _u8(alpha)])
    Image.fromarray(rgba, mode="RGBA").save(Path(path))
