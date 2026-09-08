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


def load_image(path: str | Path) -> np.ndarray:
    """Load a photo as float32 RGB in [0, 1], EXIF rotation applied."""
    path = Path(path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        _register_heif()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


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
