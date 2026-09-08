"""Optional person masking (``--remove-people``).

Candid shots have students standing in front of the board.  MediaPipe's selfie
segmentation is the only thing here that is a learned model -- everything else
in the pipeline is classical image processing -- and it is an optional extra,
so nothing breaks when it is not installed.
"""

from __future__ import annotations

import cv2
import numpy as np

DILATE_PX = 15


class MediaPipeMissing(RuntimeError):
    pass


def person_mask(img: np.ndarray, dilate_px: int = DILATE_PX, threshold: float = 0.5) -> np.ndarray:
    """Boolean mask of people in a float32 RGB image, dilated to catch fringes."""
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise MediaPipeMissing(
            "--remove-people needs the optional dependency: install with "
            "`uv sync --extra people` (or `pip install mediapipe`)."
        ) from exc

    rgb8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(rgb8)
    mask = result.segmentation_mask > float(threshold)

    r = max(0, int(dilate_px))
    if r:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask
