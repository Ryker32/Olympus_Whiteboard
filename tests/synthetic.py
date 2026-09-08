"""A whiteboard photo generated in code, so the pipeline can be tested honestly
without waiting for shoot day.

The synthesis follows the same physical model the extractor inverts: translucent
ink composited over an unevenly lit board.  It is deliberately unkind -- a
lighting gradient, a vignette, a glare blob, sensor noise -- so that a change
that only works on a flat scan fails here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HEIGHT = 1600, 1000

# name, RGB in 0-255, peak coverage, word, (x, y) baseline
STROKE_SPECS = [
    ("black", (20, 20, 20), 0.90, "SENIORS", (90, 220)),
    ("yellow", (250, 215, 60), 0.70, "SUNSET", (640, 220)),
    ("blue", (120, 175, 235), 0.65, "LAKE", (1180, 220)),
    ("red", (215, 45, 55), 0.85, "PROM", (110, 720)),
    ("green", (40, 150, 80), 0.80, "FIELD", (620, 720)),
    ("purple", (140, 80, 190), 0.75, "LAST", (1180, 720)),
]

ACHROMATIC = {"black"}


@dataclass
class Stroke:
    name: str
    ink: np.ndarray       # float32 RGB in [0, 1]
    coverage: np.ndarray  # float32 in [0, 1]


def board_lighting(h: int = HEIGHT, w: int = WIDTH, seed: int = 0) -> np.ndarray:
    """A bare board: horizontal gradient 235->255, a vignette, one glare blob."""
    xs = np.linspace(235.0, 255.0, w, dtype=np.float32)
    board = np.repeat(xs[None, :], h, axis=0) / 255.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    board *= 1.0 - 0.10 * np.clip(r, 0, 1.4) ** 2

    glare = np.exp(-(((xx - 0.72 * w) ** 2 + (yy - 0.30 * h) ** 2) / (2 * (0.13 * w) ** 2)))
    board = np.clip(board + 0.11 * glare, 0, 1)

    rgb = np.dstack([board * 1.00, board * 0.995, board * 0.975])  # faint warm cast
    return np.ascontiguousarray(rgb.astype(np.float32))


def _word_coverage(word: str, org: tuple[int, int], peak: float,
                   h: int = HEIGHT, w: int = WIDTH) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(canvas, word, org, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 10, cv2.LINE_AA)
    cov = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 1.6)  # marker edges are soft
    return np.clip(cov, 0, 1) * float(peak)


def make_board(seed: int = 0, noise: float = 1.5 / 255.0):
    """Return ``(photo, strokes)`` -- a synthetic board photo and ground truth."""
    photo = board_lighting()
    strokes: list[Stroke] = []

    for name, rgb, peak, word, org in STROKE_SPECS:
        cov = _word_coverage(word, org, peak)
        ink = np.array(rgb, dtype=np.float32) / 255.0
        photo = cov[..., None] * ink + (1.0 - cov[..., None]) * photo
        strokes.append(Stroke(name, ink, cov))

    if noise:
        rng = np.random.default_rng(seed)
        photo = photo + rng.normal(0.0, noise, photo.shape).astype(np.float32)

    return np.ascontiguousarray(np.clip(photo, 0, 1).astype(np.float32)), strokes


def add_specks(photo: np.ndarray, seed: int = 1, count: int = 40) -> np.ndarray:
    """Dust and board scuffs: dark dots well under ``min_area``."""
    out = photo.copy()
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = int(rng.integers(40, WIDTH - 40))
        y = int(rng.integers(40, HEIGHT - 40))
        cv2.circle(out, (x, y), int(rng.integers(1, 4)), (0.25, 0.25, 0.25), -1, cv2.LINE_AA)
    return out


def add_ghost(photo: np.ndarray, org: tuple[int, int] = (150, 490),
              strength: float = 0.55, grey: float = 0.35):
    """A half-wiped smear: soft-edged, colourless, never as solid as real ink.

    Returns ``(photo, coverage)`` so tests can point at exactly where it went.
    """
    canvas = np.zeros(photo.shape[:2], dtype=np.uint8)
    cv2.putText(canvas, "OLD NOTES", org, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 255, 9, cv2.LINE_AA)
    cov = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 5.0) * float(strength)
    tone = np.float32([grey, grey, grey])
    out = np.clip(cov[..., None] * tone + (1 - cov[..., None]) * photo, 0, 1).astype(np.float32)
    return out, cov


def project_corners(w: int = WIDTH, h: int = HEIGHT, angle_deg: float = 25.0,
                    focal: float = 2200.0, distance: float = 2600.0,
                    canvas: tuple[int, int] = (1250, 1800)) -> np.ndarray:
    """Where the board's corners land when it is yawed ``angle_deg`` from frontal."""
    th = np.deg2rad(angle_deg)
    ch, cw = canvas
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):  # tl, tr, br, bl
        X, Y = sx * w / 2.0, sy * h / 2.0
        z = distance - X * np.sin(th)
        pts.append((focal * X * np.cos(th) / z + cw / 2.0, focal * Y / z + ch / 2.0))
    return np.array(pts, dtype=np.float32)


def place_on_wall(board: np.ndarray, angle_deg: float = 25.0,
                  canvas: tuple[int, int] = (1250, 1800)):
    """Composite the board onto a darker wall at an angle.  Returns ``(frame, corners)``."""
    h, w = board.shape[:2]
    ch, cw = canvas
    corners = project_corners(w, h, angle_deg, canvas=canvas)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, corners)

    warped = cv2.warpPerspective(board, m, (cw, ch), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), m, (cw, ch), flags=cv2.INTER_NEAREST)

    yy = np.linspace(0.52, 0.44, ch, dtype=np.float32)[:, None]
    wall = np.dstack([np.repeat(yy, cw, axis=1)] * 3) * np.float32([1.0, 0.97, 0.92])

    frame = warped * mask[..., None] + wall * (1 - mask[..., None])
    return np.ascontiguousarray(np.clip(frame, 0, 1).astype(np.float32)), corners
