"""Face-region exposure metrics for auditing how an image model lights different people.

Why not whole-image brightness? Mean brightness of the full frame mixes together
skin tone, background, wardrobe and medium (a dark oil painting vs a bright studio
photo). A correctly exposed portrait of a dark-skinned person in a dark room is
"dark" by that measure, and calling it "poorly lit" turns skin tone into a defect.

These metrics look only at the face and ask exposure questions a photographer
would ask:

* shadow_clip   - share of face pixels crushed to near-black (detail lost)
* highlight_clip- share of face pixels blown to near-white (detail lost)
* face_detail   - spread of lightness inside the face (flat/muddy vs. modeled)
* face_vs_scene - is the face lit relative to its surroundings (subject separation)
* face_L        - median CIE L* of the face, reported for context, NOT scored:
                  darker skin legitimately has lower L*.

Lightness is CIE L* (0-100), computed with OpenCV's Lab conversion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

SHADOW_L = 5.0      # L* below this counts as crushed shadow
HIGHLIGHT_L = 95.0  # L* above this counts as blown highlight
INNER = 0.6         # use the central 60% of the face box to avoid hair and background


@dataclass(frozen=True)
class FaceExposure:
    face_L: float
    shadow_clip: float
    highlight_clip: float
    face_detail: float
    face_vs_scene: float

    def as_dict(self) -> dict:
        return asdict(self)


def lightness(bgr: np.ndarray) -> np.ndarray:
    """CIE L* in 0..100 for an 8-bit BGR image."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab[..., 0].astype(np.float32) * (100.0 / 255.0)


def inner_box(box: tuple[int, int, int, int], shape: tuple[int, ...], frac: float = INNER) -> tuple[int, int, int, int]:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    iw, ih = w * frac, h * frac
    x0, y0 = int(max(0, cx - iw / 2)), int(max(0, cy - ih / 2))
    x1, y1 = int(min(shape[1], cx + iw / 2)), int(min(shape[0], cy + ih / 2))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"face box {box} does not overlap the image")
    return x0, y0, x1, y1


def face_exposure(bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> FaceExposure:
    L = lightness(bgr)
    x0, y0, x1, y1 = inner_box(face_box, L.shape)
    face = L[y0:y1, x0:x1]

    mask = np.ones(L.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    scene = L[mask]

    return FaceExposure(
        face_L=float(np.median(face)),
        shadow_clip=float(np.mean(face < SHADOW_L)),
        highlight_clip=float(np.mean(face > HIGHLIGHT_L)),
        face_detail=float(np.percentile(face, 90) - np.percentile(face, 10)),
        face_vs_scene=float(np.median(face) - np.median(scene)),
    )


def whole_image_brightness(bgr: np.ndarray) -> float:
    """The v1 metric, kept for comparison: mean grayscale value, 0..255."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
