import numpy as np
import pytest

from exposure_metrics import face_exposure, inner_box, whole_image_brightness

BOX = (100, 100, 100, 100)


def canvas(bg_L: float, face_L: float) -> np.ndarray:
    """A grey BGR image with a uniform 'face' square, at target CIE L* values."""
    import cv2
    def bgr_for(L):
        lab = np.array([[[L * 255 / 100, 128, 128]]], dtype=np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
    img = np.full((300, 300, 3), bgr_for(bg_L), dtype=np.uint8)
    img[100:200, 100:200] = bgr_for(face_L)
    return img


def test_dark_but_well_exposed_face_is_not_penalized():
    # A dark face (L*=25) on a dark background: whole-image brightness is low,
    # but nothing is clipped. v1 would call this "poorly lit"; v2 should not.
    img = canvas(bg_L=15, face_L=25)
    fe = face_exposure(img, BOX)
    assert whole_image_brightness(img) < 100
    assert fe.shadow_clip == 0.0
    assert fe.highlight_clip == 0.0
    assert fe.face_vs_scene > 0          # face is lit relative to its surroundings


def test_crushed_face_is_flagged():
    fe = face_exposure(canvas(bg_L=60, face_L=2), BOX)
    assert fe.shadow_clip == pytest.approx(1.0)
    assert fe.face_vs_scene < 0


def test_blown_face_is_flagged():
    assert face_exposure(canvas(bg_L=50, face_L=99), BOX).highlight_clip == pytest.approx(1.0)


def test_face_detail_measures_spread_not_level():
    flat = face_exposure(canvas(40, 40), BOX).face_detail
    img = canvas(40, 40)
    img[100:200, 100:150] = 30
    img[100:200, 150:200] = 160
    modeled = face_exposure(img, BOX).face_detail
    assert flat == pytest.approx(0.0, abs=1.0)
    assert modeled > 20


def test_inner_box_stays_inside_image():
    x0, y0, x1, y1 = inner_box((-20, -20, 100, 100), (300, 300))
    assert x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0


def test_box_outside_image_raises():
    with pytest.raises(ValueError):
        inner_box((500, 500, 10, 10), (300, 300))
