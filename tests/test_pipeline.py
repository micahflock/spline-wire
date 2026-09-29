import cv2
import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.pipeline import compare_to_truth, measure
from splinewire.synthetic import LINK_GRAY, circle_wrap_pins, render_photo, s_curve_pins
from tests.conftest import fit_circle

SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)


def _photo(spec, pins, tilt=30.0):
    cam = look_at_plane(FOCAL, SIZE, 180.0, tilt_deg=tilt, tilt_direction_deg=35.0,
                        roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    return render_photo(pins, spec, cam), cam


def test_s_curve_end_to_end(spec):
    pins = s_curve_pins(spec)
    img, _ = _photo(spec, pins)
    m = measure(img, spec, FOCAL)
    assert m.warnings == []
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05


@pytest.mark.parametrize("concave, side, radius", [(False, "inside", 30.0), (True, "outside", 45.0)])
def test_measured_curve_matches_the_target_circle(spec, concave, side, radius):
    pins = circle_wrap_pins(spec, radius, concave=concave)
    img, _ = _photo(spec, pins)
    m = measure(img, spec, FOCAL, side=side)
    center, r = fit_circle(m.contacts_mm)
    assert r == pytest.approx(radius, abs=0.05)
    assert np.abs(np.linalg.norm(m.contacts_mm - center, axis=1) - radius).max() < 0.1


def test_missing_pin_and_stray_ring(spec):
    pins = s_curve_pins(spec)
    img, cam = _photo(spec, pins)
    hidden = cam.project(pins[[6]])[0]
    cv2.circle(img, tuple(int(v) for v in hidden), 30, LINK_GRAY, -1)       # smudged ring
    cv2.circle(img, (150, 150), 22, 235, -1, cv2.LINE_AA)                   # stray ring on the table
    cv2.circle(img, (150, 150), 11, LINK_GRAY, -1, cv2.LINE_AA)
    m = measure(img, spec, FOCAL)
    assert len(m.order.gaps) == 1 and len(m.order.rejected) == 1
    assert len(m.warnings) == 2
    kept = np.delete(pins, 6, axis=0)
    assert compare_to_truth(m.pins_mm, kept)["max_error_mm"] < 0.05


def test_without_focal_length_warns_but_measures(spec):
    pins = s_curve_pins(spec)
    img, _ = _photo(spec, pins)
    m = measure(img, spec, None)
    assert any("focal length" in w for w in m.warnings)
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.1


def test_compare_to_truth_rejects_mirror_image():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 2)) * 20
    assert compare_to_truth(pts, pts[::-1])["max_error_mm"] < 1e-9     # reversed order is fine
    assert compare_to_truth(pts * [1, -1], pts)["max_error_mm"] > 1.0   # mirror is not
