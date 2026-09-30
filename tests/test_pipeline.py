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


def test_ring_sized_look_alike_one_pitch_past_the_end(spec):
    """A washer the size of a ring, in line with the chain one pitch past its
    end: position and size can't tell, but the chain has one pin too many and
    the look-alike sits on the table, not on a link."""
    pins = s_curve_pins(spec)
    img, cam = _photo(spec, pins, tilt=20.0)
    d = (pins[-1] - pins[-2]) / np.linalg.norm(pins[-1] - pins[-2])
    extra = pins[-1] + spec.pitch_mm * d
    c = cam.project(extra[None])[0]
    r_px = np.linalg.norm(cam.project((extra + [spec.ring_outer_mm / 2, 0])[None])[0] - c)
    ci = tuple(int(round(v * 16)) for v in c)
    cv2.circle(img, ci, int(round(r_px * 16)), 235, -1, cv2.LINE_AA, 4)
    cv2.circle(img, ci, int(round(r_px * 0.4 * 16)), 205, -1, cv2.LINE_AA, 4)
    m = measure(img, spec, FOCAL)
    assert len(m.order.indices) == len(pins)
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05


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


def test_scale_fit_separates_print_scale_from_shape_error():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(13, 2)) * 30
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    measured = (truth @ rot.T) * 0.995 + [3.0, 4.0]      # a part printed 0.5% small
    t = compare_to_truth(measured, truth)
    assert t["scale"] == pytest.approx(0.995)
    assert t["max_error_scaled_mm"] < 1e-9
    assert t["max_error_mm"] > 0.1
