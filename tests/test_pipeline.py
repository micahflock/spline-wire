from dataclasses import replace

import cv2
import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import pins_from_turns
from splinewire.contact import contact_points, object_sign
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
    cv2.circle(img, (150, 150), 34, LINK_GRAY, -1, cv2.LINE_AA)             # a stray fiducial on
    cv2.circle(img, (150, 150), 21, 235, -1, cv2.LINE_AA)                   # a scrap of link
    if spec.fiducial == "ring":
        cv2.circle(img, (150, 150), 8, LINK_GRAY, -1, cv2.LINE_AA)
    m = measure(img, spec, FOCAL)
    assert len(m.order.gaps) == 1 and len(m.order.rejected) == 1
    assert len(m.warnings) == 1 and "not detected" in m.warnings[0]
    assert m.notes == ["ignored 1 look-alike detection(s) not on the chain"]   # a note, not a warning
    kept = np.delete(pins, 6, axis=0)
    assert compare_to_truth(m.pins_mm, kept)["max_error_mm"] < 0.05


def test_ring_sized_look_alike_one_pitch_past_the_end(spec):
    """A washer (or white disc) the size of a pin's fiducial, in line with the
    chain one pitch past its end: position and size can't tell, but the chain
    has one pin too many and the look-alike sits on the table, not on a link."""
    pins = s_curve_pins(spec)
    img, cam = _photo(spec, pins, tilt=20.0)
    d = (pins[-1] - pins[-2]) / np.linalg.norm(pins[-1] - pins[-2])
    extra = pins[-1] + spec.pitch_mm * d
    c = cam.project(extra[None])[0]
    r_px = np.linalg.norm(cam.project((extra + [spec.fiducial_mm / 2, 0])[None])[0] - c)
    ci = tuple(int(round(v * 16)) for v in c)
    cv2.circle(img, ci, int(round(r_px * 16)), 235, -1, cv2.LINE_AA, 4)
    if spec.fiducial == "ring":
        cv2.circle(img, ci, int(round(r_px * 0.4 * 16)), 205, -1, cv2.LINE_AA, 4)
    m = measure(img, spec, FOCAL)
    assert len(m.order.indices) == len(pins)
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05
    assert m.warnings == [] and len(m.notes) == 1      # on the table, not the chain going on


def test_pin_count_set_too_low_warns(spec):
    """Look-alikes left out are only a note, but pins of the chain left out
    because the pin count is set too low must be a warning."""
    pins = s_curve_pins(spec)
    img, _ = _photo(spec, pins, tilt=20.0)
    m = measure(img, replace(spec, n_pins=len(pins) - 2), FOCAL)
    assert len(m.order.indices) == len(pins) - 2
    assert any("go on past its end" in w for w in m.warnings)


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


def test_sharp_corner_end_to_end(spec):
    """The O-ring chain bends to 106° at a joint; with max_bend_deg raised
    the photo is ordered and measured through the corner."""
    turns = np.radians(np.r_[np.zeros(4), 106.0, 20.0, -106.0, np.zeros(4)])
    pins = pins_from_turns(spec.pitch_mm, turns)[: spec.n_pins]
    img, _ = _photo(spec, pins)
    narrow = replace(spec, max_bend_deg=80.0)                   # the print-in-place chain's limit
    assert len(measure(img, narrow, FOCAL).order.indices) < spec.n_pins     # loses the corner
    m = measure(img, replace(spec, max_bend_deg=125.0), FOCAL)
    assert len(m.order.indices) == spec.n_pins and m.order.gaps == []
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05



@pytest.mark.parametrize("concave, radius", [(False, 30.0), (True, 45.0)])
def test_side_picked_by_a_point_in_the_photo(spec, concave, radius):
    """A point picked on either side of the chain chooses that edge, and the
    contact edge drawn on the photo lies where the camera sees it."""
    pins = circle_wrap_pins(spec, radius, concave=concave)
    img, cam = _photo(spec, pins)
    mid = pins[len(pins) // 2]
    toward_center = cam.project(np.array([mid * 0.5]))[0]      # the chain curls toward the center
    away = cam.project(np.array([mid * 1.5]))[0]
    for point, expected in ((toward_center, "inside"), (away, "outside")):
        m = measure(img, spec, FOCAL, side=tuple(point))
        assert m.object_side == expected and m.side_px == tuple(point)
        np.testing.assert_allclose(m.contacts_mm, measure(img, spec, FOCAL, side=expected).contacts_mm)

        true_px = cam.project(contact_points(pins, spec.half_width_mm, object_sign(pins, expected)))
        drawn = m.to_photo_px(m.contacts_mm)
        # both run along the chain, maybe in opposite directions
        assert min(np.abs(drawn - true_px).max(), np.abs(drawn[::-1] - true_px).max()) < 2.0


def test_contact_edge_of_a_straight_chain_is_drawn_link_by_link(spec):
    """A straight chain has no width to fix a homography with, so the edge
    is placed on the photo by each link's own scale and direction: on the
    side picked, the half-width away from the pin line."""
    pins = np.c_[np.arange(spec.n_pins) * spec.pitch_mm, np.zeros(spec.n_pins)]
    img, cam = _photo(spec, pins, tilt=15.0)
    for sign in (1, -1):                                       # the object above the chain (+y), then below
        picked = cam.project(np.array([[60.0, sign * 15.0]]))[0]
        m = measure(img, spec, FOCAL, side=tuple(picked))
        drawn = m.to_photo_px(m.contacts_mm)
        true_px = cam.project(contact_points(pins, spec.half_width_mm, sign))
        assert min(np.abs(drawn - true_px).max(), np.abs(drawn[::-1] - true_px).max()) < 3.0
        # the wrong side of the chain would be twice the half-width away: ~70 px here
        wrong_px = cam.project(contact_points(pins, spec.half_width_mm, -sign))
        assert min(np.abs(drawn - wrong_px).max(), np.abs(drawn[::-1] - wrong_px).max()) > 40.0


def test_curve_stays_on_the_object_where_a_pin_was_missed(spec):
    """The pins either side of a missed one are not joined by a link. The
    middle of the line between them is 1.1 mm inside this pipe, and must not
    be given as a point on it."""
    radius = 30.0
    pins = circle_wrap_pins(spec, radius)
    img, cam = _photo(spec, pins)
    cv2.circle(img, tuple(int(v) for v in cam.project(pins[[6]])[0]), 30, LINK_GRAY, -1)
    m = measure(img, spec, FOCAL)
    assert len(m.order.gaps) == 1
    assert len(m.contacts_mm) == spec.n_pins - 3              # neither of the two links at the missed pin
    center, r = fit_circle(m.contacts_mm)
    assert r == pytest.approx(radius, abs=0.05)
    assert np.abs(np.linalg.norm(m.contacts_mm - center, axis=1) - radius).max() < 0.1


def test_a_chain_too_short_to_deskew_says_how_long_it_must_be(spec):
    """Five links fix the plane; a two-joint test piece has three."""
    short = replace(spec, n_pins=4)
    img, _ = _photo(short, s_curve_pins(short, bend_deg=20.0))
    with pytest.raises(ValueError, match=r"found 4 pins in a row.* at least 5 links \(6 pins\)"):
        measure(img, short, FOCAL)
    six = replace(spec, n_pins=6)
    img, _ = _photo(six, s_curve_pins(six, bend_deg=20.0))
    assert len(measure(img, six, FOCAL).pins_mm) == 6
    with pytest.raises(ValueError, match=r"at least 6 links \(7 pins\) when the focal length is unknown"):
        measure(img, six, None)


def test_pins_named_as_misfits_are_never_in_the_result(spec, monkeypatch):
    """Pins that turn out not to belong are dropped and the chain solved
    again, a few rounds at most. The last round's are dropped too."""
    import splinewire.pipeline as pipeline

    pins = s_curve_pins(spec)
    img, _ = _photo(spec, pins)
    named = []

    def an_end_pin_each_time(fiducials, order, rect, spec, keep=frozenset()):
        named.append(order.indices[0])
        return {order.indices[0]}

    monkeypatch.setattr(pipeline, "_misfits", an_end_pin_each_time)
    m = measure(img, spec, FOCAL)
    assert len(named) == pipeline._MISFIT_ROUNDS
    assert not set(named) & set(m.order.indices)
    assert len(m.order.indices) == spec.n_pins - len(named)


def test_a_busy_background_without_the_chain_says_so(spec):
    """Hundreds of dot-like spots and no chain among them: the message says
    what to change, not only that the chain was not found."""
    from splinewire.detect import Fiducial
    rng = np.random.default_rng(5)
    spots = [Fiducial(center_px=(float(x), float(y)), outer_axes_px=(float(d), float(d)), inner_axes_px=(0.0, 0.0),
                      angle_deg=0.0, contrast=float(c), polarity=1)
             for x, y, d, c in zip(rng.uniform(0, 4000, 700), rng.uniform(0, 3000, 700),
                                   np.exp(rng.uniform(np.log(10), np.log(40), 700)), rng.uniform(20, 80, 700))]
    img = np.zeros((3000, 4000), np.uint8)
    with pytest.raises(ValueError, match=r"pins in a row.*the background has 700 dot-like spots.*plain surface"):
        measure(img, spec, FOCAL, detected=spots)
