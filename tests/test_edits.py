"""Hand corrections: remove a detection, add a pin."""
import cv2
import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.detect import detect_fiducials
from splinewire.edits import Edits, pin_to_pitch
from splinewire.pipeline import compare_to_truth, measure
from splinewire.synthetic import LINK_GRAY, render_photo, s_curve_pins

SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)
MISSED = [5, 6]            # two pins in a row: the ordering cannot step over both


@pytest.fixture
def scene(spec):
    pins = s_curve_pins(spec)
    cam = look_at_plane(FOCAL, SIZE, 180.0, tilt_deg=30.0, tilt_direction_deg=35.0,
                        roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    return pins, cam, render_photo(pins, spec, cam)


def _without(detected, cam, pins, which):
    """The detections minus those nearest the projected pins `which`."""
    gone = {int(np.argmin([np.hypot(*(np.array(f.center_px) - cam.project(pins[[k]])[0]))
                           for f in detected])) for k in which}
    return [f for i, f in enumerate(detected) if i not in gone]


def _clicks(cam, pins, which, jitter_px=3.0, seed=1):
    """Where a person would click on pins `which`: close, not exact."""
    rng = np.random.default_rng(seed)
    pts = cam.project(pins[which]) + rng.uniform(-jitter_px, jitter_px, (len(which), 2))
    return tuple((float(x), float(y)) for x, y in pts)


def test_two_missed_pins_in_a_row_cut_the_chain_until_they_are_added(spec, scene):
    pins, cam, img = scene
    detected = _without(detect_fiducials(img, spec), cam, pins, MISSED)
    cut = measure(img, spec, FOCAL, detected=detected)
    assert len(cut.order.indices) < len(pins) - 2            # an end is lost along with them

    edits = Edits(add_px=_clicks(cam, pins, MISSED))
    m = measure(img, spec, FOCAL, edits=edits, detected=detected)
    assert len(m.order.indices) == len(pins) and m.warnings == []
    assert [mp.snapped for mp in m.manual] == [True, True]    # the dots were there to snap to
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05


def test_clicks_are_snapped_to_the_fiducial_not_taken_as_they_are(spec, scene):
    pins, cam, img = scene
    detected = _without(detect_fiducials(img, spec), cam, pins, MISSED)
    off = Edits(add_px=_clicks(cam, pins, MISSED, jitter_px=6.0, seed=7))
    m = measure(img, spec, FOCAL, edits=off, detected=detected)
    for mp in m.manual:
        true = cam.project(pins[[MISSED[mp.edit]]])[0]
        assert np.hypot(*(np.array(m.fiducials[mp.fiducial].center_px) - true)) < 0.5
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.05


def test_a_pin_under_glare_can_be_added_by_hand(spec, scene):
    """No dot to see: the click is used as it is, and the result says so."""
    pins, cam, img = scene
    for k in MISSED:
        cv2.circle(img, tuple(int(v) for v in cam.project(pins[[k]])[0]), 34, LINK_GRAY, -1)
    assert len(measure(img, spec, FOCAL).order.indices) < len(pins) - 2

    exact = tuple((float(x), float(y)) for x, y in cam.project(pins[MISSED]))
    m = measure(img, spec, FOCAL, edits=Edits(add_px=exact))
    assert len(m.order.indices) == len(pins)
    assert [mp.snapped for mp in m.manual] == [False, False]
    assert any("fitted to the link lengths" in w for w in m.warnings)
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.1


def test_a_clicked_pin_is_fitted_to_the_pitch_of_its_links(spec, scene):
    """With no dot to snap to, a click 6 px off (~0.7 mm) still lands near the true pin:
    the links either side of it are exactly one pitch long."""
    pins, cam, img = scene
    k = 6
    cv2.circle(img, tuple(int(v) for v in cam.project(pins[[k]])[0]), 34, LINK_GRAY, -1)
    click = cam.project(pins[[k]])[0] + (4.2, -4.2)
    edits = Edits(add_px=((float(click[0]), float(click[1])),))

    m = measure(img, spec, FOCAL, edits=edits)
    assert [mp.snapped for mp in m.manual] == [False]
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.15
    assert m.rectification.residual_max_mm < 0.03          # every link, the clicked pin's too


def test_pins_in_a_row_keep_the_freedom_the_links_leave_but_stay_on_pitch(spec, scene):
    pins, cam, img = scene
    for k in MISSED:
        cv2.circle(img, tuple(int(v) for v in cam.project(pins[[k]])[0]), 34, LINK_GRAY, -1)
    clicks = _clicks(cam, pins, MISSED, jitter_px=4.0, seed=3)
    m = measure(img, spec, FOCAL, edits=Edits(add_px=clicks))
    assert len(m.order.indices) == len(pins) and m.rectification.residual_max_mm < 0.03
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.5


def test_pin_to_pitch_puts_a_pin_where_both_its_links_allow():
    step = lambda angle: 10.0 * np.array([np.cos(angle), np.sin(angle)])
    chain = np.zeros((4, 2))
    for k, angle in enumerate([0.0, 0.45, -0.45], start=1):      # a clear bend at pin 2
        chain[k] = chain[k - 1] + step(angle)
    clicked = chain.copy()
    clicked[2] += (0.8, -0.6)
    links = [(0, 1), (1, 2), (2, 3)]
    out = pin_to_pitch(clicked, links, [2], 10.0, sigma_click_mm=0.5)
    assert np.linalg.norm(out[2] - chain[2]) < 0.05 < np.linalg.norm(clicked[2] - chain[2])
    np.testing.assert_array_equal(out[[0, 1, 3]], clicked[[0, 1, 3]])      # only free pins move
    np.testing.assert_array_equal(pin_to_pitch(clicked, links, [], 10.0, 0.5), clicked)


def test_removing_an_end_pin(spec, scene):
    pins, cam, img = scene
    click = cam.project(pins[[-1]])[0] + (3.0, -2.0)
    m = measure(img, spec, FOCAL, edits=Edits(remove_px=((float(click[0]), float(click[1])),)))
    assert len(m.order.indices) == len(pins) - 1
    assert compare_to_truth(m.pins_mm, pins[:-1])["max_error_mm"] < 0.05


def test_a_removal_takes_one_detection_and_ignores_empty_space(spec, scene):
    pins, cam, img = scene
    detected = detect_fiducials(img, spec)
    near = cam.project(pins[[3]])[0]
    m = measure(img, spec, FOCAL, detected=detected, edits=Edits(
        remove_px=((float(near[0]), float(near[1])), (30.0, 30.0))))
    assert len(m.fiducials) == len(detected) - 1


def test_adding_where_a_pin_already_is_changes_nothing(spec, scene):
    pins, cam, img = scene
    base = measure(img, spec, FOCAL)
    click = cam.project(pins[[4]])[0] + (2.0, 2.0)
    m = measure(img, spec, FOCAL, edits=Edits(add_px=((float(click[0]), float(click[1])),)))
    assert len(m.fiducials) == len(base.fiducials) and len(m.order.indices) == len(pins)
    np.testing.assert_allclose(m.pins_mm, base.pins_mm, atol=0.02)


def test_a_chain_clicked_pin_by_pin_without_any_detection(spec, scene):
    pins, cam, img = scene
    detected = detect_fiducials(img, spec)
    everything = tuple((float(x), float(y)) for x, y in (f.center_px for f in detected))
    clicks = _clicks(cam, pins, list(range(len(pins))), jitter_px=3.0)
    m = measure(img, spec, FOCAL, detected=detected, edits=Edits(add_px=clicks, remove_px=everything))
    assert len(m.order.indices) == len(pins)
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.1


def test_a_stray_dot_off_the_chain_is_left_out_with_a_warning(spec, scene):
    """Pins added by hand still have to join the chain."""
    pins, cam, img = scene
    m = measure(img, spec, FOCAL, edits=Edits(add_px=((150.0, 150.0),)))
    assert len(m.order.indices) == len(pins)
    assert len(m.manual) == 1 and m.manual[0].fiducial in m.order.rejected
    assert any("were left out" in w for w in m.warnings)


@pytest.mark.parametrize("doc, message", [
    ("nope", "object"),
    ({"add": "x"}, "list"),
    ({"add": [[1, 2, 3]]}, "[x, y]"),
    ({"add": [["a", 2]]}, "[x, y]"),
    ({"remove": [[float("nan"), 1]]}, "finite"),
    ({"remove": [[5000, 10]]}, "outside"),
    ({"add": [[1, 1]] * 301}, "at most"),
])
def test_edits_from_the_browser_are_checked(doc, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        Edits.from_json(doc, (2000, 1500))


def test_edits_round_trip():
    e = Edits.from_json({"add": [[10.123456, 20]], "remove": [[1, 2]]}, (100, 100))
    assert e.add_px == ((10.12, 20.0),) and Edits.from_json(e.to_json()) == e
    assert not Edits() and e
