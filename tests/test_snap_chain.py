"""The snap-fit chain (data/chain-snap.yaml): thin white rims on tall pins,
round sleeves that touch the target."""
from dataclasses import replace

import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import ChainSpec, chain_presets, load_chain_spec, pins_from_turns
from splinewire.detect import detect_fiducials, snap_fiducial
from splinewire.pipeline import compare_to_truth, measure
from splinewire.synthetic import circle_wrap_pins, render_snap_photo
from tests.conftest import fit_circle

SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)
WHITE_PAPER, GREY, BLACK = 250, 120, 40


@pytest.fixture
def snap() -> ChainSpec:
    return load_chain_spec(chain_presets()["snap"])


def _photo(spec, pins, table, tilt=30.0, distance=230.0):
    cam = look_at_plane(FOCAL, SIZE, distance, tilt_deg=tilt, tilt_direction_deg=35.0,
                        roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    return render_snap_photo(pins, spec, cam, table_gray=table), cam


def test_preset(snap):
    assert (snap.pitch_mm, snap.half_width_mm, snap.contact) == (13.0, 4.68, "round")
    assert (snap.fiducial, snap.fiducial_mm, snap.ring_inner_mm) == ("ring", 7.2, 5.48)
    assert load_chain_spec(chain_presets()["default"]).contact == "edge"


@pytest.mark.parametrize("table", [WHITE_PAPER, GREY, BLACK])
@pytest.mark.parametrize("tilt", [0.0, 35.0])
def test_every_rim_is_found_whatever_lies_beyond_it(snap, table, tilt):
    """At a tilt the table shows right next to each rim's far side, and the
    first pin has no sleeve round it at all: on white paper the rims have
    no outer edge there."""
    pins = pins_from_turns(snap.pitch_mm, np.radians([20, 35, -10, -40, 60, 15, -25, 90, -30, 10, -95]))
    img, cam = _photo(snap, pins, table, tilt)
    found = detect_fiducials(img, snap)
    assert len(found) == snap.n_pins
    m = measure(img, snap, FOCAL, detected=found)
    assert m.warnings == []
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.03
    sizes = np.array([f.outer_axes_px[0] for f in found]) * m.rectification.depth_mm / FOCAL
    assert np.median(sizes) == pytest.approx(snap.fiducial_mm, rel=0.05)   # reported as the rim, not its hole


@pytest.mark.parametrize("concave, side, radius", [(False, "inside", 30.0), (True, "outside", 60.0)])
def test_measured_curve_matches_the_target_circle(snap, concave, side, radius):
    pins = circle_wrap_pins(snap, radius, concave=concave)
    img, _ = _photo(snap, pins, WHITE_PAPER)
    m = measure(img, snap, FOCAL, side=side)
    assert len(m.contacts_mm) == snap.n_pins                             # one per sleeve
    center, r = fit_circle(m.contacts_mm)
    assert r == pytest.approx(radius, abs=0.05)
    assert np.abs(np.linalg.norm(m.contacts_mm - center, axis=1) - radius).max() < 0.05


def test_a_bend_past_the_limit_breaks_the_chain_rather_than_misordering_it(snap):
    """At 120 degrees the pins either side of a joint are one pitch apart,
    like a link: nothing in the photo says which order they come in."""
    pins = pins_from_turns(snap.pitch_mm, np.radians([20, 35, -10, -40, 60, 15, -25, 90, -30, 10, -120]))
    img, _ = _photo(snap, pins, GREY, tilt=20.0)
    m = measure(img, snap, FOCAL)
    assert len(m.pins_mm) == snap.n_pins - 1
    assert any("found 12 pins" in w for w in m.warnings)
    assert compare_to_truth(m.pins_mm, pins[:-1])["max_error_mm"] < 0.03


def test_a_click_snaps_to_a_rim(snap):
    pins = circle_wrap_pins(snap, 30.0)
    img, cam = _photo(snap, pins, WHITE_PAPER)
    found = detect_fiducials(img, snap)
    truth = cam.project(pins)
    for k in (0, 6):
        click = truth[k] + [6.0, -5.0]
        hit = snap_fiducial(img, tuple(click), snap, found[0])
        assert hit is not None
        assert np.hypot(*(np.array(hit.center_px) - truth[k])) < 2.0
    assert snap_fiducial(img, (40.0, 40.0), snap, found[0]) is None       # bare table


def test_wide_rings_keep_their_own_detector(snap, monkeypatch):
    import splinewire.detect as detect
    called = []
    monkeypatch.setattr(detect, "detect_rings", lambda image, ratio: called.append(ratio) or [])
    detect.detect_fiducials(np.zeros((50, 50), np.uint8), replace(snap, fiducial_mm=5.0, ring_inner_mm=2.0))
    assert called == [0.4]
