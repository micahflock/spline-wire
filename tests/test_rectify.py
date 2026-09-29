import numpy as np
import pytest

from splinewire.camera import look_at_plane
from splinewire.chain import pins_from_turns
from splinewire.pipeline import rigid_fit_errors
from splinewire.rectify import rectify_chain

SIZE = (2000, 1500)
FOCAL = 1500.0
SHAPES = {
    "s-curve": np.radians(np.r_[np.full(5, 14.0), np.full(6, -14.0)]),
    "pipe": np.full(11, 0.4),
    "gentle-arc": np.full(11, 10 / 300),
    "straight": np.zeros(11),
}


def _photo(turns, tilt=25.0):
    pins = pins_from_turns(10.0, turns)
    cam = look_at_plane(FOCAL, SIZE, 180.0, tilt_deg=tilt, tilt_direction_deg=30.0,
                        roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
    return pins, cam.project(pins)


def _links(n):
    return [(i, i + 1) for i in range(n - 1)]


@pytest.mark.parametrize("shape", SHAPES)
def test_recovers_exact_shape_without_noise(shape):
    pins, px = _photo(SHAPES[shape])
    r = rectify_chain(px, _links(len(pins)), 10.0, SIZE, FOCAL)
    # rigid_fit_errors disallows reflection, so this also checks handedness
    assert rigid_fit_errors(r.pins_mm, pins).max() < 1e-4
    assert r.residual_max_mm < 1e-6
    np.testing.assert_allclose(r.pins_mm[0], [0.0, 0.0])
    if shape != "straight":   # a straight chain says nothing about tilt
        assert r.tilt_deg == pytest.approx(25.0, abs=1e-3)


def test_output_axes_follow_the_photo():
    pins = pins_from_turns(10.0, np.full(11, 0.2))
    cam = look_at_plane(FOCAL, SIZE, 180.0, target_mm=tuple(pins.mean(axis=0)))
    r = rectify_chain(cam.project(pins), _links(len(pins)), 10.0, SIZE, FOCAL)
    np.testing.assert_allclose(r.pins_mm, pins - pins[0], atol=1e-6)


@pytest.mark.parametrize("shape", ["s-curve", "pipe", "gentle-arc"])
def test_half_pixel_noise_stays_well_under_a_millimetre(shape):
    rng = np.random.default_rng(1)
    pins, px = _photo(SHAPES[shape])
    errs = [
        rigid_fit_errors(
            rectify_chain(px + rng.normal(0, 0.5, px.shape), _links(len(pins)), 10.0, SIZE, FOCAL).pins_mm,
            pins,
        ).max()
        for _ in range(10)
    ]
    assert np.median(errs) < 0.4


def test_unknown_focal_length_is_estimated_for_a_curved_chain():
    pins, px = _photo(SHAPES["s-curve"])
    r = rectify_chain(px, _links(len(pins)), 10.0, SIZE, focal_px=None)
    assert r.focal_estimated
    assert r.focal_px == pytest.approx(FOCAL, rel=1e-3)
    assert rigid_fit_errors(r.pins_mm, pins).max() < 1e-3


def test_links_across_a_missing_pin_can_be_left_out():
    pins, px = _photo(SHAPES["s-curve"])
    keep = [i for i in range(len(pins)) if i != 6]
    links = [(a, b) for a, b in _links(len(keep)) if keep[b] - keep[a] == 1]
    r = rectify_chain(px[keep], links, 10.0, SIZE, FOCAL)
    assert rigid_fit_errors(r.pins_mm, pins[keep]).max() < 1e-4


def test_too_few_links_raises():
    pins, px = _photo(SHAPES["s-curve"])
    with pytest.raises(ValueError, match="links"):
        rectify_chain(px[:5], _links(5), 10.0, SIZE, FOCAL)
