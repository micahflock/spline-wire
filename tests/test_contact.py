import numpy as np
import pytest

from splinewire.chain import ChainSpec
from splinewire.contact import contact_points, object_sign, spline_samples
from splinewire.synthetic import circle_wrap_pins, s_curve_pins


def _spec(**kw):
    base = dict(pitch_mm=10.0, half_width_mm=4.0, fiducial_mm=5.0, n_pins=13)
    return ChainSpec(**{**base, **kw})


def test_convex_wrap_contacts_lie_on_the_target_circle():
    spec = _spec()
    pins = circle_wrap_pins(spec, radius_mm=25.0)
    c = contact_points(pins, spec.half_width_mm, object_sign(pins, "inside"))
    assert len(c) == spec.n_pins - 1                 # one per link
    np.testing.assert_allclose(np.linalg.norm(c, axis=1), 25.0, atol=1e-9)
    # the pins themselves stand off the circle by more than the half-width (chord sag)
    assert np.linalg.norm(pins, axis=1).min() > 25.0 + spec.half_width_mm + 0.4


def test_concave_contacts_lie_on_the_target_circle():
    spec = _spec()
    pins = circle_wrap_pins(spec, radius_mm=40.0, concave=True)
    c = contact_points(pins, spec.half_width_mm, object_sign(pins, "outside"))
    assert len(c) == spec.n_pins                     # one per pin
    np.testing.assert_allclose(np.linalg.norm(c[1:-1], axis=1), 40.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(c[[0, -1]], axis=1), 40.0, atol=0.05)


def test_object_side():
    pins = circle_wrap_pins(_spec(), radius_mm=25.0)   # counter-clockwise: curls left
    assert object_sign(pins, "inside") == 1
    assert object_sign(pins, "outside") == -1
    assert object_sign(pins[::-1], "inside") == -1


def test_zero_width_chain_contacts_are_on_the_pin_line():
    spec = _spec(half_width_mm=0.0)
    pins = s_curve_pins(spec)
    c = contact_points(pins, 0.0, 1)
    for p in c:  # every contact point lies on some link segment
        d = min(_dist_to_segment(p, a, b) for a, b in zip(pins[:-1], pins[1:]))
        assert d < 1e-9


def test_spline_passes_through_points():
    pts = circle_wrap_pins(_spec(), radius_mm=25.0)
    dense = spline_samples(pts, per_segment=10)
    np.testing.assert_allclose(dense[::10], pts, atol=1e-9)
    # a spline through points on a circle stays close to the circle between them
    assert np.abs(np.linalg.norm(dense, axis=1) - np.linalg.norm(pts[0])).max() < 0.05


def _dist_to_segment(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return np.linalg.norm(p - (a + t * (b - a)))
