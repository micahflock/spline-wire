import numpy as np
import pytest

from splinewire.chain import ChainSpec, link_lengths, pins_from_turns


def test_spec_file_loads(spec):
    assert spec.pitch_mm == 10.0
    assert spec.n_pins == 13


@pytest.mark.parametrize("kwargs, match", [
    ({"pitch_mm": 0}, "pitch_mm"),
    ({"ring_inner_mm": 6.0}, "ring_inner_mm"),
    ({"ring_outer_mm": 12.0}, "ring_outer_mm"),
    ({"n_pins": 2}, "n_pins"),
])
def test_spec_rejects_bad_values(kwargs, match):
    base = dict(pitch_mm=10.0, half_width_mm=4.0, ring_outer_mm=5.0, ring_inner_mm=2.5, n_pins=13)
    with pytest.raises(ValueError, match=match):
        ChainSpec(**{**base, **kwargs})


def test_pins_from_turns_keeps_pitch_and_turns():
    turns = np.radians([10, -30, 45, 0, 20])
    pins = pins_from_turns(10.0, turns, start_mm=(3.0, 4.0), heading_rad=0.5)
    assert len(pins) == 7
    np.testing.assert_allclose(pins[0], [3.0, 4.0])
    np.testing.assert_allclose(link_lengths(pins), 10.0)
    d = np.diff(pins, axis=0)
    headings = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    np.testing.assert_allclose(np.diff(headings), turns, atol=1e-12)
