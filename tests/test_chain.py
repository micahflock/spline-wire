import numpy as np
import pytest

from splinewire.chain import ChainSpec, link_lengths, load_chain_spec, pins_from_turns, spec_from_dict


def test_spec_file_loads():
    from tests.conftest import REPO
    from splinewire.chain import load_chain_spec
    spec = load_chain_spec(REPO / "data" / "chain.yaml")
    assert spec.pitch_mm == 10.0 and spec.n_pins == 13
    assert (spec.fiducial, spec.fiducial_mm) == ("dot", 5.0)
    assert spec.max_bend_deg == 125.0                               # the O-ring chain


@pytest.mark.parametrize("kwargs, match", [
    ({"pitch_mm": 0}, "pitch_mm"),
    ({"fiducial": "ring", "ring_inner_mm": 6.0}, "ring_inner_mm"),
    ({"fiducial": "ring"}, "ring_inner_mm"),
    ({"fiducial_mm": 12.0}, "fiducial_mm"),
    ({"fiducial": "square"}, "fiducial"),
    ({"n_pins": 2}, "n_pins"),
    ({"max_bend_deg": 170.0}, "max_bend_deg"),
    ({"contact": "point"}, "contact"),
])
def test_spec_rejects_bad_values(kwargs, match):
    base = dict(pitch_mm=10.0, half_width_mm=4.0, fiducial_mm=5.0, n_pins=13)
    with pytest.raises(ValueError, match=match):
        ChainSpec(**{**base, **kwargs})


def test_older_ring_spec_still_loads():
    old = dict(pitch_mm=10.0, half_width_mm=4.0, ring_outer_mm=5.0, ring_inner_mm=2.0, n_pins=13)
    s = spec_from_dict(old)
    assert (s.fiducial, s.fiducial_mm, s.ring_inner_mm) == ("ring", 5.0, 2.0)
    assert spec_from_dict({**old, "fiducial_mm": 5.0, "fiducial": "dot"}).fiducial == "dot"
    assert s.max_bend_deg == 80.0 and s.contact == "edge"            # files from before the settings
    assert spec_from_dict({**old, "max_bend_deg": 125}).max_bend_deg == 125.0


def test_pins_from_turns_keeps_pitch_and_turns():
    turns = np.radians([10, -30, 45, 0, 20])
    pins = pins_from_turns(10.0, turns, start_mm=(3.0, 4.0), heading_rad=0.5)
    assert len(pins) == 7
    np.testing.assert_allclose(pins[0], [3.0, 4.0])
    np.testing.assert_allclose(link_lengths(pins), 10.0)
    d = np.diff(pins, axis=0)
    headings = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    np.testing.assert_allclose(np.diff(headings), turns, atol=1e-12)


def test_the_snap_fit_chain_is_the_default_preset():
    """The snap-fit chain is what everything starts with (the owner's choice,
    2026-11-18); data/chain.yaml stays the O-ring dot chain, under "oring"."""
    from splinewire.chain import DEFAULT_PRESET, chain_presets, default_chain_path, oring_chain_path
    presets = chain_presets()
    assert DEFAULT_PRESET == "snap" and next(iter(presets)) == "snap"
    assert default_chain_path() == presets["snap"] and default_chain_path().name == "chain-snap.yaml"
    assert presets["oring"] == oring_chain_path() and oring_chain_path().name == "chain.yaml"
    assert load_chain_spec(default_chain_path()).contact == "round"
    assert load_chain_spec(oring_chain_path()).fiducial == "dot"
