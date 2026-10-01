"""End to end on realistic photos that broke the original detector.

Each scene comes from splinewire/scene.py at a reduced size (3 MP) to keep
the suite fast; experiments/cv_benchmark.py runs the full-size sweep.
"""
from dataclasses import replace

import numpy as np
import pytest

from splinewire.pipeline import compare_to_truth, measure
from splinewire.scene import BAD_PRINT, PRESETS, render_scene
from splinewire.synthetic import circle_wrap_pins, s_curve_pins

SIZE = dict(image_size=(2000, 1500))


def _measure(spec, env, pins, seed=0):
    scene = render_scene(pins, spec, env, seed=seed)
    m = measure(scene.image, spec, scene.focal_px)
    return scene, m


def _assert_whole_chain(m, scene, pins, max_mm=0.1):
    assert len(m.order.indices) == len(pins) and not m.order.gaps
    got = m.pins_px
    d = np.linalg.norm(got[:, None] - scene.pins_px[None], axis=2)
    assert d.min(axis=1).max() < 1.5, "a pin in the chain is not a true pin"
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < max_mm


@pytest.mark.parametrize("name, overrides", [
    # Extrusion lines catching a lamp, which bridged rings to the white rim.
    ("grey-table", dict(tilt_direction_deg=224.0, roll_deg=132.0)),
    # A shadow edge across the chain.
    ("shadow", dict(tilt_direction_deg=224.0, roll_deg=132.0)),
    # Close up: seam blobs a few pixels high failed the strict ellipse test.
    ("close", dict(px_per_mm=15.0)),
    # Over-extruded, wobbly, blobby print.
    ("bad-print", {}),
    # Black links on a black table.
    ("black-table", {}),
])
def test_hard_lighting_and_prints(spec, name, overrides):
    env = replace(PRESETS[name], **{**SIZE, "px_per_mm": 7.0, **overrides})
    pins = s_curve_pins(spec)
    scene, m = _measure(spec, env, pins, seed=2)
    _assert_whole_chain(m, scene, pins)


def test_printed_letters_are_not_pins(spec):
    """A printed page: 'O', '0', '@' are ring-like (dark on light, many sizes)."""
    env = replace(PRESETS["text"], **SIZE, px_per_mm=7.0)
    pins = circle_wrap_pins(spec, 30.0)
    scene, m = _measure(spec, env, pins, seed=1)
    assert len(m.fiducials) > 30                   # plenty of look-alikes were detected...
    _assert_whole_chain(m, scene, pins)           # ...and none made it into the chain


def test_washers_near_the_chain_end_are_not_pins(spec):
    """Clutter always puts one washer in line with a chain end, 1-2 pitches out."""
    env = replace(PRESETS["clutter"], **SIZE, px_per_mm=7.0)
    pins = s_curve_pins(spec)
    for seed in (0, 1):
        scene, m = _measure(spec, env, pins, seed=seed)
        _assert_whole_chain(m, scene, pins)


def test_matte_filament_survives_the_lamp_reflection(spec):
    env = replace(PRESETS["glare-matte"], **SIZE, px_per_mm=7.0)
    pins = s_curve_pins(spec)
    scene, m = _measure(spec, env, pins)
    _assert_whole_chain(m, scene, pins)


def test_bad_print_is_still_accurate(spec):
    env = replace(PRESETS["daylight"], **SIZE, px_per_mm=7.0, print_quality=BAD_PRINT, tilt_deg=30.0)
    pins = circle_wrap_pins(spec, 45.0, concave=True)
    scene, m = _measure(spec, env, pins, seed=4)
    _assert_whole_chain(m, scene, pins, max_mm=0.15)
