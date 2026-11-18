"""Chains whose fiducials are darker than their links (fiducial_shade:
"dark", black on white links): detection reads the photo inverted, where
such a chain looks like any light-on-dark one."""
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import chain_presets, load_chain_spec, oring_chain_path, pins_from_turns, spec_from_dict
from splinewire.detect import detect_fiducials, snap_fiducial
from splinewire.pipeline import compare_to_truth, measure
from splinewire.session import PhotoSession
from splinewire.synthetic import render_photo, render_snap_photo, s_curve_pins, write_synthetic_photo

SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)
TURNS = np.radians([20, 35, -10, -40, 60, 15, -25, 40, -30, 10, -45])


def _chain(name: str):
    return load_chain_spec(chain_presets()["snap"] if name == "snap" else oring_chain_path())


def _photo(spec, pins, tilt, table=None):
    cam = look_at_plane(FOCAL, SIZE, 260.0 if spec.contact == "round" else 200.0, tilt_deg=tilt,
                        tilt_direction_deg=35.0, roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    img = render_snap_photo(pins, spec, cam, table_gray=table) if spec.contact == "round" else render_photo(pins, spec, cam)
    return img, cam


def test_presets_are_light_on_dark_and_old_settings_read_as_light():
    assert {load_chain_spec(p).fiducial_shade for p in chain_presets().values()} == {"light"}
    raw = {k: v for k, v in asdict(_chain("snap")).items() if k != "fiducial_shade"}
    assert spec_from_dict(raw).fiducial_shade == "light"           # saved before the setting existed
    assert spec_from_dict({**raw, "fiducial_shade": "dark"}).fiducial_shade == "dark"
    with pytest.raises(ValueError, match="fiducial_shade must be one of"):
        spec_from_dict({**raw, "fiducial_shade": "grey"})


@pytest.mark.parametrize("name", ["snap", "oring"])
@pytest.mark.parametrize("table", [240, 120, 30])
@pytest.mark.parametrize("tilt", [0.0, 40.0])
def test_dark_fiducials_on_light_links_measure_as_well(name, table, tilt):
    """Simulated: 0.008 mm worst pin, against 0.006 mm for the same chain light
    on dark (36 photos of both chains on light, grey and dark tables)."""
    spec = replace(_chain(name), fiducial_shade="dark")
    pins = pins_from_turns(spec.pitch_mm, TURNS)
    img, _ = _photo(spec, pins, tilt, table if name == "snap" else None)
    m = measure(img, spec, FOCAL)
    assert len(m.pins_mm) == spec.n_pins and m.warnings == []
    assert compare_to_truth(m.pins_mm, pins)["max_error_mm"] < 0.02
    assert detect_fiducials(img, replace(spec, fiducial_shade="light")) == []   # the setting is what finds them


def test_a_click_snaps_to_a_dark_rim():
    spec = replace(_chain("snap"), fiducial_shade="dark")
    pins = pins_from_turns(spec.pitch_mm, TURNS)
    img, cam = _photo(spec, pins, 25.0, 240)
    found = detect_fiducials(img, spec)
    target = cam.project(pins[6:7])[0]
    snapped = snap_fiducial(img, (target[0] + 3.0, target[1] - 2.0), spec, found[0])
    near = min(found, key=lambda f: np.hypot(*(np.array(f.center_px) - target)))
    assert snapped is not None and np.hypot(*(np.array(snapped.center_px) - near.center_px)) < 0.05


def test_the_session_detects_again_when_the_shade_changes(tmp_path):
    spec = replace(_chain("oring"), fiducial_shade="dark")
    path = tmp_path / "dark.jpg"
    write_synthetic_photo(path, s_curve_pins(spec), spec, SIZE)
    s = PhotoSession.open(path.read_bytes(), path.name)
    light = s.measure({"chain": asdict(replace(spec, fiducial_shade="light"))})
    assert not light["ok"] and "found 0 fiducials" in light["error"]
    dark = s.measure({"chain": asdict(spec)})
    json.dumps(dark)
    assert dark["ok"] and dark["result"]["pins"] == spec.n_pins
