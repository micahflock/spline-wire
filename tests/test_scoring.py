"""Test targets (gauges, plaque poses) and scoring against them."""
import json
from dataclasses import replace

import numpy as np
import pytest

from splinewire.chain import chain_presets, load_chain_spec
from splinewire.cli import main
from splinewire.scoring import REPORT_SCHEMA, fit_circle, score, score_photo, target_side
from splinewire.synthetic import write_synthetic_photo
from splinewire.targets import GAUGES, PLAQUE_POSES, chain_on_gauge, plaque_pins, plaque_turns, targets_json


@pytest.fixture(scope="module")
def snap():
    return load_chain_spec(chain_presets()["snap"])


def _moved(pts, angle=1.1, shift=(31.0, -17.0), reverse=False):
    c, s = np.cos(angle), np.sin(angle)
    out = pts @ np.array([[c, -s], [s, c]]).T + shift
    return out[::-1] if reverse else out


def _chain_side(gauge):
    return (1 if gauge.concave else -1) if gauge.concave is not None else -1


@pytest.mark.parametrize("name", list(GAUGES))
def test_a_chain_lying_on_a_gauge_scores_as_the_model_predicts(snap, name):
    """Pins placed as the sleeves would sit on the edge, then moved and turned
    as a measurement would be: round gauges and the straight edge come out
    exact; the S is off only by the contact model's own error at its
    inflection (contact points on each pin's bisector)."""
    gauge = GAUGES[name]
    pins = _moved(chain_on_gauge(gauge, snap, _chain_side(gauge), start_mm=6.0), reverse=name.endswith("r40"))
    r = score(name, pins, snap, sign=None)
    exact = not name.startswith("s-")
    assert r["inner"]["max_mm"] < (0.001 if exact else 0.03)
    if gauge.radius_mm is not None:
        assert r["inner"]["radius_error_mm"] == pytest.approx(0.0, abs=1e-3)


def test_a_chain_off_scale_shows_in_the_radius_and_the_points(snap):
    """A 1% scale error (a wrong pitch) moves the pins 1% further out: on R40
    that is 0.44 mm more radius at the pins, which the score reports."""
    pins = chain_on_gauge(GAUGES["convex-r40"], snap, -1) * 1.01
    r = score("convex-r40", pins, snap, sign=1)
    assert r["inner"]["radius_error_mm"] == pytest.approx(0.01 * (40 + snap.half_width_mm), abs=0.01)
    assert r["inner"]["max_mm"] > 0.3


def test_the_wrong_side_is_twice_the_sleeve_off(snap):
    pins = chain_on_gauge(GAUGES["convex-r60"], snap, -1)
    right, wrong = score("convex-r60", pins, snap, sign=1), score("convex-r60", pins, snap, sign=-1)
    assert right["inner"]["max_mm"] < 0.001
    assert wrong["inner"]["radius_error_mm"] == pytest.approx(2 * snap.half_width_mm, abs=0.01)   # outside the pins


def test_the_ends_are_scored_with_and_without_them(snap):
    pins = chain_on_gauge(GAUGES["convex-r40"], snap, -1)
    r = score("convex-r40", pins, snap, sign=1)
    assert r["all"]["points"] == snap.n_pins and r["inner"]["points"] == snap.n_pins - 2


def test_a_chain_running_off_a_gauge_is_refused(snap):
    with pytest.raises(ValueError, match="runs off"):
        chain_on_gauge(GAUGES["straight"], snap, 1, start_mm=100.0)


def test_fit_circle_is_geometric():
    a = np.linspace(0, 1.2, 9)
    pts = np.c_[3 + 50 * np.cos(a), -2 + 50 * np.sin(a)]
    center, radius = fit_circle(pts)
    assert radius == pytest.approx(50.0, abs=1e-9) and center == pytest.approx([3, -2], abs=1e-9)


@pytest.mark.parametrize("pose", list(PLAQUE_POSES))
def test_plaque_poses_keep_their_links_apart(snap, pose):
    """Pins two or more apart along the chain stay farther apart than two
    sleeves, so the fused print has no links through one another; bends stay
    inside the link's range (+102/-127 degrees)."""
    pins = plaque_pins(pose, snap)
    assert np.abs(plaque_turns(pose, snap.n_pins)).max() <= 100.0
    d = np.linalg.norm(pins[:, None] - pins[None], axis=2)
    far = np.abs(np.arange(len(pins))[:, None] - np.arange(len(pins))[None]) >= 2
    assert d[far].min() > 2 * snap.half_width_mm + 1.0


def test_plaque_score_takes_a_cut_chain_against_the_part_it_matches(snap):
    truth = plaque_pins("plaque-tight", snap)
    r = score("plaque-tight", _moved(truth[2:], reverse=True), snap)
    assert r["pins_matched"] == snap.n_pins - 2 and r["first_pin"] == 2
    assert r["max_error_mm"] < 1e-9
    with pytest.raises(ValueError, match="has 13"):
        score("plaque-s", np.vstack([truth, truth[-1:] + 13]), snap)


def test_targets_json_is_complete(snap):
    doc = json.loads(json.dumps(targets_json(snap)))
    assert [g["name"] for g in doc["gauges"]] == list(GAUGES)
    assert [p["name"] for p in doc["plaques"]] == list(PLAQUE_POSES)
    assert target_side("concave-r60") == "outside" and target_side("s-r60") is None


def test_a_photo_is_scored_with_each_focal_length(snap, tmp_path):
    """A synthetic photo of the chain round the R40 disc: EXIF and estimated
    focal lengths both measure it within a few hundredths, and the best-fit
    focal length lands on the true one (26 mm)."""
    photo = tmp_path / "c40.jpg"
    write_synthetic_photo(photo, chain_on_gauge(GAUGES["convex-r40"], snap, -1, 5.0), snap, (3000, 2250),
                          tilt_deg=25.0)
    r = score_photo(photo.read_bytes(), photo.name, "convex-r40", snap, fit_focal=True)
    assert r["pins_found"] == snap.n_pins
    for key in ("exif", "estimated", "fitted"):
        assert r["focal"][key]["score"]["inner"]["max_mm"] < 0.03, key
    assert r["focal"]["fitted"]["focal_35mm"] == pytest.approx(26.0, abs=0.5)


def test_score_command_reads_the_test_pages_report(snap, tmp_path, capsys):
    dark = replace(snap, fiducial_shade="dark", n_pins=11)
    write_synthetic_photo(tmp_path / "T1.jpg", plaque_pins("plaque-s", dark), dark, (3000, 2250))
    report = {"schema": REPORT_SCHEMA, "tests": [
        {"id": "T1", "photo": "T1.jpg", "chain": "snap", "shade": "dark", "pins": 11, "target": "plaque-s"},
        {"id": "T2", "photo": "none.jpg", "chain": "snap", "pins": 13, "target": "object"},
    ]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert main(["score", "--report", str(tmp_path / "report.json"), "--json", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "T1" in out and "T2" not in out
    scores = json.loads((tmp_path / "s.json").read_text())
    assert scores[0]["focal"]["exif"]["score"]["max_error_mm"] < 0.02


def test_score_command_measures_the_chain_at_its_own_pitch_and_a_plaque_at_its_printed_one(snap, tmp_path, capsys):
    """A chain whose pitch came out 12.98 mm (phase 0): its photo on a gauge,
    measured at 13.00, comes out 0.15% large; at 12.98, exact. A plaque is
    measured at 13.00 whatever pitch the report gives the chain."""
    short = replace(snap, pitch_mm=12.98)
    write_synthetic_photo(tmp_path / "G.jpg", chain_on_gauge(GAUGES["concave-r100"], short, 1, 5.0), short, (3000, 2250))
    write_synthetic_photo(tmp_path / "P.jpg", plaque_pins("plaque-s", snap), snap, (3000, 2250))
    tests = [{"id": "G", "photo": "G.jpg", "chain": "snap", "pins": 13, "target": "concave-r100"},
             {"id": "P", "photo": "P.jpg", "chain": "snap", "pins": 13, "target": "plaque-s"}]
    radius = {}
    for pitch in (None, 12.98):
        doc = {"schema": REPORT_SCHEMA, "tests": [dict(t, pitch_mm=pitch) for t in tests]}
        (tmp_path / "r.json").write_text(json.dumps(doc))
        assert main(["score", "--report", str(tmp_path / "r.json"), "--json", str(tmp_path / "s.json")]) == 0
        g, p = json.loads((tmp_path / "s.json").read_text())
        radius[pitch] = g["focal"]["exif"]["score"]["inner"]["radius_error_mm"]
        assert p["focal"]["exif"]["score"]["max_error_mm"] < 0.02
    assert radius[12.98] == pytest.approx(0.0, abs=0.03)
    assert radius[None] == pytest.approx(0.02 / 12.98 * (100 - snap.half_width_mm), abs=0.05)


def test_a_focal_override_beats_the_exif(snap, tmp_path, capsys):
    """A photo whose EXIF says 24 mm but was taken at 26 mm (as the iPhone's
    in-page still, phase 1): the override measures it as the truth; the
    report's focal_override_35mm does the same."""
    photo = tmp_path / "P.jpg"
    from splinewire.camera import focal_px_from_35mm, look_at_plane
    from splinewire.synthetic import render_snap_photo, save_photo
    pins = plaque_pins("plaque-s", snap)
    cam = look_at_plane(focal_px_from_35mm(26.0, (3000, 2250)), (3000, 2250), 220.0, tilt_deg=38.0,
                        tilt_direction_deg=30.0, target_mm=tuple(pins.mean(axis=0)))
    save_photo(photo, render_snap_photo(pins, snap, cam), 24.0)
    r24 = score_photo(photo.read_bytes(), photo.name, "plaque-s", snap)
    r26 = score_photo(photo.read_bytes(), photo.name, "plaque-s", snap, focal_override_35mm=26.0)
    assert r26["focal"]["exif"]["source"] == "override"
    assert r26["focal"]["exif"]["score"]["max_error_mm"] < 0.03 < r24["focal"]["exif"]["score"]["max_error_mm"]
    doc = {"schema": REPORT_SCHEMA, "tests": [{"id": "P", "photo": "P.jpg", "chain": "snap", "pins": 13,
                                                "target": "plaque-s", "focal_override_35mm": 26.0}]}
    (tmp_path / "r.json").write_text(json.dumps(doc))
    assert main(["score", "--report", str(tmp_path / "r.json"), "--json", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())[0]["focal"]["exif"]["focal_35mm"] == pytest.approx(26.0, abs=0.01)


def test_score_command_needs_a_target(tmp_path, capsys):
    assert main(["score", str(tmp_path / "x.jpg")]) == 1
    assert "--target" in capsys.readouterr().err
    assert main(["score", str(tmp_path / "x.jpg"), "--target", "nope"]) == 1
    assert "no target 'nope'" in capsys.readouterr().err


def test_the_web_pages_targets_are_up_to_date(snap):
    """web/src/pipeline/targets.json (the test page's targets) is targets_json:
    run web/scripts/golden_scoring.py after changing a gauge or a pose."""
    from pathlib import Path
    web = json.loads((Path(__file__).parent.parent / "web" / "src" / "pipeline" / "targets.json").read_text())
    assert web == json.loads(json.dumps(targets_json(snap)))
