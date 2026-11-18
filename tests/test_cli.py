import json

import pytest

from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.cli import main
from splinewire.synthetic import s_curve_pins, write_synthetic_photo
from tests.conftest import REPO

CHAIN = str(REPO / "data" / "chain.yaml")


def test_synth_then_measure_round_trip(tmp_path, capsys):
    assert main(["synth", "--chain", CHAIN, "--out", str(tmp_path / "syn"), "--shape", "pipe",
                 "--size", "2000x1500"]) == 0
    photo = tmp_path / "syn" / "pipe.jpg"
    truth = tmp_path / "syn" / "pipe-truth.json"
    assert main(["measure", str(photo), "--chain", CHAIN, "--out", str(tmp_path / "res"),
                 "--truth", str(truth)]) == 0

    doc = json.loads((tmp_path / "res" / "pipe.json").read_text())
    assert doc["schema"] == "spline-wire/points@1"
    assert doc["diagnostics"]["focal_estimated"] is False     # read from the JPEG's EXIF
    assert doc["truth_comparison"]["max_error_mm"] < 0.05
    assert len(doc["curve_points"]) == 12
    out = capsys.readouterr().out
    for suffix in (".json", "-curve.csv", "-curve.svg", "-curve.dxf", "-fusion-cm.csv", "-preview.jpg"):
        assert (tmp_path / "res" / f"pipe{suffix}").exists()
        assert f"pipe{suffix}" in out                           # every file written is named


def test_synth_realistic_environment_round_trip(tmp_path, monkeypatch):
    from dataclasses import replace

    from splinewire import scene
    small = {k: replace(v, image_size=(1600, 1200), px_per_mm=6.0) for k, v in scene.PRESETS.items()}
    monkeypatch.setattr(scene, "PRESETS", small)
    assert main(["synth", "--chain", CHAIN, "--out", str(tmp_path / "syn"), "--env", "shadow"]) == 0
    photo = tmp_path / "syn" / "s-curve-shadow.jpg"
    assert main(["measure", str(photo), "--chain", CHAIN, "--out", str(tmp_path / "res"),
                 "--truth", str(tmp_path / "syn" / "s-curve-truth.json")]) == 0
    doc = json.loads((tmp_path / "res" / "s-curve-shadow.json").read_text())
    assert doc["diagnostics"]["focal_estimated"] is False
    assert doc["truth_comparison"]["max_error_mm"] < 0.1


def test_test_part_writes_svg_and_truth(tmp_path):
    assert main(["test-part", "--chain", CHAIN, "--out", str(tmp_path)]) == 0
    svg = (tmp_path / "s-curve.svg").read_text()
    assert svg.startswith("<svg") and 'width="' in svg and "mm" in svg
    truth = json.loads((tmp_path / "s-curve-truth.json").read_text())
    assert len(truth["pin_points"]) == 13


def test_pins_sets_the_chain_length_for_one_photo(tmp_path, capsys):
    assert main(["synth", "--out", str(tmp_path), "--size", "2000x1500"]) == 0
    capsys.readouterr()
    assert main(["measure", str(tmp_path / "s-curve.jpg"), "--pins", "11", "--out", str(tmp_path / "res")]) == 0
    doc = json.loads((tmp_path / "res" / "s-curve.json").read_text())
    assert len(doc["pin_points"]) == 11                      # the snap-fit chain's file says 13
    assert capsys.readouterr().out.startswith("11 pins found")


def test_what_goes_wrong_is_said_in_a_line(tmp_path, capsys):
    """The program's own messages (no chain in the photo, a setting out of
    range) and files it cannot read: one line each and exit code 1."""
    from PIL import Image

    Image.new("L", (400, 300), 200).save(tmp_path / "blank.jpg")
    (tmp_path / "notes.jpg").write_text("not a photo")
    (tmp_path / "bad.yaml").write_text("pitch_mm: 10\nhalf_width_mm: 4\n")
    (tmp_path / "worse.yaml").write_text("- just\n- a list\n")
    spec = load_chain_spec(default_chain_path())           # measure's default: the snap-fit chain
    write_synthetic_photo(tmp_path / "chain.jpg", s_curve_pins(spec), spec, (2000, 1500))
    runs = {
        "found 0 fiducials": ["measure", str(tmp_path / "blank.jpg")],
        "is not a photo this program can read": ["measure", str(tmp_path / "notes.jpg")],
        "No such file": ["measure", str(tmp_path / "missing.jpg")],
        "n_pins must be >= 3": ["measure", str(tmp_path / "chain.jpg"), "--pins", "2"],
        "focal length must be a positive": ["measure", str(tmp_path / "chain.jpg"), "--focal-35mm", "0"],
        "bad.yaml has no fiducial_mm": ["measure", str(tmp_path / "chain.jpg"), "--chain", str(tmp_path / "bad.yaml")],
        "worse.yaml must hold the chain's settings": ["test-part", "--chain", str(tmp_path / "worse.yaml")],
    }
    for message, argv in runs.items():
        assert main([*argv, "--out", str(tmp_path / "out")]) == 1, argv
        err = capsys.readouterr().err
        assert err.startswith("error: ") and message in err and "Traceback" not in err, (argv, err)
        assert len(err.strip().splitlines()) == 1


def test_defaults_snap_for_measuring_oring_for_hardware(tmp_path, capsys):
    """measure and synth start with the snap-fit chain; synth's realistic
    scenes and the hardware commands with the O-ring dot chain they draw."""
    from splinewire.cli import DEFAULT_CHAIN, ORING_CHAIN
    assert DEFAULT_CHAIN.name == "chain-snap.yaml" and ORING_CHAIN.name == "chain.yaml"
    assert main(["test-part", "--out", str(tmp_path / "part")]) == 0
    truth = json.loads((tmp_path / "part" / "s-curve-truth.json").read_text())
    assert truth["pin_points"][1][0] == pytest.approx(10.0)    # the O-ring chain's 10 mm pitch
    assert main(["synth", "--env", "shadow", "--chain", str(DEFAULT_CHAIN), "--out", str(tmp_path / "s")]) == 1
    assert "the realistic scenes (--env) draw the O-ring dot chain" in capsys.readouterr().err


def test_plaque_command_writes_its_files(tmp_path, capsys):
    """test-plaque, as typed: the options reach the generator and the printing
    notes are shown."""
    for needed in ("shapely", "trimesh"):
        pytest.importorskip(needed)
    assert main(["test-plaque", "--shape", "pipe", "--radius", "40", "--out", str(tmp_path / "plaque")]) == 0
    assert (tmp_path / "plaque" / "pipe-plaque.stl").stat().st_size > 0
    assert len(json.loads((tmp_path / "plaque" / "pipe-truth.json").read_text())["pin_points"]) == 13
    assert "Start with WHITE" in capsys.readouterr().out
