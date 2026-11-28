import json

import numpy as np
import pytest

from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.chainlink import check_character
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


def test_chain_link_command_makes_the_link_and_card(tmp_path, capsys):
    card = tmp_path / "card.svg"
    assert main(["chain-link", "--pitch", "12.987", "--half-width", "4.39", "--qr", str(card)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "https://micahflock.github.io/spline-wire/phone/?debug#cal=1;base=snap;p=12.987;h=4.39;k=k"
    assert card.read_text().startswith("<?xml") and "pitch 12.987 mm" in card.read_text()
    assert main(["chain-link", "--pitch", "13", "--shade", "dark", "--channel", "b", "--pins", "20",
                 "--page", "https://example.org/phone/"]) == 0
    assert capsys.readouterr().out.strip() == "https://example.org/phone/#cal=1;base=snap;p=13;shade=dark;ch=b;n=20;k=f"
    assert main(["chain-link", "--pitch", "14"]) == 1
    assert "more than 2% from the snap chain's 13 mm" in capsys.readouterr().err
    assert main(["chain-link", "--pitch", "13", "--base", "bolt"]) == 1
    assert 'no chain preset named "bolt"' in capsys.readouterr().err


def test_a_chain_off_its_pitch_measured_with_its_card(tmp_path, capsys):
    """synth --pitch renders a chain 1.5% long; measured as nominal, the curve is 1.5% small
    (a pitch error is a scale error of the same fraction); with --pitch or its card, it is right."""
    assert main(["synth", "--pitch", "13.2", "--out", str(tmp_path), "--shape", "pipe", "--size", "2000x1500"]) == 0
    truth = json.loads((tmp_path / "pipe-truth.json").read_text())
    assert np.linalg.norm(np.diff(np.array(truth["pin_points"]), axis=0), axis=1) == pytest.approx(13.2)
    photo, args = str(tmp_path / "pipe.jpg"), ["--truth", str(tmp_path / "pipe-truth.json")]

    def measured(*extra):
        out = tmp_path / f"res{len(extra)}{extra[-1][-4:] if extra else ''}"
        assert main(["measure", photo, "--out", str(out), *args, *extra]) == 0
        return json.loads((out / "pipe.json").read_text())

    nominal = measured()
    assert nominal["truth_comparison"]["scale"] == pytest.approx(13.0 / 13.2, abs=5e-4)
    assert nominal["truth_comparison"]["max_error_mm"] > 0.3
    by_pitch = measured("--pitch", "13.2")
    assert by_pitch["truth_comparison"]["max_error_mm"] < 0.05
    capsys.readouterr()
    card = "https://micahflock.github.io/spline-wire/phone/?debug#cal=1;base=snap;p=13.2;k=" + check_character("1;base=snap;p=13.2")
    by_card = measured("--chain-link", card)
    assert by_card["truth_comparison"] == by_pitch["truth_comparison"]
    assert "pitch 13.2 mm, half width 4.39 mm (calibrated, from the chain card)" in capsys.readouterr().out


def test_chain_link_on_measure_is_checked(tmp_path, capsys):
    assert main(["synth", "--out", str(tmp_path), "--size", "2000x1500"]) == 0
    photo = str(tmp_path / "s-curve.jpg")
    capsys.readouterr()
    assert main(["measure", photo, "--chain-link", "1;base=snap;p=13;k=0"]) == 1
    assert "code is damaged (check character)" in capsys.readouterr().err
    assert main(["measure", photo, "--chain", CHAIN, "--chain-link", "1;base=snap;p=13;k=1"]) == 1
    assert "not both" in capsys.readouterr().err
    assert main(["measure", photo, "--out", str(tmp_path / "res"),
                 "--chain-link", "#cal=1;base=snap;p=13;ch=b;k=" + check_character("1;base=snap;p=13;ch=b")]) == 0
    assert "pitch 13 mm, half width 4.39 mm, channel b (calibrated, from the chain card)" in capsys.readouterr().out
    assert main(["measure", photo, "--out", str(tmp_path / "res"), "--channel", "luma",       # --channel wins
                 "--chain-link", "#cal=1;base=snap;p=13;ch=b;k=" + check_character("1;base=snap;p=13;ch=b")]) == 0
    assert "half width 4.39 mm (calibrated" in capsys.readouterr().out
