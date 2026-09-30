import json

from splinewire.cli import main


def test_synth_then_measure_round_trip(tmp_path, capsys):
    chain = "data/chain.yaml"
    assert main(["synth", "--chain", chain, "--out", str(tmp_path / "syn"), "--shape", "pipe",
                 "--size", "2000x1500"]) == 0
    photo = tmp_path / "syn" / "pipe.jpg"
    truth = tmp_path / "syn" / "pipe-truth.json"
    assert main(["measure", str(photo), "--chain", chain, "--out", str(tmp_path / "res"),
                 "--truth", str(truth)]) == 0

    doc = json.loads((tmp_path / "res" / "pipe.json").read_text())
    assert doc["schema"] == "spline-wire/points@1"
    assert doc["diagnostics"]["focal_estimated"] is False     # read from the JPEG's EXIF
    assert doc["truth_comparison"]["max_error_mm"] < 0.05
    assert len(doc["curve_points"]) == 12
    for suffix in ("-curve.csv", "-curve.svg", "-preview.jpg"):
        assert (tmp_path / "res" / f"pipe{suffix}").exists()


def test_test_part_writes_svg_and_truth(tmp_path):
    assert main(["test-part", "--chain", "data/chain.yaml", "--out", str(tmp_path)]) == 0
    svg = (tmp_path / "s-curve.svg").read_text()
    assert svg.startswith("<svg") and 'width="' in svg and "mm" in svg
    truth = json.loads((tmp_path / "s-curve-truth.json").read_text())
    assert len(truth["pin_points"]) == 13
