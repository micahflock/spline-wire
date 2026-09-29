import json

from PIL import Image

from splinewire.process import process_photo
from splinewire.selftest import run_selftest
from splinewire.synthetic import s_curve_pins, write_synthetic_photo, write_truth


def test_process_photo_writes_outputs_and_compares_truth(spec, tmp_path):
    pins = s_curve_pins(spec)
    photo, truth = tmp_path / "chain photo é.jpg", tmp_path / "truth.json"
    write_synthetic_photo(photo, pins, spec, (2000, 1500))
    write_truth(truth, pins)
    res = process_photo(photo, spec, tmp_path / "résultats", truth_path=truth)
    assert res.truth_comparison["max_error_mm"] < 0.05
    assert all(p.is_file() for p in res.outputs.values())   # non-ASCII paths included
    doc = json.loads(res.outputs["json"].read_text(encoding="utf-8"))
    assert doc["truth_comparison"] == res.truth_comparison


def test_heic_photo_with_exif(spec, tmp_path):
    pins = s_curve_pins(spec)
    jpg = tmp_path / "p.jpg"
    write_synthetic_photo(jpg, pins, spec, (2000, 1500))
    heic = tmp_path / "p.heic"
    with Image.open(jpg) as im:
        im.save(heic, exif=im.getexif(), quality=95)
    res = process_photo(heic, spec, tmp_path / "out")
    assert not res.measurement.rectification.focal_estimated


def test_focal_override_beats_exif(spec, tmp_path):
    pins = s_curve_pins(spec)
    photo = tmp_path / "p.jpg"
    write_synthetic_photo(photo, pins, spec, (2000, 1500), focal_35mm=26)
    res = process_photo(photo, spec, tmp_path / "out", focal_35mm=52)
    assert res.measurement.rectification.focal_px > 2900   # twice the EXIF focal (~1500 px)


def test_selftest_passes(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)   # GUI part is skipped off Windows without a display
    log = tmp_path / "selftest.log"
    assert run_selftest(log) == 0
    assert "OK: 0 failure(s)" in log.read_text()
