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


def test_selftest_passes(tmp_path):
    log = tmp_path / "selftest.log"
    assert run_selftest(log) == 0
    assert "OK: 0 failure(s)" in log.read_text()


def test_fusion_outputs(spec, tmp_path):
    import ezdxf
    import numpy as np

    pins = s_curve_pins(spec)
    photo = tmp_path / "p.jpg"
    write_synthetic_photo(photo, pins, spec, (2000, 1500))
    res = process_photo(photo, spec, tmp_path / "out")
    curve = res.measurement.contacts_mm

    # ImportSplineCSV format: no header, x,y,z in centimeters
    rows = [line.split(",") for line in res.outputs["fusion_csv"].read_text().splitlines()]
    assert all(len(r) == 3 for r in rows)
    np.testing.assert_allclose(np.array(rows, float)[:, :2] * 10, curve, atol=1e-3)

    # DXF: millimeters, one spline passing through every curve point, plus the points
    doc = ezdxf.readfile(res.outputs["dxf"])
    assert doc.header["$INSUNITS"] == 4
    splines = doc.modelspace().query("SPLINE")
    points = doc.modelspace().query("POINT")
    assert len(splines) == 1 and len(points) == len(curve)
    samples = np.array(list(splines[0].construction_tool().approximate(20000)))[:, :2]
    gaps = [np.linalg.norm(samples - p, axis=1).min() for p in curve]
    assert max(gaps) < 0.01


def test_focal_source_and_photo_info(spec, tmp_path):
    from PIL import Image as PILImage

    pins = s_curve_pins(spec)
    with_exif = tmp_path / "exif.jpg"
    write_synthetic_photo(with_exif, pins, spec, (2000, 1500), focal_35mm=26)
    res = process_photo(with_exif, spec, tmp_path / "out", default_focal_35mm=40)
    assert res.focal_source == "exif"                    # EXIF beats the default
    assert res.info["focal_35mm"] == 26 and res.info["format"] == "JPEG"
    assert res.info["width"] == 2000

    stripped = tmp_path / "stripped.jpg"                 # what a metadata-stripping upload sends
    with PILImage.open(with_exif) as im:
        im.save(stripped, quality=95)
    res = process_photo(stripped, spec, tmp_path / "out", default_focal_35mm=26)
    assert res.focal_source == "default" and res.info["focal_35mm"] is None
    assert not res.measurement.rectification.focal_estimated
    assert process_photo(stripped, spec, tmp_path / "out").focal_source == "estimated"
    assert process_photo(stripped, spec, tmp_path / "out", focal_35mm=26).focal_source == "override"
