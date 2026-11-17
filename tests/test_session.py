"""The in-memory session the browser build runs: bytes in, JSON out."""
import json
from dataclasses import asdict

import numpy as np
import pytest

import splinewire.session as session_module
from splinewire.process import process_photo
from splinewire.session import SCHEMA, PhotoSession
from splinewire.synthetic import s_curve_pins, write_synthetic_photo


@pytest.fixture
def photo(spec, tmp_path):
    path = tmp_path / "chain.jpg"
    pins = s_curve_pins(spec)
    write_synthetic_photo(path, pins, spec, (2000, 1500))
    return path, pins


def options(spec, **more) -> dict:
    return {"chain": asdict(spec), **more}


def test_measures_bytes_and_returns_a_json_document(spec, photo):
    path, pins = photo
    s = PhotoSession.open(path.read_bytes(), "chain.jpg")
    doc = s.measure(options(spec, truth={"name": "truth.json", "pin_points": pins.tolist()}))
    json.dumps(doc)                                     # nothing in it but JSON types
    assert doc["schema"] == SCHEMA and doc["ok"] and doc["error"] is None
    assert doc["result"]["pins"] == spec.n_pins
    assert doc["result"]["focal_source"] == "exif"
    assert doc["result"]["truth"]["max_error_mm"] < 0.05
    assert len(doc["curve"]["points"]) == doc["result"]["curve_points"]
    assert [p["k"] for p in doc["view"]["pins"]] == list(range(spec.n_pins))
    assert (doc["view"]["width"], doc["view"]["height"]) == (2000, 1500)


def test_same_numbers_and_files_as_process_photo(spec, photo, tmp_path):
    path, _ = photo
    res = process_photo(path, spec, tmp_path / "out")
    s = PhotoSession.open(path.read_bytes(), path.name)
    doc = s.measure(options(spec))
    assert np.allclose(doc["curve"]["pins"], res.measurement.pins_mm, atol=1e-4)
    texts = s.outputs()
    for kind in ("json", "csv", "svg", "fusion_csv"):
        assert res.outputs[kind].read_bytes().decode("utf-8") == texts[kind], kind
    assert "SPLINE" in texts["dxf"] and "CURVE_POINTS" in texts["dxf"]   # each DXF has its own GUID
    assert texts["tsv"].startswith("x_mm\ty_mm\n")


def test_an_edit_measures_again_without_detecting_again(spec, photo, monkeypatch):
    path, _ = photo
    s = PhotoSession.open(path.read_bytes(), path.name)
    first = s.measure(options(spec))
    monkeypatch.setattr(session_module, "detect_fiducials",
                        lambda *a: pytest.fail("detected the same photo twice"))
    pin = first["view"]["pins"][0]
    doc = s.measure(options(spec, edits={"remove": [[pin["x"], pin["y"]]], "add": []}))
    assert doc["ok"] and doc["result"]["pins"] == spec.n_pins - 1
    assert doc["view"]["removed"] == [{"x": pin["x"], "y": pin["y"], "edit": 0}]
    side = s.measure(options(spec, side=[10.0, 10.0]))
    assert side["view"]["side_px"] == [10.0, 10.0]


def test_no_chain_is_a_document_not_an_exception(spec):
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("L", (640, 480), 128).save(buf, "JPEG")
    s = PhotoSession.open(buf.getvalue(), "blank.jpg")
    doc = s.measure(options(spec))
    assert not doc["ok"] and "found 0 fiducials" in doc["error"]
    assert doc["result"] is None and doc["curve"] is None
    assert doc["view"]["pins"] == [] and doc["view"]["width"] == 640
    with pytest.raises(ValueError, match="no measurement yet"):
        s.outputs()


def test_bad_input_is_explained(spec, photo):
    path, _ = photo
    with pytest.raises(ValueError, match="notes.txt is not a photo"):
        PhotoSession.open(b"not a photo", "notes.txt")
    s = PhotoSession.open(path.read_bytes(), path.name)
    assert "must be an object with a chain" in s.measure({})["error"]
    assert "side must be" in s.measure(options(spec, side="left"))["error"]
    assert "between 5 and 500" in s.measure(options(spec, saved_focal_35mm=4000))["error"]
    assert "the chain is missing" in s.measure({"chain": {"pitch_mm": 10}})["error"]


def test_saved_focal_is_used_only_without_exif(spec, photo):
    path, _ = photo
    s = PhotoSession.open(path.read_bytes(), path.name)
    assert s.measure(options(spec, saved_focal_35mm=40))["result"]["focal_source"] == "exif"
    assert s.measure(options(spec, focal_override_35mm=40))["result"]["focal_source"] == "override"
    s.exif_focal_px = None
    assert s.measure(options(spec, saved_focal_35mm=40))["result"]["focal_source"] == "default"
    assert s.measure(options(spec))["result"]["focal_source"] == "estimated"
