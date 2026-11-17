"""The browser worker's bridge: commands in, JSON out, few photos open at once."""
import io
import json
import subprocess
import sys
from dataclasses import asdict

import pytest
from PIL import Image

from splinewire import bridge
from splinewire.output import points_tsv
from splinewire.session import OUTPUT_NAMES
from splinewire.synthetic import s_curve_pins, write_synthetic_photo


@pytest.fixture(autouse=True)
def nothing_open():
    for photo in list(bridge._open):
        bridge.close(photo)
    yield
    for photo in list(bridge._open):
        bridge.close(photo)


@pytest.fixture(scope="module")
def photo_bytes(tmp_path_factory):
    from splinewire.chain import load_chain_spec
    from tests.conftest import REPO

    spec = load_chain_spec(REPO / "data" / "chain.yaml")
    path = tmp_path_factory.mktemp("bridge") / "chain.jpg"
    write_synthetic_photo(path, s_curve_pins(spec), spec, (1600, 1200))
    return path.read_bytes()


def call(cmd, data=None, **args):
    return json.loads(bridge.call(cmd, json.dumps(args), data))


def ok(cmd, data=None, **args):
    out = call(cmd, data, **args)
    assert out["ok"], out
    return out["result"]


def measure_args(spec, **more):
    return {"chain": asdict(spec), **more}


class FakeJsProxy:
    """What a Uint8Array looks like from Python under Pyodide."""
    def __init__(self, data):
        self.data = data

    def to_bytes(self):
        return self.data


def test_open_measure_outputs(dot_spec, photo_bytes):
    info = ok("open", FakeJsProxy(photo_bytes), photo="a", name="C:\\photos\\IMG_7.jpg")["info"]
    assert (info["width"], info["height"]) == (1600, 1200)
    doc = ok("measure", photo="a", options=measure_args(dot_spec))
    assert doc["ok"] and doc["result"]["pins"] == dot_spec.n_pins
    out = ok("outputs", photo="a")
    assert set(out["files"]) == {"json", "csv", "svg", "dxf", "fusion_csv", "tsv"}
    assert out["names"] == {k: v.format(stem="IMG_7") for k, v in OUTPUT_NAMES.items()}
    assert out["files"]["tsv"] == points_tsv(bridge._open["a"].last.measurement.contacts_mm)


def test_the_least_recently_used_photo_is_closed_first(dot_spec, photo_bytes):
    assert bridge.MAX_OPEN_PHOTOS == 2
    ok("open", photo_bytes, photo="a", name="a.jpg")
    ok("open", memoryview(photo_bytes), photo="b", name="b.jpg")
    ok("measure", photo="a", options=measure_args(dot_spec))      # a is now the most recent
    ok("open", photo_bytes, photo="c", name="c.jpg")
    assert ok("stats")["open"] == ["a", "c"]
    gone = call("measure", photo="b", options=measure_args(dot_spec))
    assert gone == {"ok": False, "error": {"kind": "closed", "message": "photo 'b' is not open"}}
    assert call("outputs", photo="b")["error"]["kind"] == "closed"
    # Opened again under the same id: it is back, and the oldest makes room.
    ok("open", photo_bytes, photo="b", name="b.jpg")
    assert ok("stats")["open"] == ["c", "b"]


def test_close_and_reopen(dot_spec, photo_bytes):
    ok("open", photo_bytes, photo="a", name="a.jpg")
    assert ok("close", photo="a") is None
    assert ok("close", photo="a") is None                        # closing twice is fine
    assert call("measure", photo="a", options=measure_args(dot_spec))["error"]["kind"] == "closed"
    assert call("measure", photo="never", options={})["error"]["kind"] == "closed"


def test_errors_for_people_and_bugs(dot_spec, photo_bytes):
    bad = call("open", b"not a photo", photo="x", name="notes.txt")
    assert bad["error"]["kind"] == "person" and "notes.txt is not a photo" in bad["error"]["message"]
    assert "type" not in bad["error"]
    assert ok("stats")["open"] == []
    ok("open", photo_bytes, photo="a", name="a.jpg")
    no_result = call("outputs", photo="a")
    assert no_result["error"] == {"kind": "person", "message": "this photo has no measurement yet"}
    # A measurement that finds no chain, or bad options, is a document, not a failure.
    doc = ok("measure", photo="a", options={"chain": {"pitch_mm": 10}})
    assert not doc["ok"] and "the chain is missing" in doc["error"]
    bug = call("nonsense")
    assert bug["error"]["kind"] == "bug" and bug["error"]["type"] == "LookupError"
    garbled = json.loads(bridge.call("measure", "{not json"))
    assert garbled["error"]["kind"] == "bug" and garbled["error"]["type"] == "TypeError"


def test_reopening_a_photo_with_bad_bytes_leaves_it_closed(photo_bytes):
    ok("open", photo_bytes, photo="a", name="a.jpg")
    assert call("open", b"junk", photo="a", name="a.jpg")["error"]["kind"] == "person"
    assert ok("stats")["open"] == []


def test_analyse_orders_points_from_a_measurement(dot_spec, photo_bytes):
    ok("open", photo_bytes, photo="a", name="a.jpg")
    view = ok("measure", photo="a", options=measure_args(dot_spec))["view"]
    pins = view["pins"]
    shuffled = pins[5:] + pins[:5]
    frame = {"points": [[p["x"], p["y"]] for p in shuffled], "diameters": [2 * p["r"] for p in shuffled],
             "width": view["width"], "height": view["height"], "chain": asdict(dot_spec), "focal_px": None}
    a = ok("analyse", **frame)
    ks = [shuffled[i]["k"] for i in a["order"]]
    assert ks in (list(range(len(pins))), list(range(len(pins)))[::-1])
    assert a["tilt_deg"] is not None and a["gaps"] == []
    assert call("analyse", points=[], chain={})["error"]["kind"] == "person"
    assert ok("analyse", **{**frame, "points": frame["points"][:2], "diameters": [1, 1]})["order"] == []


def test_a_blank_photo_opens_and_measures_to_no_chain(dot_spec):
    buf = io.BytesIO()
    Image.new("L", (320, 240), 90).save(buf, "PNG")
    assert ok("open", buf.getvalue(), photo="blank", name="blank.png")["info"]["format"] == "PNG"
    doc = ok("measure", photo="blank", options=measure_args(dot_spec))
    assert not doc["ok"] and doc["view"]["width"] == 320


def test_versions():
    v = ok("versions")
    assert set(v) >= {"splinewire", "python", "numpy", "scipy", "opencv", "pillow"}


def test_loading_the_bridge_imports_only_what_measuring_needs():
    """Every import costs seconds under Pyodide (S1: scipy.interpolate 2.1 s,
    ezdxf 1.4 s). ezdxf is for the DXF output only, pillow-heif and segno are
    not in the browser at all, and the pipeline needs nothing from SciPy but
    optimize and interpolate (which bring linalg, sparse, spatial, special
    and fft with them)."""
    code = ("import sys, json, splinewire.bridge; "
            "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('ezdxf', 'segno', 'scipy', 'matplotlib'))))")
    loaded = json.loads(subprocess.run([sys.executable, "-c", code], check=True,
                                       capture_output=True, text=True).stdout)
    assert not [m for m in loaded if m.split(".")[0] in ("ezdxf", "segno", "matplotlib")]
    unneeded = {"ndimage", "signal", "stats", "integrate", "io", "cluster", "odr", "datasets", "differentiate"}
    assert not [m for m in loaded if m.startswith("scipy.") and m.split(".")[1] in unneeded]
