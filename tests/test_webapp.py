"""The local web app over real HTTP."""
import io
import json
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest
from PIL import Image

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.synthetic import (
    LINK_GRAY, render_photo, s_curve_pins, save_photo, write_synthetic_photo, write_truth,
)
from splinewire.webapp.server import AppState, _safe_name, start_server


@pytest.fixture
def served(tmp_path):
    app = AppState(tmp_path / "ws", settings={}, persist=False)
    server = start_server(app, 0)
    yield app, server
    server.shutdown()
    server.server_close()


def call(server, path, data=None, token=None, host=None, ip="127.0.0.1"):
    req = urllib.request.Request(f"http://{ip}:{server.port}{path}", data=data,
                                 method="POST" if data is not None else "GET")
    if token:
        req.add_header("X-Token", token)
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_upload_measure_and_fetch_results(served, spec, tmp_path):
    app, server = served
    pins = s_curve_pins(spec)
    photo, truth = tmp_path / "IMG_0001.JPG", tmp_path / "truth.json"
    write_synthetic_photo(photo, pins, spec, (2000, 1500))
    write_truth(truth, pins)
    assert call(server, "/api/truth?name=truth.json", truth.read_bytes(), app.token)[0] == 200
    status, body = call(server, "/api/upload?name=IMG_0001.JPG", photo.read_bytes(), app.token)
    assert status == 200
    pid = json.loads(body)["id"]
    assert app.wait_idle(60)

    detail = json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    assert detail["status"] == "done" and detail["source"] == "computer"
    assert detail["result"]["truth"]["max_error_mm"] < 0.05
    assert detail["result"]["focal_source"] == "exif"
    assert len(detail["curve"]["points"]) == detail["result"]["curve_points"]
    assert set(detail["downloads"]) >= {"dxf", "csv", "svg", "json"}
    # the uploaded bytes are stored unchanged (so is their metadata)
    assert app.photos[pid].path.read_bytes() == photo.read_bytes()

    status, tsv = call(server, f"/api/photo/{pid}/points.tsv", token=app.token)
    assert status == 200 and tsv.decode().startswith("x_mm\ty_mm")
    assert call(server, f"/api/photo/{pid}/preview.jpg?t={app.token}")[1][:2] == b"\xff\xd8"
    assert call(server, f"/api/photo/{pid}/download/dxf?t={app.token}")[0] == 200


def test_bad_photo_reports_a_readable_error(served, tmp_path):
    app, server = served
    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(blank)
    pid = json.loads(call(server, "/api/upload?name=blank.jpg", blank.read_bytes(), app.token)[1])["id"]
    assert app.wait_idle(30)
    detail = json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    assert detail["status"] == "error" and detail["error"].startswith("found 0 fiducials")


def test_token_and_host_are_required(served):
    app, server = served
    assert call(server, "/api/state")[0] == 403
    assert call(server, "/api/state", token="wrong")[0] == 403
    assert call(server, "/api/state", token=app.token, host="attacker.example")[0] == 421
    assert call(server, "/api/state", token=app.token)[0] == 200
    assert call(server, "/api/ping")[0] == 200                  # used to detect a running copy


def test_settings_are_validated(served):
    app, server = served
    bad = {"chain": {**app.settings["chain"], "pitch_mm": 3}}
    status, body = call(server, "/api/settings", json.dumps(bad).encode(), app.token)
    assert status == 400 and b"pitch_mm" in body
    assert app.settings["chain"]["pitch_mm"] != 3
    ok = {"side": "outside", "phone_focal_35mm": "24"}
    assert call(server, "/api/settings", json.dumps(ok).encode(), app.token)[0] == 200
    assert app.settings["side"] == "outside" and app.settings["phone_focal_35mm"] == 24


def test_phone_side_is_limited(served):
    """From another device: pages and uploads work; settings, downloads and quit don't."""
    app, server = served
    if not server.lan_ips:
        pytest.skip("no LAN address in this environment")
    ip = server.lan_ips[0]
    local_state = lambda: json.loads(call(server, "/api/state", token=app.token)[1])
    assert local_state()["network"]["phone_seen"] is None
    status, html = call(server, "/", ip=ip)                     # desktop page redirects to /phone
    assert status == 200 and b"Take photo" in html and b"Install Fusion" not in html
    assert call(server, "/phone", ip=ip)[0] == 200
    assert call(server, "/api/local-token", ip=ip)[0] == 403
    assert call(server, "/api/state", token=app.token, ip=ip)[0] == 200
    assert call(server, "/api/settings", b"{}", app.token, ip=ip)[0] == 403
    assert call(server, "/api/photo/1/edits", b"{}", app.token, ip=ip)[0] == 403
    assert call(server, "/api/photo/1/image.jpg", token=app.token, ip=ip)[0] == 403
    assert call(server, "/api/quit", b"", app.token, ip=ip)[0] == 403
    assert call(server, "/api/firewall/allow", b"", app.token, ip=ip)[0] == 403
    assert "network" not in json.loads(call(server, "/api/state", token=app.token, ip=ip)[1])
    assert local_state()["network"]["phone_seen"] is not None   # the desktop page says "phone reached"


def test_network_status_for_the_desktop_page(served):
    app, server = served
    state = json.loads(call(server, "/api/state", token=app.token)[1])
    assert set(state["network"]) == {"phone_seen", "firewall"}
    assert "ok" in state["network"]["firewall"]
    assert call(server, "/api/firewall/check", b"", app.token)[0] == 200


@pytest.mark.parametrize("name, safe", [
    ("IMG_0001.HEIC", "IMG_0001.HEIC"),
    ("../../evil.jpg", "evil.jpg"),
    ("C:\\Users\\x\\photo.jpg", "photo.jpg"),
    ("photo<>:*?.jpg", "photo_.jpg"),
    ("", "photo"),
])
def test_upload_names_are_made_safe(name, safe):
    assert _safe_name(name) == safe


def _photo_with_glare(spec, path, hidden):
    """A photo with the fiducials of pins `hidden` washed out, as glare would; returns
    where every pin is in the photo."""
    size = (2000, 1500)
    pins = s_curve_pins(spec)
    cam = look_at_plane(focal_px_from_35mm(26, size), size, 180.0, tilt_deg=30.0, tilt_direction_deg=35.0,
                        roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    img = render_photo(pins, spec, cam)
    px = cam.project(pins)
    for k in hidden:
        cv2.circle(img, tuple(int(v) for v in px[k]), 34, LINK_GRAY, -1)
    save_photo(path, img, 26)
    return pins, px


def test_pins_can_be_added_and_removed_by_hand(served, dot_spec, tmp_path):
    spec = dot_spec
    app, server = served
    pins, px = _photo_with_glare(spec, tmp_path / "glare.jpg", hidden=[5, 6])
    pid = json.loads(call(server, "/api/upload?name=glare.jpg", (tmp_path / "glare.jpg").read_bytes(), app.token)[1])["id"]
    assert app.wait_idle(60)
    detail = lambda: json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    post = lambda doc: call(server, f"/api/photo/{pid}/edits", json.dumps(doc).encode(), app.token)

    before = detail()
    assert before["status"] == "done" and before["result"]["pins"] < len(pins) - 2
    view = before["view"]
    assert (view["width"], view["height"]) == (2000, 1500)
    assert len(view["pins"]) == before["result"]["pins"]
    assert [p["k"] for p in view["pins"]] == list(range(len(view["pins"])))
    assert view["edits"] == {"add": [], "remove": []} and view["removed"] == []

    # the photo for the editor is the upright photo, unmarked
    status, jpeg = call(server, f"/api/photo/{pid}/image.jpg?t={app.token}")
    assert status == 200 and Image.open(io.BytesIO(jpeg)).size == (2000, 1500)

    # add the two washed-out pins by clicking where they are, a few pixels off
    clicks = [[float(x) + 3, float(y) - 2] for x, y in px[[5, 6]]]
    assert post({"add": clicks, "remove": []})[0] == 200
    assert app.wait_idle(60)
    after = detail()
    assert after["result"]["pins"] == len(pins)
    assert any("fitted to the link lengths" in w for w in after["result"]["warnings"])
    added = [p for p in after["view"]["pins"] if "edit" in p]
    assert sorted(p["edit"] for p in added) == [0, 1] and not any(p["snapped"] for p in added)
    assert after["view"]["edits"]["add"] == [[round(c[0], 2), round(c[1], 2)] for c in clicks]
    assert after["edits"] == 2                               # shown in the photo list
    doc = json.loads(app.photos[pid].result.outputs["json"].read_text(encoding="utf-8"))
    assert doc["edits"]["add"] == after["view"]["edits"]["add"]     # saved with the points

    truth = tmp_path / "truth.json"
    write_truth(truth, pins)
    call(server, "/api/truth?name=truth.json", truth.read_bytes(), app.token)
    assert app.wait_idle(60)
    assert detail()["result"]["truth"]["max_error_mm"] < 0.3       # clicks were 3.6 px (~0.4 mm) off, nothing to snap to
    call(server, "/api/truth/clear", b"", app.token)
    assert app.wait_idle(60)

    # remove an end pin: it comes back as a ghost to click again
    end = [float(v) for v in px[-1]]
    assert post({"add": clicks, "remove": [end]})[0] == 200
    assert app.wait_idle(60)
    d = detail()
    assert d["result"]["pins"] == len(pins) - 1
    assert d["view"]["removed"] == [{"x": round(end[0], 2), "y": round(end[1], 2), "edit": 0}]

    assert post({"add": [], "remove": []})[0] == 200          # reset
    assert app.wait_idle(60)
    assert detail()["result"]["pins"] == before["result"]["pins"]


def test_bad_edits_are_refused(served, dot_spec, tmp_path):
    spec = dot_spec
    app, server = served
    write_synthetic_photo(tmp_path / "p.jpg", s_curve_pins(spec), spec, (2000, 1500))
    pid = json.loads(call(server, "/api/upload?name=p.jpg", (tmp_path / "p.jpg").read_bytes(), app.token)[1])["id"]
    assert app.wait_idle(60)
    post = lambda path, body: call(server, path, body, app.token)
    assert post(f"/api/photo/{pid}/edits", b'{"add": [[5000, 10]]}')[0] == 400      # outside the photo
    assert post(f"/api/photo/{pid}/edits", b'{"add": [["x", 1]]}')[0] == 400
    assert post(f"/api/photo/{pid}/edits", b"not json")[0] == 400
    assert post("/api/photo/999/edits", b"{}")[0] == 404
    assert app.photos[pid].edits.add_px == ()                                         # none were kept


def test_a_photo_that_failed_can_still_be_fixed_by_hand(served, tmp_path):
    """The editor must not vanish when measuring fails (it is when it is most needed)."""
    app, server = served
    blank = tmp_path / "blank.jpg"
    Image.new("RGB", (800, 600), (200, 200, 200)).save(blank)
    pid = json.loads(call(server, "/api/upload?name=blank.jpg", blank.read_bytes(), app.token)[1])["id"]
    assert app.wait_idle(30)
    detail = lambda: json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    d = detail()
    assert d["status"] == "error" and d["view"]["pins"] == [] and (d["view"]["width"], d["view"]["height"]) == (800, 600)
    call(server, f"/api/photo/{pid}/edits", json.dumps({"add": [[100, 100], [180, 110]], "remove": []}).encode(), app.token)
    assert app.wait_idle(30)
    d = detail()
    assert d["status"] == "error"                                   # too few pins to measure
    assert [(p["x"], p["y"], p["edit"]) for p in d["view"]["unused"]] == [(100.0, 100.0, 0), (180.0, 110.0, 1)]
