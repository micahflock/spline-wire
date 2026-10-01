"""The local web app over real HTTP."""
import json
import urllib.error
import urllib.request

import pytest

from splinewire.synthetic import s_curve_pins, write_synthetic_photo, write_truth
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
    from PIL import Image
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
