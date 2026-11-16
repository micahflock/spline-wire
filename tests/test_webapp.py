"""The local web app over real HTTP."""
import io
import json
import socket
import threading
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
from splinewire.webapp import server as server_module
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


def raw(server, request: bytes) -> bytes:
    """Send bytes as they are and return everything the server answers,
    for requests no well-behaved client makes (a body cut short)."""
    with socket.create_connection(("127.0.0.1", server.port), timeout=20) as s:
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)
        answer = b""
        while chunk := s.recv(65536):
            answer += chunk
    return answer


def upload_request(server, app, name: str, length: int) -> bytes:
    return (f"POST /api/upload?name={name} HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
            f"X-Token: {app.token}\r\nContent-Length: {length}\r\n\r\n").encode()


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
    # Shown as JPEGs only: the full-size drawing (37 MB for a 12 MP photo) is not kept per photo.
    assert app.photos[pid].result.preview is None and app.photos[pid].preview_jpeg


def test_an_upload_cut_short_is_refused_and_not_saved(served, dot_spec, tmp_path):
    """A phone that drops off the Wi-Fi mid-upload must not leave half a photo in the list."""
    app, server = served
    write_synthetic_photo(tmp_path / "p.jpg", s_curve_pins(dot_spec), dot_spec, (2000, 1500))
    data = (tmp_path / "p.jpg").read_bytes()
    answer = raw(server, upload_request(server, app, "cut.jpg", len(data)) + data[: len(data) // 2])
    assert answer.startswith(b"HTTP/1.1 400") and b"cut off" in answer
    assert app.photos == {} and not list((tmp_path / "ws").rglob("*cut.jpg"))
    assert call(server, "/api/upload?name=whole.jpg", data, app.token)[0] == 200     # and whole, it is taken
    assert app.wait_idle(60)


def test_an_upload_over_the_limit_gets_an_answer_not_a_broken_connection(served, monkeypatch):
    app, server = served
    monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 1000)
    status, body = call(server, "/api/upload?name=big.jpg", b"x" * 3000, app.token)
    assert status == 400 and b"too large" in body and app.photos == {}
    # Its body was read and dropped, so the same connection carries on with the next request.
    again = (f"GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n\r\n").encode()
    answer = raw(server, upload_request(server, app, "big.jpg", 3000) + b"x" * 3000 + again)
    assert answer.count(b"HTTP/1.1 400") == 1 and answer.count(b"HTTP/1.1 200") == 1
    # One far over the limit is not read at all: answered, then hung up on.
    answer = raw(server, upload_request(server, app, "huge.jpg", 10**9) + b"x" * 100 + again)
    assert answer.count(b"HTTP/1.1") == 1 and b"Connection: close" in answer


def test_a_file_that_is_not_a_photo_says_so(served):
    app, server = served
    pid = json.loads(call(server, "/api/upload?name=notes.txt", b"not an image at all", app.token)[1])["id"]
    assert app.wait_idle(30)
    detail = json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    assert detail["status"] == "error" and "is not a photo this program can read" in detail["error"]
    assert str(app.workspace) not in detail["error"]                 # a message, not a traceback with paths


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

    sharp = {"chain": {**app.settings["chain"], "max_bend_deg": "125"}}
    assert call(server, "/api/settings", json.dumps(sharp).encode(), app.token)[0] == 200
    assert app.spec().max_bend_deg == 125.0
    older_page = {"chain": {k: v for k, v in app.settings["chain"].items() if k != "max_bend_deg"}}
    assert call(server, "/api/settings", json.dumps(older_page).encode(), app.token)[0] == 200
    assert app.spec().max_bend_deg == 125.0                     # kept, not zeroed


@pytest.mark.parametrize("path, body", [
    ("/api/settings", b"[1, 2]"),
    ("/api/settings", b'{"chain": "x"}'),
    ("/api/settings", b'{"chain": {"pitch_mm": "wide"}}'),
    ("/api/settings", b'{"phone_focal_35mm": [24]}'),
    ("/api/settings", b"\xff\xfe"),
    ("/api/truth?name=t.json", b"[1, 2]"),
    ("/api/truth?name=t.json", b'{"pin_points": [[0, 0], [1, 1]]}'),
    ("/api/truth?name=t.json", b'{"pin_points": [[0, 0], [1, "a"], [2, 2]]}'),
    ("/api/truth?name=t.json", b"\xff\xfe"),
    ("/api/open-folder", b'"photos"'),
])
def test_malformed_requests_are_refused_with_a_message(served, path, body):
    """Anything a page could not have sent is a 400 with a reason, never a crash (500)."""
    app, server = served
    before = json.dumps(app.settings, sort_keys=True)
    status, answer = call(server, path, body, app.token)
    assert status == 400 and json.loads(answer)["error"]
    assert json.dumps(app.settings, sort_keys=True) == before


@pytest.mark.parametrize("saved", [
    None, [], "text", {"token": 5}, {"token": ""}, {"chain": []}, {"chain": {"pitch_mm": "abc"}},
    {"chain": {"pitch_mm": -1}}, {"chain": {"n_pins": 1.5}}, {"side": "left"}, {"phone_focal_35mm": "24"},
    {"focal_override_35mm": 4000}, {"truth": 7},
])
def test_a_damaged_settings_file_does_not_stop_the_app(tmp_path, saved, dot_spec):
    """Settings are a convenience. Whatever is in the file (an older version's,
    hand-edited, cut short), the app starts, with defaults for what it can't use."""
    app = AppState(tmp_path / "ws", settings=saved, persist=False)
    assert isinstance(app.token, str) and len(app.token) >= 16
    assert app.spec() == dot_spec
    assert app.settings["side"] == "inside" and app.settings["truth"] is None
    assert app.settings["phone_focal_35mm"] is None and app.settings["focal_override_35mm"] is None


def test_settings_that_are_fine_are_kept(tmp_path):
    saved = {"token": "a-token-from-last-time", "side": "outside", "phone_focal_35mm": 24, "truth": "t.json",
             "chain": {"pitch_mm": 12.5, "n_pins": 21, "max_bend_deg": 80}}
    app = AppState(tmp_path / "ws", settings=saved, persist=False)
    assert app.token == "a-token-from-last-time" and app.settings["side"] == "outside"
    assert app.settings["phone_focal_35mm"] == 24 and app.settings["truth"] == "t.json"
    spec = app.spec()
    assert (spec.pitch_mm, spec.n_pins, spec.max_bend_deg, spec.fiducial) == (12.5, 21, 80.0, "dot")


def test_settings_from_before_the_dot_are_brought_forward(tmp_path):
    """Every earlier version saved a ring, ring_outer_mm by ring_inner_mm. Left at the
    old default, the chain follows today's (a dot); a ring set up on purpose stays one."""
    old_default = {"pitch_mm": 10.0, "half_width_mm": 4.0, "ring_outer_mm": 5.0, "ring_inner_mm": 2.0, "n_pins": 13}
    spec = AppState(tmp_path / "a", settings={"chain": old_default}, persist=False).spec()
    assert (spec.fiducial, spec.fiducial_mm, spec.ring_inner_mm) == ("dot", 5.0, 0.0)
    custom = {**old_default, "ring_outer_mm": 6.0, "ring_inner_mm": 2.4}
    spec = AppState(tmp_path / "b", settings={"chain": custom}, persist=False).spec()
    assert (spec.fiducial, spec.fiducial_mm, spec.ring_inner_mm) == ("ring", 6.0, 2.4)


def test_chain_presets_and_what_touches_the_object(served):
    app, server = served
    state = json.loads(call(server, "/api/state", token=app.token)[1])
    presets = state["settings"]["chain_presets"]
    assert presets["default"]["contact"] == "edge" and "n_pins" not in presets["snap"]
    pins = app.settings["chain"]["n_pins"]
    assert call(server, "/api/settings", json.dumps({"chain": presets["snap"]}).encode(), app.token)[0] == 200
    spec = app.spec()
    assert (spec.pitch_mm, spec.contact, spec.fiducial, spec.n_pins) == (13.0, "round", "ring", pins)
    older_page = {"chain": {k: v for k, v in app.settings["chain"].items() if k != "contact"}}
    assert call(server, "/api/settings", json.dumps(older_page).encode(), app.token)[0] == 200
    assert app.spec().contact == "round"                        # kept
    bad = {"chain": {**app.settings["chain"], "contact": "point"}}
    assert call(server, "/api/settings", json.dumps(bad).encode(), app.token)[0] == 400


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
    assert call(server, "/api/photo/1/options", b"{}", app.token, ip=ip)[0] == 404      # allowed; no such photo
    assert call(server, "/api/photo/1/image-small.jpg", token=app.token, ip=ip)[0] == 404
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
    ("100%41.jpg", "100_41.jpg"),
    ("", "photo"),
])
def test_upload_names_are_made_safe(name, safe):
    assert _safe_name(name) == safe


def test_an_upload_name_is_decoded_once(served):
    app, server = served
    pid = json.loads(call(server, "/api/upload?name=100%2541%20mm.jpg", b"x", app.token)[1])["id"]
    assert app.photos[pid].name == "100_41 mm.jpg"            # %25 is a percent sign, not the start of %41 = "A"
    assert app.wait_idle(30)


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


def test_pin_count_and_side_are_set_per_photo(served, dot_spec, tmp_path):
    """Each photo has its own pin count and object side; the last ones used
    are what the next photo gets."""
    spec = dot_spec
    app, server = served
    size = (2000, 1500)
    pins = s_curve_pins(spec)
    cam = look_at_plane(focal_px_from_35mm(26, size), size, 180.0, tilt_deg=25.0, tilt_direction_deg=35.0,
                        roll_deg=-15.0, target_mm=tuple(pins.mean(axis=0)))
    save_photo(tmp_path / "p.jpg", render_photo(pins, spec, cam), 26)
    data = (tmp_path / "p.jpg").read_bytes()
    state = lambda: json.loads(call(server, "/api/state", token=app.token)[1])
    detail = lambda pid: json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])
    options = lambda pid, doc: call(server, f"/api/photo/{pid}/options", json.dumps(doc).encode(), app.token)

    assert state()["defaults"] == {"n_pins": spec.n_pins, "side": "inside"}
    first = json.loads(call(server, "/api/upload?name=a.jpg&pins=11&side=outside", data, app.token)[1])["id"]
    assert app.wait_idle(60)
    d = detail(first)
    assert (d["n_pins"], d["side"], d["result"]["object_side"]) == (11, "outside", "outside")
    assert d["result"]["pins"] == 11 and d["result"]["warnings"]       # the chain, cut to 11 pins
    assert state()["defaults"] == {"n_pins": 11, "side": "outside"}

    # the right count clears the warning, and becomes the default; the first photo keeps its side
    assert options(first, {"n_pins": spec.n_pins})[0] == 200
    second = json.loads(call(server, "/api/upload?name=b.jpg", data, app.token)[1])["id"]
    assert app.wait_idle(60)
    assert detail(first)["result"]["pins"] == spec.n_pins and detail(first)["result"]["warnings"] == []
    assert (detail(second)["n_pins"], detail(second)["side"]) == (spec.n_pins, "outside")

    # pick the side by a point in the photo, on either side of the chain
    px = cam.project(pins)
    k = len(pins) // 4                     # in the first bend
    a, b = px[k], px[k + 1]
    normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    sides = []
    for sgn in (1, -1):
        point = ((a + b) / 2 + sgn * 60 * normal).tolist()
        assert options(second, {"side": point})[0] == 200
        assert app.wait_idle(60)
        d = detail(second)
        assert d["side"] == "picked" and d["view"]["side_px"] == point
        sides.append(d["result"]["object_side"])
        # the drawn contact edge is on the picked side of the chain
        edge = np.array(d["view"]["curve"])
        near = edge[np.argmin(np.linalg.norm(edge - (a + b) / 2, axis=1))]
        assert np.dot(near - (a + b) / 2, sgn * normal) > 0
    assert sorted(sides) == ["inside", "outside"]
    assert state()["defaults"]["side"] == "outside"            # a picked point is only this photo's
    doc = json.loads(app.photos[second].result.outputs["json"].read_text(encoding="utf-8"))
    assert doc["object_side_picked_px"] == [round(v, 1) for v in point]

    assert options(second, {"side": "inside"})[0] == 200
    assert app.wait_idle(60)
    assert detail(second)["side"] == "inside"
    assert detail(second)["view"]["side_px"] is None

    # bad values are refused and change nothing
    for bad in ({"n_pins": 2}, {"n_pins": 12.5}, {"n_pins": "x"}, {"n_pins": 1e400}, {"side": "left"},
                {"side": [1]}, {"side": ["a", 2]}, {"side": [1e6, 5]}, [1, 2]):
        assert options(second, bad)[0] == 400, bad
    assert call(server, "/api/upload?name=c.jpg&pins=2", data, app.token)[0] == 400
    assert call(server, "/api/upload?name=c.jpg&side=left", data, app.token)[0] == 400
    assert options("999", {"n_pins": 5})[0] == 404
    assert (app.photos[second].n_pins, app.photos[second].side) == (spec.n_pins, "inside")
    assert len(app.photos) == 2


def test_settings_keep_the_pin_count_when_the_page_leaves_it_out(served):
    app, server = served
    app.settings["chain"]["n_pins"] = 21
    chain = {k: v for k, v in app.settings["chain"].items() if k != "n_pins"}
    assert call(server, "/api/settings", json.dumps({"chain": chain}).encode(), app.token)[0] == 200
    assert app.settings["chain"]["n_pins"] == 21


def test_a_photo_changed_while_it_is_measured_is_not_called_done_early(served, dot_spec, tmp_path, monkeypatch):
    """Options changed while a photo is being measured queue it again. What was
    being measured is then already out of date: the photo stays queued until
    the last change has been measured, so whoever waits for "done" (the phone
    page, wait_idle) gets that result and not the one before it."""
    app, server = served
    write_synthetic_photo(tmp_path / "p.jpg", s_curve_pins(dot_spec), dot_spec, (2000, 1500))
    pid = json.loads(call(server, "/api/upload?name=p.jpg", (tmp_path / "p.jpg").read_bytes(), app.token)[1])["id"]
    assert app.wait_idle(60)

    measuring, go_on = threading.Event(), threading.Event()
    real = server_module.process_photo

    def held_up(*args, **kwargs):
        measuring.set()
        go_on.wait(20)
        return real(*args, **kwargs)

    seen = []
    changed = app._changed
    monkeypatch.setattr(app, "_changed", lambda: (seen.append((app.photos[pid].status, app._queue.qsize())), changed()))
    monkeypatch.setattr(server_module, "process_photo", held_up)
    app.set_options(pid, {"n_pins": 11})
    assert measuring.wait(20)                                  # being measured as an 11-pin chain...
    monkeypatch.setattr(server_module, "process_photo", real)
    app.set_options(pid, {"n_pins": dot_spec.n_pins})          # ...when it is changed back
    go_on.set()
    assert app.wait_idle(60)
    assert ("done", 1) not in seen                             # never done with a change still waiting
    result = json.loads(call(server, f"/api/photo/{pid}", token=app.token)[1])["result"]
    assert result["pins"] == dot_spec.n_pins and result["warnings"] == []
