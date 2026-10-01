"""Headless check that a packaged build works: `SplineWire.exe --selftest LOG`.

Starts the web app on a spare port with a temporary workspace, uploads
synthetic chain photos over HTTP (JPEG with EXIF, HEIC, and one with its
metadata stripped), checks the measurements against truth, edits pins by
hand, fetches both pages and the images, checks the security rules, installs the Fusion
add-in into a temporary folder and reads the Windows Firewall settings. Exit code 0 means every check passed.
"""
from __future__ import annotations

import faulthandler
import json
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from splinewire import version_string
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.synthetic import s_curve_pins, write_synthetic_photo, write_truth

MAX_ERROR_MM = 0.05
TIMEOUT_S = 180   # a hang dumps every thread's stack to the log and exits


def run_selftest(log_path: Path | None = None) -> int:
    watchdog = None
    if log_path:
        watchdog = open(log_path, "w", encoding="utf-8")  # replaced by the report on success
        faulthandler.dump_traceback_later(TIMEOUT_S, exit=True, file=watchdog)
    try:
        return _run(log_path)
    finally:
        if watchdog:
            faulthandler.cancel_dump_traceback_later()
            watchdog.close()


def _run(log_path: Path | None) -> int:
    from splinewire.webapp.server import AppState, start_server

    log: list[str] = []
    failures = 0

    def check(name: str, fn) -> None:
        nonlocal failures
        _progress(name)
        try:
            log.append(f"PASS {name}: {fn()}")
        except Exception:
            failures += 1
            log.append(f"FAIL {name}:\n{traceback.format_exc()}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = load_chain_spec(default_chain_path())
        pins = s_curve_pins(spec)
        jpg, truth = tmp / "selftest.jpg", tmp / "truth.json"
        write_synthetic_photo(jpg, pins, spec, (2000, 1500))
        write_truth(truth, pins)
        heic, bare = tmp / "selftest.heic", tmp / "stripped.jpg"
        with Image.open(jpg) as im:
            im.save(heic, exif=im.getexif(), quality=95)
            im.save(bare, quality=95)                  # no EXIF, like some iOS uploads

        app = AppState(tmp / "workspace", settings={"phone_focal_35mm": 26}, persist=False)
        server = start_server(app, 0)
        client = _Client(f"http://127.0.0.1:{server.port}", app.token)
        try:
            check("app responds", lambda: client.get("/api/ping"))
            check("truth file", lambda: client.post("/api/truth?name=truth.json", truth.read_bytes()))

            def measure(path: Path, focal_source: str) -> str:
                pid = client.post_json(f"/api/upload?name={path.name}", path.read_bytes())["id"]
                if not app.wait_idle(90):
                    raise AssertionError("processing did not finish")
                d = client.get_json(f"/api/photo/{pid}")
                if d["status"] != "done":
                    raise AssertionError(f"{d['status']}: {d['error']}")
                r = d["result"]
                err = r["truth"]["max_error_mm"]
                if err > MAX_ERROR_MM:
                    raise AssertionError(f"max error {err:.4f} mm > {MAX_ERROR_MM} mm")
                if r["focal_source"] != focal_source:
                    raise AssertionError(f"focal from {r['focal_source']}, expected {focal_source}")
                for part in ("preview.jpg", "thumb.jpg", "points.tsv", "download/dxf", "image.jpg"):
                    client.get(f"/api/photo/{pid}/{part}")
                return f"max error {err:.4f} mm, focal from {r['focal_source']}"

            check("jpeg upload", lambda: measure(jpg, "exif"))
            check("heic upload", lambda: measure(heic, "exif"))
            check("upload without metadata", lambda: measure(bare, "default"))

            def pin_editing() -> str:
                client.post("/api/truth/clear", b"")          # a truth file wants every pin
                pid = client.post_json(f"/api/upload?name={jpg.name}", jpg.read_bytes())["id"]

                def measured() -> dict:
                    if not app.wait_idle(90):
                        raise AssertionError("processing did not finish")
                    d = client.get_json(f"/api/photo/{pid}")
                    if d["status"] != "done":
                        raise AssertionError(f"{d['status']}: {d['error']}")
                    return d

                def edit(add: list, remove: list) -> dict:
                    client.post(f"/api/photo/{pid}/edits", json.dumps({"add": add, "remove": remove}).encode())
                    return measured()

                first = measured()
                end = first["view"]["pins"][-1]
                removed = edit([], [[end["x"], end["y"]]])
                if removed["result"]["pins"] != first["result"]["pins"] - 1 or len(removed["view"]["removed"]) != 1:
                    raise AssertionError("removing the end pin did not shorten the chain")
                back = edit([[end["x"] + 3, end["y"] - 2]], [])
                added = [p for p in back["view"]["pins"] if "edit" in p]
                if back["result"]["pins"] != first["result"]["pins"] or [p["snapped"] for p in added] != [True]:
                    raise AssertionError("a click near the removed pin did not snap back onto it")
                if client.get(f"/api/photo/{pid}/image.jpg")[:2] != b"\xff\xd8":
                    raise AssertionError("no photo for the editor")
                client.post("/api/truth?name=truth.json", truth.read_bytes())
                if not app.wait_idle(90):
                    raise AssertionError("processing did not finish")
                return f"removed and re-added pin {first['result']['pins']}, snapped"

            check("pin editing", pin_editing)

            def pages() -> str:
                for page in ("/", "/phone"):
                    html = client.get(page, token=False)
                    if b"Spline Wire" not in html or b"{{VERSION}}" in html:
                        raise AssertionError(f"{page} did not render")
                client.get("/api/qr.svg")
                return "desktop, phone, QR"

            check("pages", pages)

            def security() -> str:
                refused = [
                    client.status("/api/state", token=False),
                    client.status("/api/state", host="attacker.example"),
                ]
                if refused != [403, 421]:
                    raise AssertionError(f"expected [403, 421], got {refused}")
                return "token and Host checks refuse"

            check("security", security)

            def fusion_addin() -> str:
                from splinewire.fusion_addin import install_addin
                fusion_root = tmp / "Autodesk" / "Autodesk Fusion 360"
                fusion_root.mkdir(parents=True)
                dest = install_addin(fusion_root / "API" / "AddIns")
                files = sorted(p.name for p in dest.iterdir())
                if not {"SplineWire.py", "SplineWire.manifest", "curvedata.py"} <= set(files):
                    raise AssertionError(f"add-in files missing: {files}")
                return ", ".join(files)

            check("fusion add-in install", fusion_addin)

            def firewall() -> str:
                from splinewire.webapp.firewall import status
                st = status()
                if st.get("error"):
                    raise AssertionError(st["error"])
                if client.get_json("/api/state").get("network") is None:
                    raise AssertionError("no network status in /api/state")
                return json.dumps(st)

            check("firewall status", firewall)
        finally:
            server.shutdown()
            server.server_close()

    report = "\n".join([f"Spline Wire {version_string()}"] + log
                       + [f"{'OK' if not failures else 'FAILED'}: {failures} failure(s)"])
    if log_path:
        Path(log_path).write_text(report + "\n", encoding="utf-8")
    if sys.stdout:
        print(report)
    return 1 if failures else 0


class _Client:
    def __init__(self, base: str, token: str) -> None:
        self.base, self.token = base, token

    def _request(self, path: str, data: bytes | None, token: bool, host: str | None):
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        if token:
            req.add_header("X-Token", self.token)
        if host:
            req.add_header("Host", host)
        return urllib.request.urlopen(req, timeout=30)

    def get(self, path: str, token: bool = True) -> bytes:
        with self._request(path, None, token, None) as resp:
            return resp.read()

    def get_json(self, path: str):
        return json.loads(self.get(path))

    def post(self, path: str, data: bytes) -> bytes:
        with self._request(path, data, True, None) as resp:
            return resp.read()

    def post_json(self, path: str, data: bytes):
        return json.loads(self.post(path, data))

    def status(self, path: str, token: bool = True, host: str | None = None) -> int:
        try:
            with self._request(path, None, token, host) as resp:
                return resp.status
        except urllib.error.HTTPError as err:
            return err.code


def _progress(step: str) -> None:
    if sys.stdout:
        print(f"selftest: {step}", flush=True)
