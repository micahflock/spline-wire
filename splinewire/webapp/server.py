"""Local web app: the desktop UI in a browser, plus photo upload from a phone.

One small HTTP server (standard library only) on the computer:

- http://localhost:PORT/        the desktop page (this computer only)
- http://LAN-IP:PORT/phone      the phone page, opened by scanning a QR code

Uploads arrive as the raw request body, so the photo's original bytes (and
whatever metadata the phone kept) are saved as-is. A worker thread measures
each photo in turn.

Security: every API call needs the app's token, sent in an X-Token header.
A custom header can't be sent cross-site without a CORS preflight, which is
never granted, so other websites can't drive the app (CSRF). Images and
downloads, which the browser fetches without custom headers, take the token
as ?t= instead. The Host header must name this computer, which defeats DNS
rebinding. Settings, file downloads, installing the Fusion add-in, opening
folders, firewall changes and quitting are further limited to this computer.

Phones connect over the local network, which Windows Firewall blocks for a
new program until it is allowed; see firewall.py.
"""
from __future__ import annotations

import hmac
import io
import json
import mimetypes
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import numpy as np
from PIL import Image

from splinewire import __version__, version_string
from splinewire.chain import spec_from_dict, ChainSpec, default_chain_path, load_chain_spec
from splinewire.contact import spline_samples
from splinewire.fusion_addin import install_addin
from splinewire.output import crop_to_chain, points_tsv
from splinewire.process import PHOTO_SUFFIXES, PhotoResult, process_photo
from splinewire.settings import load_settings, save_settings
from splinewire.webapp.firewall import Firewall

DEFAULT_PORT = 8765
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
STATIC = Path(__file__).resolve().parent / "static"
CHAIN_KEYS = ("pitch_mm", "half_width_mm", "fiducial", "fiducial_mm", "ring_inner_mm", "n_pins")
# The chain every earlier version shipped (rings); saved unchanged, it follows the new default.
OLD_DEFAULT_CHAIN = {"ring_outer_mm": 5.0, "ring_inner_mm": 2.0}
FULL_FRAME_DIAGONAL_MM = 43.2666
DOWNLOADS = {"dxf": "dxf", "csv": "csv", "svg": "svg", "json": "json", "fusion_csv": "fusion-cm.csv"}


def _migrate_chain(saved: dict) -> dict:
    """Saved chain settings from before fiducials had a type (all rings).
    Left at the old default ring, they follow today's default instead; a
    ring someone set up on purpose stays a ring."""
    saved = dict(saved)
    if "ring_outer_mm" in saved and "fiducial_mm" not in saved:
        outer = saved.pop("ring_outer_mm")
        if {"ring_outer_mm": float(outer), "ring_inner_mm": float(saved.get("ring_inner_mm", 0))} \
                == OLD_DEFAULT_CHAIN:
            saved.pop("ring_inner_mm", None)
        else:
            saved.update(fiducial="ring", fiducial_mm=float(outer))
    return saved


@dataclass
class Photo:
    id: str
    name: str
    path: Path
    source: str                 # "phone" or "computer"
    received: float
    status: str = "queued"      # queued, processing, done, error
    error: str | None = None
    result: PhotoResult | None = None
    preview_jpeg: bytes | None = None
    thumb_jpeg: bytes | None = None


class AppState:
    """Photos, settings and the processing queue. Thread-safe."""

    def __init__(self, workspace: Path, settings: dict | None = None,
                 settings_file: Path | None = None, persist: bool = True) -> None:
        self.workspace = Path(workspace)
        self.settings_file = settings_file
        self.persist = persist
        self.settings = self._with_defaults(
            settings if settings is not None else load_settings(settings_file))
        self.token: str = self.settings["token"]
        self.lock = threading.RLock()
        self.photos: dict[str, Photo] = {}
        self.version = 0
        self._next_id = 1
        self._queue: queue.Queue[str] = queue.Queue()
        self._save()
        threading.Thread(target=self._work, daemon=True, name="splinewire-worker").start()

    # -- settings --------------------------------------------------------

    @staticmethod
    def _with_defaults(s: dict) -> dict:
        s = dict(s)
        s.setdefault("token", secrets.token_urlsafe(18))
        try:
            base = load_chain_spec(default_chain_path())
            chain = {k: getattr(base, k) for k in CHAIN_KEYS}
        except (OSError, KeyError, ValueError):
            chain = {"pitch_mm": 10.0, "half_width_mm": 4.0, "fiducial": "dot", "fiducial_mm": 5.0,
                     "ring_inner_mm": 0.0, "n_pins": 13}
        s["chain"] = {**chain, **_migrate_chain(s.get("chain", {}))}
        s.setdefault("side", "inside")
        s.setdefault("focal_override_35mm", None)
        s.setdefault("phone_focal_35mm", None)
        s.setdefault("truth", None)
        return s

    def _save(self) -> None:
        if self.persist:
            save_settings(self.settings, self.settings_file)

    def spec(self) -> ChainSpec:
        return spec_from_dict(self.settings["chain"])

    def update_settings(self, changes: dict) -> None:
        """Validate and apply settings from the page, then re-measure every photo."""
        with self.lock:
            new = json.loads(json.dumps(self.settings))
            if "chain" in changes:
                c = changes["chain"]
                new["chain"] = {k: float(c.get(k) or 0.0) for k in CHAIN_KEYS if k != "fiducial"}
                new["chain"]["n_pins"] = int(new["chain"]["n_pins"])
                new["chain"]["fiducial"] = str(c.get("fiducial", "dot"))
            if "side" in changes:
                if changes["side"] not in ("inside", "outside"):
                    raise ValueError("side must be inside or outside")
                new["side"] = changes["side"]
            for key in ("focal_override_35mm", "phone_focal_35mm"):
                if key in changes:
                    value = changes[key]
                    value = None if value in (None, "") else float(value)
                    if value is not None and not 5 <= value <= 300:
                        raise ValueError("focal lengths are 35 mm-equivalent, between 5 and 300 mm")
                    new[key] = value
            old, self.settings = self.settings, new
            try:
                self.spec()            # raises ValueError with a readable message if invalid
            except (ValueError, KeyError):
                self.settings = old
                raise
            self._save()
        self.reprocess_all()

    def set_truth(self, name: str, data: bytes) -> None:
        doc = json.loads(data.decode("utf-8"))
        if len(doc.get("pin_points", [])) < 3:
            raise ValueError("not a Spline Wire truth file (needs pin_points)")
        folder = self.workspace / "truth"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / _safe_name(name or "truth.json")
        path.write_bytes(data)
        with self.lock:
            self.settings["truth"] = str(path)
            self._save()
        self.reprocess_all()

    def clear_truth(self) -> None:
        with self.lock:
            self.settings["truth"] = None
            self._save()
        self.reprocess_all()

    # -- photos ------------------------------------------------------------

    def add_photo(self, name: str, data: bytes, source: str) -> Photo:
        name = _safe_name(name or "photo.jpg")
        if Path(name).suffix.lower() not in PHOTO_SUFFIXES:
            name += ".jpg"
        folder = self.workspace / "photos"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        n = 1
        while path.exists():
            path = path.with_name(f"{path.stem}-{n}{path.suffix}")
            n += 1
        path.write_bytes(data)
        with self.lock:
            photo = Photo(id=str(self._next_id), name=name, path=path, source=source, received=time.time())
            self._next_id += 1
            self.photos[photo.id] = photo
            self._changed()
        self._queue.put(photo.id)
        return photo

    def delete_photo(self, photo_id: str) -> None:
        with self.lock:
            if self.photos.pop(photo_id, None) is not None:
                self._changed()

    def reprocess_all(self) -> None:
        with self.lock:
            ids = list(self.photos)
            for pid in ids:
                self.photos[pid].status = "queued"
            self._changed()
        for pid in ids:
            self._queue.put(pid)

    def wait_idle(self, timeout: float = 120.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if all(p.status in ("done", "error") for p in self.photos.values()) and self._queue.empty():
                    return True
            time.sleep(0.05)
        return False

    def _changed(self) -> None:
        self.version += 1

    def _work(self) -> None:
        while True:
            pid = self._queue.get()
            with self.lock:
                photo = self.photos.get(pid)
                if photo is None:
                    continue
                photo.status = "processing"
                self._changed()
                settings = json.loads(json.dumps(self.settings))
            try:
                truth = settings["truth"]
                result = process_photo(
                    photo.path, self.spec(), self.workspace / "results",
                    focal_35mm=settings["focal_override_35mm"], side=settings["side"],
                    truth_path=Path(truth) if truth and Path(truth).is_file() else None,
                    default_focal_35mm=settings["phone_focal_35mm"],
                )
                preview = _jpeg(crop_to_chain(result.preview, result.measurement)[:, :, ::-1], 1600)
                thumb = _jpeg(crop_to_chain(result.preview, result.measurement)[:, :, ::-1], 240)
                with self.lock:
                    photo.result, photo.error, photo.status = result, None, "done"
                    photo.preview_jpeg, photo.thumb_jpeg = preview, thumb
                    self._changed()
            except Exception as err:   # every failure is reported on its photo
                # ValueErrors are the pipeline's own messages for people; show
                # anything else with its type so it can be reported.
                detail = str(err) if isinstance(err, ValueError) else \
                    "".join(traceback.format_exception_only(type(err), err)).strip()
                with self.lock:
                    photo.result, photo.error, photo.status = None, detail, "error"
                    photo.preview_jpeg = photo.thumb_jpeg = None
                    self._changed()

    # -- views for the pages -------------------------------------------------

    def state(self, local: bool, lan_urls: list[str]) -> dict:
        with self.lock:
            return {
                "app": "splinewire",
                "version": version_string(),
                "state_version": self.version,
                "photos": [self._summary(p) for p in reversed(list(self.photos.values()))],
                "settings": self._public_settings() if local else None,
                "phone_urls": lan_urls if local else None,
                "workspace": str(self.workspace) if local else None,
            }

    def detail(self, photo_id: str) -> dict | None:
        with self.lock:
            photo = self.photos.get(photo_id)
            if photo is None:
                return None
            out = self._summary(photo)
            r = photo.result
            if r is not None:
                m = r.measurement
                out["curve"] = {
                    "points": _rounded(m.contacts_mm),
                    "pins": _rounded(m.pins_mm),
                    "spline": _rounded(spline_samples(m.contacts_mm)),
                }
                out["downloads"] = [k for k in DOWNLOADS if k in r.outputs]
            return out

    def points_text(self, photo_id: str) -> str | None:
        with self.lock:
            photo = self.photos.get(photo_id)
            return points_tsv(photo.result.measurement.contacts_mm) if photo and photo.result else None

    def _public_settings(self) -> dict:
        s = {k: v for k, v in self.settings.items() if k != "token"}
        s["truth_name"] = Path(s["truth"]).name if s.get("truth") else None
        return s

    def _summary(self, p: Photo) -> dict:
        out = {"id": p.id, "name": p.name, "source": p.source, "received": p.received,
               "status": p.status, "error": p.error}
        r = p.result
        if r is not None:
            m, rect = r.measurement, r.measurement.rectification
            diag = float(np.hypot(r.info["width"], r.info["height"]))
            out["result"] = {
                "pins": len(m.order.indices),
                "missing_pins": len(m.order.gaps),
                "rejected": len(m.order.rejected),
                "curve_points": len(m.contacts_mm),
                "tilt_deg": round(rect.tilt_deg, 1),
                "focal_35mm": round(rect.focal_px * FULL_FRAME_DIAGONAL_MM / diag, 1),
                "focal_source": r.focal_source,
                "link_residual_rms_mm": round(rect.residual_rms_mm, 3),
                "link_residual_max_mm": round(rect.residual_max_mm, 3),
                "object_side": m.object_side,
                "warnings": list(m.warnings),
                "truth": r.truth_comparison,
            }
            out["info"] = r.info
        return out


class _Handler(BaseHTTPRequestHandler):
    server: "SplineWireServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:   # keep the console readable
        pass

    # -- plumbing ------------------------------------------------------------

    @property
    def app(self) -> AppState:
        return self.server.app

    def _local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in {"localhost", "127.0.0.1", "::1"} | set(self.server.lan_ips)

    def _token_ok(self, query: dict) -> bool:
        given = self.headers.get("X-Token") or (query.get("t") or [""])[0]
        return hmac.compare_digest(given.encode(), self.app.token.encode())

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload too large ({length // 1_000_000} MB)")
        buf = io.BytesIO()
        while buf.tell() < length:
            chunk = self.rfile.read(min(1 << 20, length - buf.tell()))
            if not chunk:
                break
            buf.write(chunk)
        return buf.getvalue()

    def _page(self, name: str) -> None:
        html = (STATIC / name).read_text(encoding="utf-8").replace("{{VERSION}}", version_string())
        self._send(200, html.encode(), "text/html; charset=utf-8",
                   {"Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; "
                                               "style-src 'self' 'unsafe-inline'; "
                                               "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'"})

    # -- routes ----------------------------------------------------------------

    def do_GET(self) -> None:
        self._route("GET")

    def do_HEAD(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def _route(self, method: str) -> None:
        try:
            if not self._host_ok():
                return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
            if not self._local():
                self.server.phone_seen = time.time()   # proof the network path works
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            if method == "GET" and path == "/api/ping":
                return self._json({"app": "splinewire", "version": __version__})
            if method == "GET" and path == "/":
                if not self._local():
                    return self._send(302, b"", "text/plain", {"Location": "/phone"})
                return self._page("desktop.html")
            if method == "GET" and path == "/phone":
                return self._page("phone.html")
            if method == "GET" and path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if method == "GET" and path == "/api/local-token" and self._local():
                return self._json({"token": self.app.token})
            if not path.startswith("/api/") or not self._token_ok(query):
                return self._error(HTTPStatus.FORBIDDEN, "missing or wrong token")
            self._api(method, path, query)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as err:
            self._error(HTTPStatus.BAD_REQUEST, str(err))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc(limit=3))

    def _api(self, method: str, path: str, query: dict) -> None:
        app, local = self.app, self._local()
        parts = path.strip("/").split("/")[1:]          # after "api"

        if method == "GET" and parts == ["state"]:
            out = app.state(local, self.server.phone_urls())
            if local:
                out["network"] = {"phone_seen": self.server.phone_seen,
                                  "firewall": self.server.firewall.snapshot()}
            return self._json(out)
        if method == "POST" and parts == ["upload"]:
            name = unquote((query.get("name") or ["photo.jpg"])[0])
            data = self._body()
            if not data:
                raise ValueError("empty upload")
            photo = app.add_photo(name, data, "computer" if local else "phone")
            return self._json({"id": photo.id})

        if parts[:1] == ["photo"] and len(parts) >= 2:
            pid, rest = parts[1], parts[2:]
            if method == "GET" and not rest:
                detail = app.detail(pid)
                return self._json(detail) if detail else self._error(404, "no such photo")
            if method == "GET" and rest in (["preview.jpg"], ["thumb.jpg"]):
                with app.lock:
                    photo = app.photos.get(pid)
                    data = photo and (photo.preview_jpeg if rest[0] == "preview.jpg" else photo.thumb_jpeg)
                return self._send(200, data, "image/jpeg") if data else self._error(404, "no preview")
            if method == "GET" and rest == ["points.tsv"]:
                text = app.points_text(pid)
                return self._send(200, text.encode(), "text/plain; charset=utf-8") if text \
                    else self._error(404, "not measured")
            if not local:
                return self._error(HTTPStatus.FORBIDDEN, "only from this computer")
            if method == "GET" and len(rest) == 2 and rest[0] == "download" and rest[1] in DOWNLOADS:
                with app.lock:
                    photo = app.photos.get(pid)
                    file = photo.result.outputs.get(rest[1]) if photo and photo.result else None
                if not file or not Path(file).is_file():
                    return self._error(404, "no such file")
                ctype = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                return self._send(200, Path(file).read_bytes(), ctype,
                                  {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(file.name)}"})
            if method == "POST" and rest == ["delete"]:
                app.delete_photo(pid)
                return self._json({"ok": True})

        if not local:
            return self._error(HTTPStatus.FORBIDDEN, "only from this computer")
        if method == "GET" and parts == ["qr.svg"]:
            return self._send(200, self.server.qr_svg().encode(), "image/svg+xml")
        if method == "POST" and parts == ["settings"]:
            app.update_settings(json.loads(self._body() or b"{}"))
            return self._json({"ok": True})
        if method == "POST" and parts == ["truth"]:
            app.set_truth(unquote((query.get("name") or ["truth.json"])[0]), self._body())
            return self._json({"ok": True})
        if method == "POST" and parts == ["truth", "clear"]:
            app.clear_truth()
            return self._json({"ok": True})
        if method == "POST" and parts == ["reprocess"]:
            app.reprocess_all()
            return self._json({"ok": True})
        if method == "POST" and parts == ["firewall", "check"]:
            self.server.firewall.refresh()
            return self._json({"ok": True})
        if method == "POST" and parts == ["firewall", "allow"]:
            # Shows the Windows admin (UAC) prompt; returns once it's answered.
            if not self.server.firewall.request_allow():
                return self._error(HTTPStatus.CONFLICT, "Windows didn't run the firewall change "
                                   "(the admin prompt was declined, or this isn't Windows).")
            return self._json({"ok": True})
        if method == "POST" and parts == ["install-addin"]:
            try:
                return self._json({"path": str(install_addin())})
            except FileNotFoundError as err:
                return self._error(HTTPStatus.CONFLICT, str(err))
        if method == "POST" and parts == ["open-folder"]:
            which = json.loads(self._body() or b"{}").get("which", "results")
            folder = app.workspace / ("photos" if which == "photos" else "results")
            folder.mkdir(parents=True, exist_ok=True)
            _open_folder(folder)
            return self._json({"ok": True})
        if method == "POST" and parts == ["quit"]:
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        return self._error(404, "unknown endpoint")


class SplineWireServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, app: AppState, port: int = DEFAULT_PORT, host: str = "0.0.0.0") -> None:
        self.app = app
        self.firewall = Firewall()
        self.phone_seen: float | None = None   # last request from another device
        self._ips: list[str] = []
        self._ips_at = 0.0
        super().__init__((host, port), _Handler)

    @property
    def lan_ips(self) -> list[str]:
        # Re-read every few seconds: joining a phone's hotspot after starting
        # the app changes this computer's address.
        if time.monotonic() - self._ips_at > 3.0:
            self._ips, self._ips_at = lan_addresses(), time.monotonic()
        return self._ips

    @property
    def port(self) -> int:
        return self.server_address[1]

    def desktop_url(self) -> str:
        return f"http://localhost:{self.port}/#t={self.app.token}"

    def phone_urls(self) -> list[str]:
        return [f"http://{ip}:{self.port}/phone#t={self.app.token}" for ip in self.lan_ips]

    def qr_svg(self) -> str:
        import segno
        urls = self.phone_urls()
        if not urls:
            return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>'
        buf = io.BytesIO()
        segno.make(urls[0], error="m").save(buf, kind="svg", scale=6, border=2, xmldecl=False,
                                           dark="#111111", light="#ffffff")
        return buf.getvalue().decode()


def lan_addresses() -> list[str]:
    """This computer's IPv4 addresses a phone on the same network can reach,
    most likely first (the one with the default route)."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))          # no packet is sent; just picks a route
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    usable = []
    for ip in found:
        if ip not in usable and not ip.startswith(("127.", "169.254.", "0.")):
            usable.append(ip)
    private = [ip for ip in usable if ip.startswith(("192.168.", "10.", "172."))]
    return private + [ip for ip in usable if ip not in private]


def start_server(app: AppState, port: int = DEFAULT_PORT, tries: int = 10) -> SplineWireServer:
    """Bind to the first free port from `port` (0 = any free port)."""
    last: OSError | None = None
    for p in ([0] if port == 0 else range(port, port + tries)):
        try:
            server = SplineWireServer(app, p)
        except OSError as err:
            last = err
            continue
        server.thread = threading.Thread(target=server.serve_forever, daemon=True, name="splinewire-http")
        server.thread.start()
        server.firewall.refresh()
        return server
    raise OSError(f"no free port from {port} to {port + tries - 1}: {last}")


def _safe_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "photo"
    return name[:80]


def _jpeg(rgb: np.ndarray, max_side: int) -> bytes:
    img = Image.fromarray(np.ascontiguousarray(rgb))
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _rounded(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]


def _open_folder(folder: Path) -> None:
    try:
        if sys.platform == "win32":
            import os
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError:
        pass
