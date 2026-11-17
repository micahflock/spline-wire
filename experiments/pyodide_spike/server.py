"""Static server for the spike, rooted at the repo, with CDN-like caching.

Everything is cacheable for a year (as a versioned CDN path would be), so a
second visit is served from the browser's HTTP cache. COOP/COEP headers make
the page cross-origin isolated, which lets it call
performance.measureUserAgentSpecificMemory().

    python experiments/pyodide_spike/server.py [--port 8770]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Bytes of file bodies sent, by path, since the last /__reset (for run.mjs).
SERVED: dict[str, int] = {}
_lock = threading.Lock()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript",
                      ".whl": "application/zip", ".zip": "application/zip"}

    def do_GET(self):
        if self.path == "/__stats":
            return self._json(SERVED)
        if self.path == "/__reset":
            with _lock:
                SERVED.clear()
            return self._json({})
        f = self.send_head()
        if f:
            try:
                n = 0
                while chunk := f.read(1 << 16):
                    self.wfile.write(chunk)
                    n += len(chunk)
            finally:
                f.close()
            with _lock:
                key = self.path.split("?")[0]
                SERVED[key] = SERVED.get(key, 0) + n

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        super().end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as srv:
        print(f"serving {ROOT} on http://127.0.0.1:{args.port}/experiments/pyodide_spike/", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
