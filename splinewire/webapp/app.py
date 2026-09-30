"""SplineWire.exe entry point: start the local web app and open it in the browser.

    splinewire-app [--port N] [--no-browser] [--workspace DIR]
    splinewire-app --selftest LOGFILE
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import webbrowser
from pathlib import Path

from splinewire import version_string
from splinewire.settings import default_workspace
from splinewire.webapp.server import DEFAULT_PORT, AppState, start_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splinewire-app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--workspace", type=Path, help="folder for photos and results")
    parser.add_argument("--selftest", metavar="LOGFILE", nargs="?", const="", default=None,
                        help="check a packaged build and exit")
    args = parser.parse_args(argv)

    if args.selftest is not None:
        from splinewire.selftest import run_selftest
        return run_selftest(Path(args.selftest) if args.selftest else None)

    if _already_running(args.port):
        print(f"Spline Wire is already running; opening it: http://localhost:{args.port}/")
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}/")
        return 0

    app = AppState(args.workspace or default_workspace())
    server = start_server(app, args.port)
    print(f"Spline Wire {version_string()}")
    print(f"  On this computer:  http://localhost:{server.port}/")
    for url in server.phone_urls()[:1]:
        print(f"  From your phone:   scan the QR code on that page (same Wi-Fi), or open")
        print(f"                     {url}")
    print(f"  Photos and results: {app.workspace}")
    print("  Phone times out? Click \"Allow phone connections\" on the page (Windows Firewall).")
    print()
    print("Keep this window open while you use Spline Wire. Close it, or press Ctrl+C, to quit.")
    if not args.no_browser:
        webbrowser.open(server.desktop_url())
    try:
        while server.thread.is_alive():      # ends when the page's Quit button stops the server
            server.thread.join(0.5)          # short waits keep Ctrl+C responsive on Windows
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def _already_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=0.5) as resp:
            return json.load(resp).get("app") == "splinewire"
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
