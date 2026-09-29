"""Headless check that a packaged build works: `SplineWire.exe --selftest LOG`.

Renders a synthetic chain photo (JPEG and HEIC, with EXIF), processes it,
checks the error against truth, and opens the GUI window briefly. Exit
code 0 means every check passed.
"""
from __future__ import annotations

import faulthandler
import sys
import tempfile
import traceback
from pathlib import Path

from PIL import Image

from splinewire import __version__, version_string
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.process import process_photo
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
    log: list[str] = []
    failures = 0

    def check(name: str, fn) -> None:
        nonlocal failures
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

        def measure(photo: Path) -> str:
            res = process_photo(photo, spec, tmp / "out", truth_path=truth)
            err = res.truth_comparison["max_error_mm"]
            if err > MAX_ERROR_MM:
                raise AssertionError(f"max error {err:.4f} mm > {MAX_ERROR_MM} mm")
            if res.measurement.rectification.focal_estimated:
                raise AssertionError("EXIF focal length was not read")
            missing = [p for p in res.outputs.values() if not p.is_file()]
            if missing:
                raise AssertionError(f"outputs not written: {missing}")
            return f"max error {err:.4f} mm"

        def heic() -> str:
            photo = tmp / "selftest.heic"
            with Image.open(jpg) as im:
                im.save(photo, exif=im.getexif(), quality=95)
            return measure(photo)

        _progress("jpeg")
        check("jpeg photo", lambda: measure(jpg))
        _progress("heic")
        check("heic photo", heic)
        def fusion_addin() -> str:
            from splinewire.fusion_addin import install_addin
            fusion_root = tmp / "Autodesk" / "Autodesk Fusion 360"
            fusion_root.mkdir(parents=True)
            dest = install_addin(fusion_root / "API" / "AddIns")
            files = sorted(p.name for p in dest.iterdir())
            if not {"SplineWire.py", "SplineWire.manifest", "curvedata.py"} <= set(files):
                raise AssertionError(f"add-in files missing: {files}")
            return ", ".join(files)

        _progress("fusion add-in")
        check("fusion add-in install", fusion_addin)
        _progress("gui")
        check("gui window", lambda: _open_gui(jpg))
        _progress("gui layout")
        check("buttons visible at high display scaling", _check_layout)

    report = "\n".join([f"Spline Wire {version_string()}"] + log
                       + [f"{'OK' if not failures else 'FAILED'}: {failures} failure(s)"])
    if log_path:
        Path(log_path).write_text(report + "\n", encoding="utf-8")
    if sys.stdout:
        print(report)
    return 1 if failures else 0


def _progress(step: str) -> None:
    if sys.stdout:
        print(f"selftest: {step}", flush=True)


def _check_layout() -> str:
    """The main buttons must stay on screen on a small, high-DPI display.

    Emulates the most zoomed-in realistic setup for this screen: display
    scaling chosen so the desktop is 1280 x 720 in scaled pixels (e.g. a
    1920 x 1080 laptop at 150%). tk scaling is pixels per point, 1.33 = 100%.
    """
    import tkinter as tk

    from splinewire.gui import App

    try:
        root = tk.Tk()
    except tk.TclError as err:
        if sys.platform == "win32":
            raise
        return f"skipped (no display: {err})"
    try:
        zoom = min(root.winfo_screenwidth() / 1280, root.winfo_screenheight() / 720)
        root.tk.call("tk", "scaling", 1.333 * max(1.0, min(2.0, zoom)))
        app = App(root, settings={}, persist=False, interactive=False)
        root.update()
        if __version__ not in root.title():
            raise AssertionError(f"version missing from title: {root.title()!r}")
        hidden = [b.cget("text") for b in (app.process_all_btn, app.copy_btn, app.install_btn)
                  if not b.winfo_ismapped()]
        if hidden:
            raise AssertionError(f"buttons not visible: {hidden}")
        return f"{root.title()!r} at {root.winfo_width()}x{root.winfo_height()}, zoom {zoom:.2f}"
    finally:
        root.destroy()


def _open_gui(photo: Path) -> str:
    import tkinter as tk

    from splinewire.gui import App

    try:
        root = tk.Tk()
    except tk.TclError as err:
        if sys.platform == "win32":
            raise
        return f"skipped (no display: {err})"
    try:
        app = App(root, photos=[photo], settings={}, persist=False, interactive=False)
        root.update()
        app.process(all_items=True)
        app.worker.join(timeout=60)
        for _ in range(20):          # let the event poller deliver the result
            root.update()
            root.after(50)
        if app.errors:
            raise AssertionError("GUI callback errors:\n" + "\n".join(app.errors))
        item = next(iter(app.items.values()))
        if item.result is None:
            raise AssertionError(f"GUI processing did not finish: {item.status} {item.error}")
        for tab in range(3):
            app.tabs.select(tab)
            root.update()
        app.copy_points()
        copied = root.clipboard_get()
        if not copied.startswith("x_mm\ty_mm") or len(copied.splitlines()) < 3:
            raise AssertionError(f"Copy points put unexpected text on the clipboard: {copied[:80]!r}")
        if app.errors:
            raise AssertionError("GUI callback errors:\n" + "\n".join(app.errors))
        return item.summary
    finally:
        root.destroy()
