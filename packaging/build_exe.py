"""Build the single-file desktop app with PyInstaller.

    uv sync --group build
    uv run python packaging/build_exe.py

Produces dist/SplineWire.exe on Windows (dist/SplineWire elsewhere).
PyInstaller cannot cross-compile, so the Windows .exe is built on Windows,
normally by .github/workflows/windows-exe.yml.
"""
import os
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    PyInstaller.__main__.run([
        str(ROOT / "packaging" / "splinewire_app.py"),
        "--name", "SplineWire",
        "--onefile",
        "--console",            # the window says the app is running; closing it quits
        "--noconfirm",
        "--clean",
        "--add-data", f"{ROOT / 'data' / 'chain.yaml'}{os.pathsep}data",
        "--add-data", f"{ROOT / 'fusion' / 'SplineWire'}{os.pathsep}fusion/SplineWire",
        "--add-data", f"{ROOT / 'splinewire' / 'webapp' / 'static'}{os.pathsep}splinewire/webapp/static",
        "--exclude-module", "tkinter",
        "--collect-all", "pillow_heif",   # bundles libheif for iPhone HEIC photos
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
    ])


if __name__ == "__main__":
    main()
