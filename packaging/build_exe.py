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
        str(ROOT / "packaging" / "splinewire_gui.py"),
        "--name", "SplineWire",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--add-data", f"{ROOT / 'data' / 'chain.yaml'}{os.pathsep}data",
        "--collect-all", "pillow_heif",   # bundles libheif for iPhone HEIC photos
        # Pillow loads its Tk bridge dynamically; without this, showing a
        # photo in the window fails with "invalid command name PyImagingPhoto".
        "--hidden-import", "PIL._tkinter_finder",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
    ])


if __name__ == "__main__":
    main()
