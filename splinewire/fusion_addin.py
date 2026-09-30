"""Install the Spline Wire add-in (fusion/SplineWire) into Autodesk Fusion."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ADDIN_NAME = "SplineWire"


def addin_source() -> Path:
    """fusion/SplineWire in the repo, or its bundled copy in a packaged app."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / "fusion" / ADDIN_NAME


def fusion_addins_dir() -> Path:
    """Fusion's per-user add-in folder."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return base / "Autodesk" / "Autodesk Fusion 360" / "API" / "AddIns"


def install_addin(addins_dir: Path | None = None) -> Path:
    """Copy the add-in into Fusion's add-in folder (replacing an older copy).

    Raises FileNotFoundError if Fusion doesn't seem to be installed.
    """
    addins_dir = Path(addins_dir) if addins_dir else fusion_addins_dir()
    fusion_root = addins_dir.parent.parent            # .../Autodesk Fusion 360
    if not fusion_root.is_dir():
        raise FileNotFoundError(
            f"Autodesk Fusion doesn't seem to be installed for this user ({fusion_root} not found). "
            "Run Fusion once, then try again."
        )
    dest = addins_dir / ADDIN_NAME
    shutil.copytree(addin_source(), dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest
