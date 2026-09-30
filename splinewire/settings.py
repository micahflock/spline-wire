"""Per-user settings and the working folder for photos and results."""
from __future__ import annotations

import json
import os
from pathlib import Path


def settings_path() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) / "SplineWire" if base else Path.home() / ".config" / "splinewire"
    return root / "settings.json"


def load_settings(path: Path | None = None) -> dict:
    try:
        return json.loads((path or settings_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict, path: Path | None = None) -> None:
    path = path or settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # settings are a convenience; never fail the app over them


def default_workspace() -> Path:
    """Where uploaded photos and results go: Documents/SplineWire."""
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "SplineWire"
