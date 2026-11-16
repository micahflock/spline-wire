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
    """The saved settings, or {} if there are none that can be read."""
    try:
        data = json.loads((path or settings_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(data: dict, path: Path | None = None) -> None:
    path = path or settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the file, then swapped in: closing the app's window can
        # stop it mid-write, and half a file would lose the chain's settings and
        # the token the phone's saved link uses.
        partial = path.with_name(path.name + ".tmp")
        partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(partial, path)
    except OSError:
        pass  # settings are a convenience; never fail the app over them


def default_workspace() -> Path:
    """Where uploaded photos and results go: Documents/SplineWire."""
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "SplineWire"
