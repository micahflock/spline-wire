"""Put the files the pages fetch under web/public/, served from the site's own origin.

    npm run assets          # from web/: runs uv run python scripts/assets.py

- data/chain*.yaml as one JSON file -> public/chains.json
- fusion/SplineWire as a zip -> public/SplineWire-fusion-addin.zip

First it checks that web/package.json's version (what the pages show) is
splinewire's __version__: one product, one number.

Until 2026-11-25 it also put Pyodide, its packages and the splinewire wheel
under public/ for the Python engine (D29 in docs/web-build.md removed it);
it deletes what an earlier run left of those.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
PUBLIC = WEB / "public"
# What the Python engine used to need (D29): removed from public/ if an earlier run left it.
RETIRED = ["pyodide", "wheels", "core-manifest.json"]


ADDIN_ZIP = "SplineWire-fusion-addin.zip"


def addin_zip(dest: Path) -> None:
    """fusion/SplineWire as a zip with a SplineWire/ folder inside, for the
    desktop page's download (decision O5). Fixed timestamps and order, so the
    same add-in gives the same bytes and a cached copy stays valid."""
    import zipfile

    src = ROOT / "fusion" / "SplineWire"
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(f"SplineWire/{p.relative_to(src).as_posix()}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())


def chains() -> dict:
    sys.path.insert(0, str(ROOT))
    from splinewire.chain import chain_presets, load_chain_spec
    return {name: asdict(load_chain_spec(path)) for name, path in chain_presets().items()}


def check_version() -> None:
    """The pages and the Python package must say the same version: bump both together."""
    site = json.loads((WEB / "package.json").read_text())["version"]
    init = (ROOT / "splinewire" / "__init__.py").read_text()
    core = re.search(r'^__version__ = "([^"]+)"$', init, re.M)
    if not core or core.group(1) != site:
        raise SystemExit(f"web/package.json has version {site}, splinewire/__init__.py "
                         f"{core.group(1) if core else '(none found)'}: make them the same")
    print(f"version {site}, the same in web/package.json and splinewire")


def main() -> None:
    check_version()
    PUBLIC.mkdir(exist_ok=True)
    for name in RETIRED:
        path = PUBLIC / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (PUBLIC / "chains.json").write_text(json.dumps(chains(), indent=2) + "\n")
    addin_zip(PUBLIC / ADDIN_ZIP)
    total = sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())
    print(f"public/: {total / 2**10:.0f} KiB (chains.json, {ADDIN_ZIP})")


if __name__ == "__main__":
    main()
