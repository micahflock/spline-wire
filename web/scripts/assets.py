"""Put everything the core worker loads under web/public/, served from the site's own origin.

    npm run assets          # from web/, after npm ci; runs: uv run python scripts/assets.py

- Pyodide core (pyodide.mjs, pyodide.asm.*, python_stdlib.zip, pyodide-lock.json)
  from the npm `pyodide` package in node_modules, so its version is the one
  package-lock.json pins -> public/pyodide/
- The Pyodide packages the pipeline needs and ezdxf's dependencies, with
  their own dependencies, from cdn.jsdelivr.net/pyodide, each checked against
  the lock file's sha256 (also checked again by Pyodide when it loads them)
  -> public/pyodide/
- ezdxf's pure-Python wheel, the version and sha256 uv.lock pins (the version
  the native tests run) -> public/wheels/
- splinewire, built as a wheel -> public/wheels/<sha256 prefix>/, so a new
  build gets a new URL even when the version number has not changed
- data/chain*.yaml as one JSON file -> public/chains.json
- public/core-manifest.json: what the worker reads to find all of the above.

First it checks that web/package.json's version (what the pages show) is
splinewire's __version__ (what the core reports): one product, one number.

Files already in place with the right sha256 are not downloaded again. Files
in public/pyodide/ and public/wheels/ that are no longer wanted are deleted:
the script owns those folders.

The site must not depend on a CDN at run time; jsdelivr is only where the
package files come from at build time (GitHub releases are not reachable from
every build machine).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from dataclasses import asdict
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
ROOT = WEB.parent
PUBLIC = WEB / "public"
PYODIDE_NPM = WEB / "node_modules" / "pyodide"
CORE = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
# What the pipeline imports, loaded before the first measurement.
BOOT = ["numpy", "scipy", "opencv-python", "pillow", "pyyaml"]
# ezdxf's dependencies (numpy aside), loaded only for the DXF output.
EXTRAS = ["pyparsing", "typing-extensions", "fonttools"]
# HEIC/HEIF decoding (the iPhone's own format), loaded only when a photo is one:
# 0.7 s and about 1 MB, which JPEG photos should not pay for.
HEIF = ["pillow-heif"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, digest: str) -> None:
    if dest.exists() and sha256(dest) == digest:
        return
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise SystemExit(f"sha256 mismatch for {url}")
    dest.write_bytes(data)
    print(f"  fetched {dest.name} ({len(data) / 2**20:.1f} MiB)")


def closure(names: list[str], packages: dict) -> list[str]:
    seen: list[str] = []

    def walk(n: str) -> None:
        if n not in seen:
            seen.append(n)
            for d in packages[n].get("depends", []):
                walk(d)

    for n in names:
        walk(n)
    return seen


def locked_wheel(name: str) -> tuple[str, str, str]:
    """(version, url, sha256) of a package's pure-Python wheel in uv.lock."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    pkg = next(p for p in lock["package"] if p["name"] == name)
    whl = next(w for w in pkg["wheels"] if w["url"].endswith("-py3-none-any.whl"))
    return pkg["version"], whl["url"], whl["hash"].removeprefix("sha256:")


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
    """The pages and the core must say the same version: bump both together."""
    site = json.loads((WEB / "package.json").read_text())["version"]
    init = (ROOT / "splinewire" / "__init__.py").read_text()
    core = re.search(r'^__version__ = "([^"]+)"$', init, re.M)
    if not core or core.group(1) != site:
        raise SystemExit(f"web/package.json has version {site}, splinewire/__init__.py "
                         f"{core.group(1) if core else '(none found)'}: make them the same")
    print(f"version {site}, the same in web/package.json and splinewire")


def main() -> None:
    check_version()
    if not PYODIDE_NPM.exists():
        raise SystemExit("node_modules/pyodide is missing: run npm ci in web/ first")
    version = json.loads((PYODIDE_NPM / "package.json").read_text())["version"]
    pyo = PUBLIC / "pyodide"
    wheels = PUBLIC / "wheels"
    pyo.mkdir(parents=True, exist_ok=True)
    wheels.mkdir(parents=True, exist_ok=True)

    print(f"Pyodide {version} core from node_modules")
    for name in CORE:
        src, dest = PYODIDE_NPM / name, pyo / name
        if not dest.exists() or sha256(dest) != sha256(src):
            shutil.copyfile(src, dest)
    lock = json.loads((pyo / "pyodide-lock.json").read_text())
    packages = lock["packages"]
    boot, extras, heif = closure(BOOT, packages), closure(EXTRAS, packages), closure(HEIF, packages)
    extras = [n for n in extras if n not in boot]
    heif = [n for n in heif if n not in boot and n not in extras]
    cdn = f"https://cdn.jsdelivr.net/pyodide/v{version}/full/"
    print("packages from jsdelivr, sha256 checked against the lock")
    for n in boot + extras + heif:
        p = packages[n]
        fetch(cdn + p["file_name"], pyo / p["file_name"], p["sha256"])

    ezdxf_version, ezdxf_url, ezdxf_sha = locked_wheel("ezdxf")
    ezdxf_file = ezdxf_url.rsplit("/", 1)[1]
    print(f"ezdxf {ezdxf_version} as uv.lock pins it")
    fetch(ezdxf_url, wheels / ezdxf_file, ezdxf_sha)

    print("splinewire wheel")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "build", "--wheel", "--quiet", "--out-dir", tmp], cwd=ROOT, check=True)
        built = next(Path(tmp).glob("spline_wire-*.whl"))
        tag = sha256(built)[:12]
        (wheels / tag).mkdir(exist_ok=True)
        sw = wheels / tag / built.name
        shutil.copyfile(built, sw)

    # The script owns these folders: drop what an earlier run left that is not wanted now.
    keep = {pyo / n for n in CORE} | {pyo / packages[n]["file_name"] for n in boot + extras + heif}
    keep |= {wheels / ezdxf_file, sw, sw.parent}
    for folder in (pyo, wheels):
        for path in sorted(folder.iterdir(), reverse=True):
            if path not in keep:
                shutil.rmtree(path) if path.is_dir() else path.unlink()

    (PUBLIC / "chains.json").write_text(json.dumps(chains(), indent=2) + "\n")
    addin_zip(PUBLIC / ADDIN_ZIP)
    size = {n: (pyo / packages[n]["file_name"]).stat().st_size for n in boot + extras + heif}
    manifest = {
        "pyodide": {"version": version, "python": lock["info"]["python"], "index": "pyodide/"},
        # Package names for pyodide.loadPackage, which finds their files (and
        # dependencies) in the lock. Sizes are for the load progress.
        "boot": {"packages": boot, "bytes": sum(size[n] for n in boot)},
        "extras": {"packages": extras, "bytes": sum(size[n] for n in extras)},
        "heif": {"packages": heif, "bytes": sum(size[n] for n in heif)},
        "wheels": {
            "splinewire": sw.relative_to(PUBLIC).as_posix(),
            "ezdxf": (wheels / ezdxf_file).relative_to(PUBLIC).as_posix(),
        },
        "chains": "chains.json",
    }
    (PUBLIC / "core-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(p.stat().st_size for p in PUBLIC.rglob("*") if p.is_file())
    print(f"public/: {total / 2**20:.1f} MiB; manifest in public/core-manifest.json")


if __name__ == "__main__":
    main()
