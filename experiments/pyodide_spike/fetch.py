"""Fetch everything the spike page serves from its own origin, into out/pyodide_spike/.

- Pyodide core (pyodide.js/.mjs, pyodide.asm.mjs, pyodide.asm.wasm,
  python_stdlib.zip, pyodide-lock.json) from the `pyodide` npm package.
- The Pyodide packages splinewire needs (and ezdxf's dependencies), with
  their dependencies, from cdn.jsdelivr.net/pyodide, checked against the
  lock file's sha256.
- ezdxf's pure-Python wheel from PyPI (micropip would fetch the same file).
- splinewire itself, built as a wheel with `uv build`.

Also records each file's size as served by jsdelivr with brotli
(Accept-Encoding: br, gzip), which is what a CDN would send, in
out/pyodide_spike/sizes.json.

    uv run python experiments/pyodide_spike/fetch.py [--version 314.0.8]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "pyodide_spike"
CORE = ["pyodide.js", "pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm",
        "python_stdlib.zip", "pyodide-lock.json"]
# What splinewire imports at run time, plus micropip and ezdxf's dependencies.
WANT = ["numpy", "scipy", "opencv-python", "pillow", "pyyaml", "micropip",
        "pyparsing", "typing-extensions", "fonttools"]


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def cdn_transfer_bytes(url: str) -> int:
    """Bytes on the wire for url with brotli/gzip allowed (body only)."""
    out = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-H", "Accept-Encoding: br, gzip",
         "-w", "%{size_download} %{http_code}", url],
        check=True, capture_output=True, text=True).stdout.split()
    if out[1] != "200":
        raise RuntimeError(f"{url}: HTTP {out[1]}")
    return int(out[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="314.0.8")
    ap.add_argument("--no-cdn-sizes", action="store_true")
    args = ap.parse_args()
    v = args.version
    cdn = f"https://cdn.jsdelivr.net/pyodide/v{v}/full/"
    pyo = OUT / "pyodide"
    wheels = OUT / "wheels"
    pyo.mkdir(parents=True, exist_ok=True)
    wheels.mkdir(parents=True, exist_ok=True)

    # Core from npm.
    meta = json.loads(get(f"https://registry.npmjs.org/pyodide/{v}"))
    tgz = get(meta["dist"]["tarball"])
    with tarfile.open(fileobj=io.BytesIO(tgz)) as tf:
        for name in CORE:
            (pyo / name).write_bytes(tf.extractfile(f"package/{name}").read())
    print(f"core from npm pyodide@{v} ({len(tgz)} bytes tarball)")

    lock = json.loads((pyo / "pyodide-lock.json").read_text())
    pkgs = lock["packages"]
    need: set[str] = set()

    def walk(n: str) -> None:
        if n not in need:
            need.add(n)
            for d in pkgs[n].get("depends", []):
                walk(d)

    for n in WANT:
        walk(n)
    files = {}
    for n in sorted(need):
        p = pkgs[n]
        dest = pyo / p["file_name"]
        if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != p["sha256"]:
            data = get(cdn + p["file_name"])
            if hashlib.sha256(data).hexdigest() != p["sha256"]:
                raise RuntimeError(f"sha256 mismatch for {p['file_name']}")
            dest.write_bytes(data)
        files[n] = p["file_name"]
        print(f"{n} {p['version']}: {dest.stat().st_size} bytes")

    # ezdxf from PyPI: the newest pure-Python wheel.
    info = json.loads(get("https://pypi.org/pypi/ezdxf/json"))
    any_whl = [u for u in info["urls"] if u["filename"].endswith("py3-none-any.whl")][0]
    (wheels / any_whl["filename"]).write_bytes(get(any_whl["url"]))
    print(f"ezdxf {info['info']['version']}: {any_whl['filename']}")

    # splinewire as a wheel.
    for old in wheels.glob("spline_wire-*.whl"):
        old.unlink()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(wheels)], cwd=ROOT, check=True)
    sw = sorted(wheels.glob("spline_wire-*.whl"))[-1]

    manifest = {
        "pyodide_version": v,
        "python": lock["info"]["python"],
        "packages": {n: {"version": pkgs[n]["version"], "file": files[n]} for n in sorted(need)},
        "ezdxf_wheel": any_whl["filename"],
        "splinewire_wheel": sw.name,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    sizes = {}
    rows = [("core", f, pyo / f, cdn + f) for f in CORE if f not in ("pyodide.mjs",)]
    rows += [(n, files[n], pyo / files[n], cdn + files[n]) for n in sorted(need)]
    rows += [("ezdxf", any_whl["filename"], wheels / any_whl["filename"], any_whl["url"]),
             ("splinewire", sw.name, sw, None)]
    for group, name, path, url in rows:
        raw = path.stat().st_size
        cdn_b = cdn_transfer_bytes(url) if url and not args.no_cdn_sizes else None
        sizes[name] = {"group": group, "bytes": raw, "cdn_transfer_bytes": cdn_b}
        print(f"{name:70s} {raw:>10d} cdn {cdn_b}")
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
