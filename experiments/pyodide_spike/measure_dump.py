"""Measure one photo the way `splinewire measure` does and dump full-precision results.

Runs unchanged natively (the CLI's reference) and inside the Pyodide worker,
so the two can be compared pin by pin:

    uv run python experiments/pyodide_spike/measure_dump.py PHOTO [--truth T] [--json OUT]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def versions() -> dict:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0], "platform": sys.platform}
    for dist in ["numpy", "scipy", "opencv-python", "opencv-python-headless", "pillow",
                 "pyyaml", "ezdxf", "spline-wire"]:
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    import cv2
    out["cv2.__version__"] = cv2.__version__
    return out


def run(photo: str, chain: str, out_dir: str, truth: str | None = None,
        repeats: int = 1) -> dict:
    t0 = time.perf_counter()
    from splinewire.chain import load_chain_spec
    from splinewire.detect import detect_fiducials
    from splinewire.pipeline import load_photo
    from splinewire.process import process_photo
    t_import = time.perf_counter() - t0

    spec = load_chain_spec(Path(chain))
    runs = []
    for _ in range(repeats):
        t = {}
        a = time.perf_counter()
        image, focal = load_photo(Path(photo))
        t["load_photo_s"] = time.perf_counter() - a
        a = time.perf_counter()
        detected = detect_fiducials(image, spec)
        t["detect_s"] = time.perf_counter() - a
        a = time.perf_counter()
        # As the CLI calls it (cli.py), with the already loaded and detected photo.
        result = process_photo(Path(photo), spec, Path(out_dir), truth_path=Path(truth) if truth else None,
                               compare_truth=True if truth else None,
                               loaded=(image, focal), detected=detected)
        t["measure_and_write_s"] = time.perf_counter() - a
        t["total_s"] = t["load_photo_s"] + t["detect_s"] + t["measure_and_write_s"]
        runs.append(t)

    m = result.measurement
    r = m.rectification
    outputs = {k: Path(p).stat().st_size for k, p in result.outputs.items()}
    return {
        "versions": versions(),
        "photo": Path(photo).name,
        "image_shape": list(image.shape),
        "focal_px_exif": focal,
        "import_s": t_import,
        "runs": runs,
        "n_fiducials": len(m.fiducials),
        "fiducials_px": [list(map(float, f.center_px)) for f in m.fiducials],
        "pins_px": m.pins_px.tolist(),
        "pins_mm": m.pins_mm.tolist(),
        "contacts_mm": m.contacts_mm.tolist(),
        "tilt_deg": r.tilt_deg,
        "focal_px": r.focal_px,
        "residual_rms_mm": r.residual_rms_mm,
        "warnings": m.warnings,
        "notes": m.notes,
        "truth_comparison": result.truth_comparison,
        "output_bytes": outputs,
        "peak_rss_bytes": _peak_rss(),
    }


def _peak_rss() -> int | None:
    try:
        import resource
    except ImportError:          # under Pyodide
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--chain", default=str(Path(__file__).resolve().parents[2] / "data" / "chain.yaml"))
    ap.add_argument("--truth")
    ap.add_argument("--out", default="out/pyodide_spike/native-out")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--json")
    args = ap.parse_args()
    res = run(args.photo, args.chain, args.out, args.truth, args.repeats)
    text = json.dumps(res, indent=1)
    if args.json:
        Path(args.json).write_text(text)
    print(json.dumps({k: res[k] for k in ("versions", "import_s", "runs", "n_fiducials",
                                          "tilt_deg", "truth_comparison")}, indent=1))


if __name__ == "__main__":
    main()
