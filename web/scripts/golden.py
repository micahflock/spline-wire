"""Golden files for the TypeScript pipeline (web/src/pipeline/): Python's
detections and measurements on the cv_benchmark scenes, so the port is
checked against the reference implementation (docs/web-build.md, D3).

For every scene: the image size, the true focal length, every fiducial
detect_fiducials found, and what pipeline.measure made of them in a few
variants (focal known or estimated, the side by name or by a photo point,
the sharpest bend 125 or 80 degrees, edge or round contact), or the
message of the ValueError it raised. Also order_chain without sizes on
the same centres.

    uv run python web/scripts/golden.py            # all 167 scenes (renders cached by cv_benchmark)
    uv run python web/scripts/golden.py --quick    # 30 scenes

Writes web/src/pipeline/golden/scenes.json.gz. Floats are written with
repr, so they reach JavaScript unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

import cv_benchmark as bench  # noqa: E402

from splinewire.detect import detect_fiducials  # noqa: E402
from splinewire.order import order_chain  # noqa: E402
from splinewire.pipeline import measure  # noqa: E402

OUT = REPO / "web" / "src" / "pipeline" / "golden" / "scenes.json.gz"


def variants(spec, focal: float, size: tuple[int, int]) -> list[dict]:
    w, h = size
    sharp = replace(spec, max_bend_deg=80.0)
    return [
        dict(name="focal-inside", spec=spec, focal=focal, side="inside"),
        dict(name="estimated-outside", spec=spec, focal=None, side="outside"),
        dict(name="bend80-point-round", spec=replace(sharp, contact="round"), focal=focal,
             side=[w / 2.0, h / 2.0]),
    ]


def one(sid: str) -> dict:
    out_dir = bench.OUT / bench.SPEC.fiducial
    meta = json.loads((out_dir / f"{sid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(out_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    fids = detect_fiducials(img, bench.SPEC)
    rec = {
        "id": sid, "size": [w, h], "focal_px": meta["focal_px"],
        "fiducials": [[*f.center_px, *f.outer_axes_px, f.contrast, f.polarity, f.surround] for f in fids],
        "runs": [],
    }
    if len(fids) >= 3:
        o = order_chain(np.array([f.center_px for f in fids]), max_turn_deg=80.0)
        rec["plain_order"] = {"indices": [int(i) for i in o.indices], "links": [list(map(int, ln)) for ln in o.links],
                              "gaps": [int(g) for g in o.gaps]}
    for v in variants(bench.SPEC, meta["focal_px"], (w, h)):
        side = v["side"] if isinstance(v["side"], str) else tuple(v["side"])
        run = {"name": v["name"], "spec": asdict(v["spec"]), "focal_px": v["focal"], "side": v["side"]}
        try:
            m = measure(img, v["spec"], v["focal"], side=side, detected=fids)
        except ValueError as err:
            run["error"] = str(err)
        else:
            r = m.rectification
            run["result"] = {
                "indices": [int(i) for i in m.order.indices],
                "links": [list(map(int, ln)) for ln in m.order.links],
                "gaps": [int(g) for g in m.order.gaps],
                "rejected": [int(i) for i in m.order.rejected],
                "pins_mm": r.pins_mm.tolist(), "focal_px": r.focal_px, "tilt_deg": r.tilt_deg,
                "residual_rms_mm": r.residual_rms_mm, "residual_max_mm": r.residual_max_mm,
                "depth_mm": r.depth_mm.tolist(),
                "object_side": m.object_side, "object_sign": m.object_sign,
                "contacts_mm": m.contacts_mm.tolist(), "warnings": m.warnings, "notes": m.notes,
            }
        rec["runs"].append(run)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="30 scenes, for a quick look")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    from splinewire.fiducials import chain_design
    scenes = bench.scene_list(80, 3)
    if args.quick:
        scenes = scenes[::6][:30]
    out_dir = bench.OUT / bench.SPEC.fiducial
    bench.render_suite(scenes, chain_design(bench.SPEC), out_dir, args.workers)
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, [sid for sid, *_ in scenes]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"chain": asdict(bench.SPEC), "scenes": recs}, separators=(",", ":"))
    with gzip.GzipFile(args.out, "wb", mtime=0) as f:
        f.write(data.encode("utf-8"))
    n_runs = sum(len(r["runs"]) for r in recs)
    n_err = sum("error" in run for r in recs for run in r["runs"])
    print(f"{len(recs)} scenes, {n_runs} runs ({n_err} ValueErrors), "
          f"{sum(len(r['fiducials']) for r in recs)} fiducials -> {args.out} ({args.out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
