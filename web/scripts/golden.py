"""Golden files for the TypeScript pipeline (web/src/pipeline/): Python's
detections and measurements on the cv_benchmark scenes (the dot chain,
data/chain.yaml) and on rendered photos of the snap-fit chain
(data/chain-snap.yaml: thin rims, round contact), so the port is checked
against the reference implementation (docs/web-build.md, D3).

For every scene: the image size, the true focal length, every fiducial
detect_fiducials found, and what pipeline.measure made of them in a few
variants (focal known or estimated, the side by name or by a photo point,
the sharpest bend 125 or 80 degrees, edge or round contact), or the
message of the ValueError it raised. Also order_chain without sizes on
the same centres.

    uv run python web/scripts/golden.py            # all 167 scenes (renders cached by cv_benchmark)
    uv run python web/scripts/golden.py --quick    # 30 scenes

The snap-fit photos are rendered once into out/golden-snap/.

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

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import chain_presets, load_chain_spec, pins_from_turns  # noqa: E402
from splinewire.detect import detect_fiducials  # noqa: E402
from splinewire.order import order_chain  # noqa: E402
from splinewire.live import analyse  # noqa: E402
from splinewire.pipeline import measure  # noqa: E402
from splinewire.synthetic import circle_wrap_pins, render_snap_photo, s_curve_pins  # noqa: E402

OUT = REPO / "web" / "src" / "pipeline" / "golden" / "scenes.json.gz"
SNAP_DIR = REPO / "out" / "golden-snap"
SNAP_SIZE = (2000, 1500)


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
    """A cv_benchmark scene (dot chain)."""
    out_dir = bench.OUT / bench.SPEC.fiducial
    meta = json.loads((out_dir / f"{sid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(out_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    return record(sid, img, bench.SPEC, meta["focal_px"], variants(bench.SPEC, meta["focal_px"], img.shape[::-1]))


def snap_scenes() -> list[tuple]:
    """(id, shape, table gray, tilt, tilt direction, roll) for the snap-fit chain."""
    shapes = ["s-curve", "pipe", "cove", "wave"]
    out = []
    for k in range(24):
        out.append((f"snap-{k:02d}", shapes[k % 4], (250, 120, 40)[k % 3], (0.0, 20.0, 35.0, 40.0)[(k // 3) % 4],
                    37.0 * k, (-15.0, 10.0, 30.0)[(k // 2) % 3]))
    return out


def snap_one(scene: tuple) -> dict:
    """A rendered snap-fit chain photo (rendered once, cached in out/golden-snap/)."""
    sid, shape, table, tilt, direction, roll = scene
    spec = load_chain_spec(chain_presets()["snap"])
    focal = focal_px_from_35mm(26, SNAP_SIZE)
    path = SNAP_DIR / f"{sid}.png"
    if not path.exists():
        pins = {"s-curve": s_curve_pins(spec), "pipe": circle_wrap_pins(spec, 40.0),
                "cove": circle_wrap_pins(spec, 60.0, concave=True),
                "wave": pins_from_turns(spec.pitch_mm, np.radians(25.0 * np.sin(0.9 * np.arange(spec.n_pins - 2))))}[shape]
        cam = look_at_plane(focal, SNAP_SIZE, 300.0, tilt_deg=tilt, tilt_direction_deg=direction, roll_deg=roll,
                            target_mm=tuple(pins.mean(axis=0)))
        img = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(int(sid[-2:])), table_gray=table)
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        (SNAP_DIR / f"{sid}.json").write_text(json.dumps({"focal_px": focal, "pins_mm": pins.tolist(),
                                                           "spec": asdict(spec)}), encoding="utf-8")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    w, h = SNAP_SIZE
    runs = [
        dict(name="focal-inside", spec=spec, focal=focal, side="inside"),
        dict(name="estimated-outside", spec=spec, focal=None, side="outside"),
        dict(name="edge-point", spec=replace(spec, contact="edge"), focal=focal, side=[w / 2.0, h / 2.0]),
    ]
    return record(sid, img, spec, focal, runs)


def record(sid: str, img: np.ndarray, spec, focal: float, runs: list[dict]) -> dict:
    h, w = img.shape[:2]
    fids = detect_fiducials(img, spec)
    rec = {
        "id": sid, "size": [w, h], "focal_px": focal,
        "fiducials": [[*f.center_px, *f.outer_axes_px, f.contrast, f.polarity, f.surround] for f in fids],
        "runs": [],
    }
    # The overlay's analysis (live.analyse) on the same blobs, as a 640 px work frame sees them.
    k = 640.0 / max(w, h)
    pts = [[f.center_px[0] * k, f.center_px[1] * k] for f in fids]
    dia = [f.outer_axes_px[0] * k for f in fids]
    small = (int(round(w * k)), int(round(h * k)))
    rec["live"] = [{"points": pts, "diameters": dia, "size": list(small), "focal_px": fpx, "chain": asdict(spec),
                    "result": analyse(pts, dia, small, spec, fpx)} for fpx in (focal * k, None)]
    if len(fids) >= 3:
        o = order_chain(np.array([f.center_px for f in fids]), max_turn_deg=80.0)
        rec["plain_order"] = {"indices": [int(i) for i in o.indices], "links": [list(map(int, ln)) for ln in o.links],
                              "gaps": [int(g) for g in o.gaps]}
    for v in runs:
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
        recs += list(ex.map(snap_one, snap_scenes()[::4] if args.quick else snap_scenes()))
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
