"""How well does detection hold up across real-world photo conditions?

Renders realistic photos (splinewire/scene.py) of several chain shapes under
every preset environment plus randomly drawn ones, then runs detection and
the whole pipeline on each and reports, per environment:

  found     true pins detected (of 13), anywhere in the photo
  false     fiducial detections that are not a pin
  chain     photos where the pipeline returned every pin, in order, with no
            stray (washer, letter) taken for a pin
  err px    centre error of detected pins: median / worst
  err mm    worst pin error in mm after the pipeline (rigid fit to truth)

The fiducial is the chain's (data/chain.yaml), or --fiducial dot|ring.
Renders are cached in out/cv-benchmark/<fiducial>/ so detector changes
re-run in a couple of minutes. --baseline <git rev> also evaluates the
whole splinewire package as it was at that revision (with that revision's
own chain.yaml), for comparison: rings only before the dot existed.

    uv run python experiments/cv_benchmark.py                 # render (once) + evaluate
    uv run python experiments/cv_benchmark.py --random 60     # plus 60 random environments
    uv run python experiments/cv_benchmark.py --fiducial ring --baseline 752b3a9
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
# SPLINEWIRE_ROOT: evaluate another copy of the package (see --baseline).
ROOT = Path(os.environ.get("SPLINEWIRE_ROOT", str(REPO)))
sys.path.insert(0, str(ROOT))

from splinewire.chain import load_chain_spec, pins_from_turns  # noqa: E402
from splinewire.synthetic import circle_wrap_pins, s_curve_pins  # noqa: E402

OUT = REPO / "out" / "cv-benchmark"
_FIDUCIALS = {"dot": dict(fiducial="dot", fiducial_mm=5.0, ring_inner_mm=0.0),
              "ring": dict(fiducial="ring", fiducial_mm=5.0, ring_inner_mm=2.0)}


def _load_spec():
    spec = load_chain_spec(ROOT / "data" / "chain.yaml")
    kind = os.environ.get("SPLINEWIRE_FIDUCIAL")
    if kind and hasattr(spec, "fiducial"):              # older packages only knew rings
        spec = replace(spec, **_FIDUCIALS[kind])
    return spec


SPEC = _load_spec()


def _size_mm(spec) -> float:
    return getattr(spec, "fiducial_mm", None) or spec.ring_outer_mm


def shapes(spec=SPEC) -> dict[str, np.ndarray]:
    return {
        "s-curve": s_curve_pins(spec),
        "pipe": circle_wrap_pins(spec, 30.0),
        "cove": circle_wrap_pins(spec, 45.0, concave=True),
        "tight": circle_wrap_pins(spec, 20.0),
        "wave": pins_from_turns(spec.pitch_mm, np.radians(28.0 * np.sin(0.9 * np.arange(spec.n_pins - 2)))),
    }


def scene_list(n_random: int, seeds: int = 3) -> list[tuple]:
    """(scene id, environment, shape, seed). Each seed varies shape and view direction."""
    from splinewire.scene import PRESETS, random_environment
    names = list(shapes())
    out = []
    for name, env in PRESETS.items():
        for s in range(seeds):
            e = replace(env, tilt_direction_deg=env.tilt_direction_deg + 97 * s,
                        roll_deg=env.roll_deg + 61 * s)
            out.append((f"{name}-{s}", e, names[(s + len(out)) % len(names)], s))
    rng = np.random.default_rng(1234)
    for k in range(n_random):
        out.append((f"random-{k:03d}", random_environment(rng, "random"), names[k % len(names)], 100 + k))
    return out


def render_one(args) -> str:
    from splinewire.scene import render_scene
    sid, env, shape, seed, design, out_dir = args
    img_path = out_dir / f"{sid}.png"
    meta_path = out_dir / f"{sid}.json"
    if img_path.exists() and meta_path.exists():
        return sid
    pins = shapes()[shape]
    sc = render_scene(pins, SPEC, env, design=design, seed=seed)
    cv2.imwrite(str(img_path), sc.image)
    meta = {"id": sid, "environment": env.name, "shape": shape, "seed": seed, "design": design.name,
            "pins_mm": sc.pins_mm.tolist(), "pins_px": sc.pins_px.tolist(), "focal_px": sc.focal_px,
            "px_per_mm": env.px_per_mm, "env": _env_dict(env)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return sid


def _env_dict(env) -> dict:
    d = asdict(env)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def render_suite(scenes, design, out_dir: Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(sid, env, shape, seed, design, out_dir) for sid, env, shape, seed in scenes
            if not (out_dir / f"{sid}.png").exists()]
    if not todo:
        return
    t = time.time()
    print(f"rendering {len(todo)} photos into {out_dir} ...", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(render_one, todo), 1):
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}  ({time.time() - t:.0f} s)", flush=True)


# ---------------------------------------------------------------------------
# Evaluation

def evaluate_one(args) -> dict:
    sid, out_dir = args
    import splinewire.pipeline as pipeline
    from splinewire.pipeline import compare_to_truth
    meta = json.loads((out_dir / f"{sid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(out_dir / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    truth_px = np.array(meta["pins_px"])
    truth_mm = np.array(meta["pins_mm"])
    ppm = meta["px_per_mm"]
    ring_px = _size_mm(SPEC) * ppm
    rec = {"id": sid, "environment": meta["environment"], "shape": meta["shape"],
           "filament": meta["env"].get("filament"), "glare": bool(meta["env"]["lamp_on_reflection"])}

    t = time.time()
    if hasattr(pipeline, "detect_fiducials"):
        rings = pipeline.detect_fiducials(img, SPEC)
    else:                                                   # before the dot
        rings = pipeline.detect_rings(img, SPEC.ring_inner_mm / SPEC.ring_outer_mm)
    rec["detect_s"] = time.time() - t
    centers = np.array([r.center_px for r in rings]).reshape(-1, 2)
    match = _match(centers, truth_px, 0.3 * ring_px)
    rec["found"] = int(np.sum(match >= 0))
    rec["false"] = int(len(centers) - rec["found"])
    errs = [float(np.linalg.norm(centers[m] - truth_px[i])) for i, m in enumerate(match) if m >= 0]
    rec["err_px_median"] = float(np.median(errs)) if errs else None
    rec["err_px_max"] = float(np.max(errs)) if errs else None

    rec.update(chain_ok=False, gaps=None, strays_in_chain=None, err_mm=None, err_mm_scaled=None)
    try:
        m = pipeline.measure(img, SPEC, meta["focal_px"])
    except Exception as e:           # noqa: BLE001 - any failure counts
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    got_px = m.pins_px
    idx = _match(truth_px, got_px, 0.3 * ring_px)       # measured pin -> true pin
    strays = int(np.sum(idx < 0))
    rec["gaps"] = len(m.order.gaps)
    rec["strays_in_chain"] = strays
    rec["pins_in_chain"] = len(got_px)
    if strays == 0 and len(got_px) >= 3:
        order = idx
        increasing = np.all(np.diff(order) > 0) or np.all(np.diff(order) < 0)
        rec["in_order"] = bool(increasing)
        t = compare_to_truth(m.pins_mm, truth_mm[order]) if increasing else None
        if t:
            rec["err_mm"] = t["max_error_mm"]
            rec["err_mm_scaled"] = t["max_error_scaled_mm"]
        rec["chain_ok"] = bool(increasing and len(got_px) == len(truth_mm))
    return rec


def _match(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """For each point of b, the index of the nearest point of a within tol, else -1."""
    out = np.full(len(b), -1)
    if len(a) == 0:
        return out
    d = np.linalg.norm(b[:, None] - a[None], axis=2)
    j = d.argmin(axis=1)
    ok = d[np.arange(len(b)), j] < tol
    out[ok] = j[ok]
    return out


def evaluate(ids: list[str], out_dir: Path, workers: int, rev: str | None = None) -> list[dict]:
    """Evaluate the working tree, or the package as of git revision `rev`."""
    if rev is None:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(evaluate_one, [(sid, out_dir) for sid in ids]))
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "archive", rev, "splinewire", "data"], cwd=REPO,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        records = Path(tmp) / "records.json"
        subprocess.run([sys.executable, __file__, "--evaluate", str(records), "--out", str(out_dir),
                        "--workers", str(workers), "--ids", *ids],
                       check=True, env={**os.environ, "SPLINEWIRE_ROOT": tmp})
        return json.loads(records.read_text(encoding="utf-8"))


def summarize(records: list[dict], title: str) -> str:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["environment"], []).append(r)
    lines = [f"### {title}", "",
             "| environment | n | found | false | chain ok | err px med / max | err mm worst |",
             "|---|---|---|---|---|---|---|"]
    order = [r["environment"] for r in records]
    for name in sorted(groups, key=lambda g: order.index(g) if g in order else 999):
        g = groups[name]
        n = len(g)
        found = np.mean([r["found"] for r in g])
        false = np.mean([r["false"] for r in g])
        ok = sum(r["chain_ok"] for r in g)
        med = [r["err_px_median"] for r in g if r["err_px_median"] is not None]
        mx = [r["err_px_max"] for r in g if r["err_px_max"] is not None]
        mm = [r["err_mm"] for r in g if r["err_mm"] is not None and r["chain_ok"]]
        lines.append(
            f"| {name} | {n} | {found:.1f}/13 | {false:.1f} | {ok}/{n} | "
            f"{np.median(med) if med else float('nan'):.2f} / {np.max(mx) if mx else float('nan'):.2f} | "
            f"{np.max(mm) if mm else float('nan'):.3f} |")
    rnd = [r for r in records if r["environment"] == "random"]
    if rnd:
        lines += ["", "Random environments by filament (lamp reflection on the chain / not):", ""]
        for fil in ("matte", "basic", "glossy"):
            for glare in (True, False):
                g = [r for r in rnd if r["filament"] == fil and r["glare"] == glare]
                if g:
                    lines.append(f"- {fil}, {'lamp reflection on chain' if glare else 'no direct reflection'}: "
                                 f"chain ok {sum(r['chain_ok'] for r in g)}/{len(g)}, "
                                 f"pins found {np.mean([r['found'] for r in g]):.1f}/13")
    all_ok = sum(r["chain_ok"] for r in records)
    mm = np.array([r["err_mm"] for r in records if r["chain_ok"] and r["err_mm"] is not None])
    lines += ["", f"**Total:** chain recovered in {all_ok}/{len(records)} photos; "
              f"worst-pin error median {np.median(mm):.3f} mm, 95th pct {np.percentile(mm, 95):.3f} mm, "
              f"max {mm.max():.3f} mm; photos over 0.5 mm: {int(np.sum(mm > 0.5))}." if len(mm) else ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=80, help="random environments to add")
    ap.add_argument("--seeds", type=int, default=3, help="photos per preset")
    ap.add_argument("--baseline", help="also evaluate the package as of this git revision")
    ap.add_argument("--only", nargs="*", help="only these environments")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--fiducial", choices=sorted(_FIDUCIALS), help="instead of chain.yaml's")
    ap.add_argument("--out", type=Path, help=f"render cache (default {OUT}/<fiducial>)")
    ap.add_argument("--json", type=Path, help="write per-photo records here")
    ap.add_argument("--evaluate", type=Path, help=argparse.SUPPRESS)   # internal: see evaluate()
    ap.add_argument("--ids", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()
    global SPEC
    if args.fiducial:
        os.environ["SPLINEWIRE_FIDUCIAL"] = args.fiducial   # for workers and --baseline
        SPEC = _load_spec()
    if args.out is None:
        args.out = OUT / SPEC.fiducial

    if args.evaluate:
        recs = evaluate(args.ids, args.out, args.workers)
        args.evaluate.write_text(json.dumps(recs), encoding="utf-8")
        return

    from splinewire.fiducials import chain_design
    scenes = scene_list(args.random, args.seeds)
    if args.only:
        scenes = [s for s in scenes if s[1].name in args.only]
    render_suite(scenes, chain_design(SPEC), args.out, args.workers)
    ids = [sid for sid, *_ in scenes]

    runs = [("current code", None)] + ([(f"code at {args.baseline}", args.baseline)] if args.baseline else [])
    all_records = {}
    for title, rev in runs:
        recs = evaluate(ids, args.out, args.workers, rev)
        all_records[title] = recs
        print(summarize(recs, title))
        print()
    if args.json:
        args.json.write_text(json.dumps(all_records, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
