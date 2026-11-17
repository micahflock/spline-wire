"""Is a video frame accurate enough to measure from? (web-build spike S5)

The web build's live overlay sees the chain in a camera video stream
(1080p, sometimes 4K); decision D7 says the measurement still comes from one
full-resolution still. This renders the benchmark scenes (cv_benchmark.py's
presets, optionally its random environments) as video frames and as the
12 MP still taken at the same moment, adds motion blur to the frames, runs
detection and the whole pipeline, and reports per resolution x blur level:

  chain ok   every pin found, in order, nothing stray (cv_benchmark's test),
             with the true focal length / with the focal estimated from the
             chain (measure(..., focal_px=None), what process.py does when a
             photo has no EXIF focal: the "estimated" focal source)
  worst      worst-pin error in mm after a rigid fit to truth, over the
             photos whose chain was recovered: median / 95th pct / max
  rms        median of each photo's rms pin error, mm
  f err      median |estimated / true focal - 1|

Framing. One phone position per scene, set by the video frame: the chain's
longest extent (pins plus the link width) spans FILL = 0.65 of the 16:9
frame's height, its short side, so the chain fits at any roll. Video is a
16:9 crop of the 4:3 sensor at full width, magnified by EIS_CROP = 1.1 for
electronic stabilisation; the still from the same position is 4032 x 3024
with the scene's focal length (24-28 mm equivalent, 26 in the presets). The
chain then images at about the same px/mm in the 12 MP still and the 4K
frame (4K: x 3840/4032 x 1.1), half that in 1080p.

Conditions:
  still-doc   the cv_benchmark render itself (the preset's own distance and
              image size), to check against docs/cv-robustness.md; shares
              cv_benchmark's cache in out/cv-benchmark/<fiducial>/
  still       12 MP from the video framing's position, no added blur
  2160p, 1080p, 720p
              video frames, each at BLUR levels of 0, 2, 4, 8 px (given at
              1080p, scaled with frame width: 4K gets 0, 4, 8, 16 px). The
              preset's own shake is kept, scaled to the frame; an added
              level replaces it when larger. Blur is scene.py's linear
              motion kernel (applied in the optics, before sensor noise) at
              a direction drawn per scene.

What a blur level means, at 1080p (f = 1586 px for a 26 mm still lens):
2 px is 0.072 deg of camera rotation during the exposure, i.e. 2.2 deg/s at
1/30 s or 4.3 deg/s at 1/60 s; or, at the ~7.8 px/mm (median) the framing gives,
0.26 mm of sideways travel: ~8 mm/s at 1/30 s, ~15 mm/s at 1/60 s. 4 and 8 px
are twice and four times that. A phone held still shakes at roughly
0.5-3 deg/s; moving it to frame the chain is 5-20 deg/s.

The video frames keep the preset's sensor model (noise, denoise, sharpening,
JPEG quality): a frame grabbed from a getUserMedia stream is not codec
compressed, and binning lowers its noise, so this is if anything harsh.

The chain is data/chain.yaml's, with max_bend_deg set by --max-bend
(default 80, at which docs/cv-robustness.md's figures were measured;
chain.yaml says 125, which the docs found gives the same dot results).

    uv run python experiments/video_frame_accuracy.py               # 29 presets x 3 seeds
    uv run python experiments/video_frame_accuracy.py --quick       # 9 presets x 1 seed, blur 0 and 4
    uv run python experiments/video_frame_accuracy.py --random 40 --json out/video-frame-accuracy/records.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import cv_benchmark as cvb  # noqa: E402
from splinewire.camera import focal_35mm_from_px, focal_px_from_35mm  # noqa: E402

OUT = REPO / "out" / "video-frame-accuracy"
STILL = (4032, 3024)
VIDEO = {"2160p": (3840, 2160), "1080p": (1920, 1080), "720p": (1280, 720)}
FILL = 0.65            # chain's longest extent / video frame height
EIS_CROP = 1.1         # video field of view narrowed by stabilisation
BLUR = (0.0, 2.0, 4.0, 8.0)          # px at 1080p
MEM_CAP_GB = 4           # per evaluation worker
QUICK_PRESETS = ["ideal", "daylight", "shake", "dim", "shadow", "wood", "text", "clutter", "worst"]

SPEC = cvb.SPEC


def _extent_mm(pins: np.ndarray) -> float:
    d = np.linalg.norm(pins[:, None] - pins[None], axis=2)
    return float(d.max()) + 2 * SPEC.half_width_mm


def conditions(blur: tuple[float, ...]) -> list[tuple[str, str | None, tuple[int, int] | None, float]]:
    """(condition name, kind, frame size, added blur in px of that frame)."""
    out = [("still-doc", None, None, 0.0), ("still", "still", STILL, 0.0)]
    for name, size in VIDEO.items():
        for b in blur:
            px = round(b * size[0] / 1920, 1)
            out.append((f"{name} blur {px:g}", "video", size, px))
    return out


def framed_env(env, shape: str, sid: str, kind: str, size, blur_px: float):
    """The scene's environment as seen from the video framing's phone position."""
    pins = cvb.shapes()[shape]
    ppm_1080 = FILL * 1080 / _extent_mm(pins)                  # px/mm in a 1920-wide frame
    f_still = focal_px_from_35mm(env.focal_35mm, STILL)
    if kind == "still":
        return replace(env, image_size=STILL, px_per_mm=ppm_1080 * STILL[0] / 1920 / EIS_CROP)
    w = size[0]
    scale = w * EIS_CROP / STILL[0]                            # frame px per still px
    shake = env.motion_blur_px * scale
    return replace(env, image_size=size, px_per_mm=ppm_1080 * w / 1920,
                   focal_35mm=focal_35mm_from_px(f_still * scale, size),
                   motion_blur_px=max(shake, blur_px),
                   motion_angle_deg=(env.motion_angle_deg if blur_px <= shake
                                     else float(zlib.crc32(sid.encode()) % 180)))


def _cond_dir(cond: str) -> Path:
    if cond == "still-doc":
        return cvb.OUT / SPEC.fiducial                          # cv_benchmark's own cache
    return OUT / SPEC.fiducial / cond.replace(" ", "-")


def render_all(scenes, conds, workers: int) -> None:
    from splinewire.fiducials import chain_design
    design = chain_design(SPEC)
    todo = []
    for cond, kind, size, blur in conds:
        d = _cond_dir(cond)
        d.mkdir(parents=True, exist_ok=True)
        for sid, env, shape, seed in scenes:
            if (d / f"{sid}.png").exists() and (d / f"{sid}.json").exists():
                continue
            e = env if kind is None else framed_env(env, shape, sid, kind, size, blur)
            todo.append((sid, e, shape, seed, design, d))
    if not todo:
        return
    # Biggest first, so the pool ends on short jobs.
    todo.sort(key=lambda a: -a[1].image_size[0] * a[1].image_size[1])
    t = time.time()
    print(f"rendering {len(todo)} photos into {OUT} ...", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(cvb.render_one, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ({time.time() - t:.0f} s)", flush=True)


# ---------------------------------------------------------------------------
# Evaluation

def _score(m, truth_px, truth_mm, tol) -> dict:
    from splinewire.pipeline import compare_to_truth
    idx = cvb._match(truth_px, m.pins_px, tol)
    r = {"chain_ok": False, "err_mm": None, "rms_mm": None, "pins_in_chain": len(m.pins_px)}
    if np.all(idx >= 0) and len(idx) >= 3:
        increasing = bool(np.all(np.diff(idx) > 0) or np.all(np.diff(idx) < 0))
        if increasing:
            c = compare_to_truth(m.pins_mm, truth_mm[idx])
            r["err_mm"], r["rms_mm"] = c["max_error_mm"], c["rms_error_mm"]
        r["chain_ok"] = increasing and len(idx) == len(truth_mm)
    return r


def evaluate_one(args) -> dict:
    cond, sid, max_bend = args
    # A photo with thousands of look-alikes (woven fabric, close up) makes the
    # ordering's pairwise arrays tens of GB; cap memory so it fails here, as
    # MemoryError, instead of taking the machine down.
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP_GB << 30, MEM_CAP_GB << 30))
    from splinewire.detect import detect_fiducials
    from splinewire.pipeline import measure
    spec = replace(SPEC, max_bend_deg=max_bend)
    d = _cond_dir(cond)
    meta = json.loads((d / f"{sid}.json").read_text(encoding="utf-8"))
    img = cv2.imread(str(d / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    truth_px, truth_mm = np.array(meta["pins_px"]), np.array(meta["pins_mm"])
    tol = 0.3 * cvb._size_mm(spec) * meta["px_per_mm"]
    span = truth_px.max(axis=0) - truth_px.min(axis=0)
    rec = {"condition": cond, "id": sid, "environment": meta["environment"], "shape": meta["shape"],
           "size": [w, h], "px_per_mm": meta["px_per_mm"], "motion_blur_px": meta["env"]["motion_blur_px"],
           "fill": float(span.max() / min(w, h)),
           "in_frame": bool((truth_px.min() >= 0) and (truth_px[:, 0].max() < w) and (truth_px[:, 1].max() < h))}
    t = time.time()
    try:
        fids = detect_fiducials(img, spec)
    except MemoryError:
        fids = None
    rec["detect_s"] = time.time() - t
    if fids is None:
        fail = {"chain_ok": False, "err_mm": None, "rms_mm": None, "error": "MemoryError in detection"}
        rec.update(found=0, false=0, err_px_median=None, err_px_max=None, true=fail, est=dict(fail))
        return rec
    centers = np.array([f.center_px for f in fids]).reshape(-1, 2)
    match = cvb._match(centers, truth_px, tol)
    rec["found"] = int(np.sum(match >= 0))
    rec["false"] = int(len(centers) - rec["found"])
    errs = [float(np.linalg.norm(centers[m] - truth_px[i])) for i, m in enumerate(match) if m >= 0]
    rec["err_px_median"] = float(np.median(errs)) if errs else None
    rec["err_px_max"] = float(np.max(errs)) if errs else None
    for key, focal in (("true", meta["focal_px"]), ("est", None)):
        try:
            m = measure(img, spec, focal, detected=fids)
        except Exception as e:              # noqa: BLE001 - any failure counts
            rec[key] = {"chain_ok": False, "err_mm": None, "rms_mm": None, "error": f"{type(e).__name__}: {e}"}
            continue
        rec[key] = _score(m, truth_px, truth_mm, tol)
        if focal is None:
            rec[key]["focal_ratio"] = float(m.rectification.focal_px / meta["focal_px"])
    return rec


def _rec_path(cond: str, sid: str, max_bend: float) -> Path:
    return OUT / SPEC.fiducial / "records" / f"bend{max_bend:g}" / cond.replace(" ", "-") / f"{sid}.json"


def _evaluate_job(job) -> tuple:
    return job, evaluate_one(job)


def evaluate(scenes, conds, workers: int, max_bend: float) -> list[dict]:
    """Records are cached per photo, so an interrupted run resumes. Workers are
    recycled (their memory grows) and capped at MEM_CAP_GB each."""
    import multiprocessing
    jobs = [(c[0], sid, max_bend) for c in conds for sid, *_ in scenes]
    todo = [j for j in jobs if not _rec_path(*j[:2], max_bend).exists()]
    t = time.time()
    if todo:
        print(f"evaluating {len(todo)} photos ...", flush=True)
        done = len(jobs) - len(todo)
        with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=8) as pool:
            for j, rec in pool.imap_unordered(_evaluate_job, todo):
                path = _rec_path(*j[:2], max_bend)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rec), encoding="utf-8")
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(jobs)}  ({time.time() - t:.0f} s)", flush=True)
    return [json.loads(_rec_path(*j[:2], max_bend).read_text(encoding="utf-8")) for j in jobs]


# ---------------------------------------------------------------------------
# Report

def _fmt(v, nd=3) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _stats(recs, key) -> tuple[int, str, str]:
    ok = [r for r in recs if r[key]["chain_ok"] and r[key]["err_mm"] is not None]
    worst = np.array([r[key]["err_mm"] for r in ok])
    rms = np.array([r[key]["rms_mm"] for r in ok])
    if not len(ok):
        return 0, "–", "–"
    w = f"{np.median(worst):.3f} / {np.percentile(worst, 95):.3f} / {worst.max():.3f}"
    return len(ok), w, f"{np.median(rms):.3f}"


def summarize(records: list[dict], conds) -> str:
    by = {}
    for r in records:
        by.setdefault(r["condition"], []).append(r)
    lines = ["| condition | px/mm | blur px | found | chain ok (true f / est f) | worst mm, true f: med / p95 / max "
             "| rms mm | worst mm, est f: med / p95 / max | rms mm | f err |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cond, *_ in conds:
        g = by.get(cond, [])
        if not g:
            continue
        n = len(g)
        n_t, w_t, r_t = _stats(g, "true")
        n_e, w_e, r_e = _stats(g, "est")
        fr = [abs(r["est"]["focal_ratio"] - 1) for r in g if r["est"].get("focal_ratio") and r["est"]["chain_ok"]]
        lines.append(f"| {cond} | {np.nanmedian([r['px_per_mm'] for r in g]):.1f} | "
                     f"{np.nanmedian([r['motion_blur_px'] for r in g]):.0f} | "
                     f"{np.mean([r['found'] for r in g]):.1f}/{SPEC.n_pins} | {n_t}/{n} · {n_e}/{n} | "
                     f"{w_t} | {r_t} | {w_e} | {r_e} | {_fmt(float(np.median(fr)) * 100 if fr else None, 1)}% |")
    # Paired: only scenes every condition recovered (true focal), so the error
    # columns compare like with like.
    ids = {r["id"] for r in records}
    common = sorted(i for i in ids if all(r["true"]["chain_ok"] for r in records if r["id"] == i))
    lines += ["", f"Paired, on the {len(common)} scenes every condition recovered: worst-pin mm, true f "
              "(median / max)  |  est f (median / max)", ""]
    for cond, *_ in conds:
        g = [r for r in by.get(cond, []) if r["id"] in common]
        if not g:
            continue
        wt = np.array([r["true"]["err_mm"] for r in g])
        we = np.array([r["est"]["err_mm"] if r["est"]["chain_ok"] else np.nan for r in g])
        lines.append(f"- {cond}: {np.median(wt):.3f} / {wt.max():.3f}  |  "
                     f"{np.nanmedian(we):.3f} / {np.nanmax(we):.3f} ({int(np.sum(np.isnan(we)))} lost)")
    # Failures by environment.
    fails = {}
    for r in records:
        if not r["true"]["chain_ok"]:
            fails.setdefault(r["condition"], []).append(f"{r['id']}({r['found']})")
    lines += ["", "Not recovered with the true focal (scene, pins found):", ""]
    for cond, *_ in conds:
        if cond in fails:
            lines.append(f"- {cond}: {', '.join(fails[cond])}")
    out_of_frame = sorted({r["id"] for r in records if not r["in_frame"]})
    fill = [r["fill"] for r in records if r["condition"].startswith("1080p") and not r.get("killed")]
    lines += ["", f"Chain span / frame short side at 1080p: median {np.median(fill):.2f}, "
              f"max {np.max(fill):.2f}; scenes with a pin out of frame: {out_of_frame or 'none'}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help=f"{len(QUICK_PRESETS)} presets x 1 seed, blur 0 and 4 px")
    ap.add_argument("--seeds", type=int, default=3, help="photos per preset (as cv_benchmark)")
    ap.add_argument("--random", type=int, default=0, help="also cv_benchmark's first N random environments")
    ap.add_argument("--only", nargs="*", help="only these environments")
    ap.add_argument("--blur", type=float, nargs="*", help="blur levels, px at 1080p (default 0 2 4 8)")
    ap.add_argument("--max-bend", type=float, default=80.0, help="ordering's bend limit (docs' figures: 80)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--json", type=Path, help="write per-photo records here")
    args = ap.parse_args()

    blur = tuple(args.blur) if args.blur else ((0.0, 4.0) if args.quick else BLUR)
    scenes = cvb.scene_list(args.random, 1 if args.quick else args.seeds)
    only = args.only or (QUICK_PRESETS if args.quick else None)
    if only:
        scenes = [s for s in scenes if s[1].name in only]
    conds = conditions(blur)
    print(f"{len(scenes)} scenes x {len(conds)} conditions; chain {SPEC.fiducial} {SPEC.fiducial_mm:g} mm, "
          f"max_bend_deg {args.max_bend:g}; fill {FILL}, EIS crop {EIS_CROP}", flush=True)
    t0 = time.time()
    render_all(scenes, conds, args.workers)
    t1 = time.time()
    records = evaluate(scenes, conds, args.workers, args.max_bend)
    t2 = time.time()
    print()
    print(summarize(records, conds))
    print(f"\nrender {t1 - t0:.0f} s, evaluate {t2 - t1:.0f} s")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
