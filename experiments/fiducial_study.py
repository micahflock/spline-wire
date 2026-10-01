"""Which fiducial design holds up best when printed with a 0.4 mm nozzle?

Every design (splinewire/fiducials.py) is rendered through the same print
model (corner rounding, gap closing, over-extrusion, seams, wobble, relief,
extrusion-line sheen) and the same environments (splinewire/scene.py), then
measured with a detector suited to it:

  ring, ring-6   the production detector (splinewire/detect.py)
  dot            the production dot detector (splinewire/detect.py)
  bullseye       the production candidate stage, then all four edges fitted
  ring-x         ring for detection, then the checker corner's saddle point
                 (cv2.cornerSubPix) for the centre, which is exact under
                 perspective, unlike an ellipse centre
  aruco          OpenCV ArUco (inverted markers), ids give the chain order

and then the same ordering (for unlabeled designs) and deskewing.

Three parts:
  1. environments: preset + random environments, as in cv_benchmark.py
  2. resolution:   px per mm swept down until each design stops working
  3. blur:         hand-shake blur swept up

    uv run python experiments/fiducial_study.py              # all parts
    uv run python experiments/fiducial_study.py --part resolution
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from splinewire import detect as D  # noqa: E402
from splinewire.chain import load_chain_spec  # noqa: E402
from splinewire.fiducials import ARUCO, ARUCO_SMALL, BULLSEYE, DOT, RING, RING6, RING_X, Design  # noqa: E402
from splinewire.order import ChainOrder, order_chain  # noqa: E402
from splinewire.pipeline import _misfits, compare_to_truth  # noqa: E402
from splinewire.rectify import rectify_chain  # noqa: E402
from splinewire.scene import BASE, PRESETS, random_environment, render_scene  # noqa: E402

SPEC = load_chain_spec(REPO / "data" / "chain.yaml")
OUT = REPO / "out" / "fiducial-study"
DESIGNS: dict[str, Design] = {d.name: d for d in [RING, RING6, DOT, BULLSEYE, RING_X, ARUCO, ARUCO_SMALL]}

# Relative edge radii (outer = 1) and, from the outside in, the zones' shades.
_EDGES = {
    "ring": [1.0, 0.4],
    "ring-6": [1.0, 0.4],
    "dot": [1.0],
    "bullseye": [1.0, 2.2 / 3.0, 1.4 / 3.0, 0.6 / 3.0],
    "ring-x": [1.0, 2.3 / 3.1],
}
# Where the innermost zone used for levels stops (ring-x: its checker centre
# is half white, so the dark level comes from the black gap around it).
_FLOOR = {"ring-x": 1.5 / 3.1}


# ---------------------------------------------------------------------------
# Detection, per design. Each returns centres (N, 2), sizes (N, 2) (outer
# ellipse major/minor diameters) and ids (N,) or None.

def detect(design: str, img: np.ndarray):
    if design.startswith("aruco"):
        return _detect_aruco(img)
    if design == "dot":                                  # the production dot detector
        found = D.detect_dots(img)
        return (np.array([f.center_px for f in found]).reshape(-1, 2),
                np.array([f.outer_axes_px for f in found]).reshape(-1, 2), None,
                [(1, f.surround) for f in found])
    edges = _EDGES[design]
    ratio = edges[1] if len(edges) > 1 else None
    cands = _candidates(img, ratio)
    grayf = img.astype(np.float32)
    found = []
    for c in cands:
        r = _refine_multi(grayf, c, edges, design)
        if r is None:
            continue
        if design == "ring-x":
            r = _saddle_centre(grayf, r)
            if r is None:
                continue
        found.append(r)
    found = _dedupe(found)
    if not found:
        return np.zeros((0, 2)), np.zeros((0, 2)), None, []
    return (np.array([f["center"] for f in found]), np.array([f["axes"] for f in found]), None,
            [(f["polarity"], f["surround"]) for f in found])


def _candidates(img, ratio):
    """The production candidate stage; for a dot, solid blobs instead of rings."""
    if ratio is not None:
        out = []
        level, scale = img, 1.0
        while True:
            g = cv2.GaussianBlur(level, (0, 0), 0.8 if scale == 1 else 1.0)
            for mask, pol in D._masks(g):
                out += D._ring_candidates(mask, pol, ratio, 8.0 if scale == 1 else 14.0, scale)
            if min(level.shape) < 2 * D._MIN_LEVEL_SIDE or scale >= 4:
                break
            level = cv2.resize(level, (level.shape[1] // 2, level.shape[0] // 2), interpolation=cv2.INTER_AREA)
            scale *= 2
        return D._dedupe_candidates(out)
    out = []
    level, scale = img, 1.0
    while True:
        g = cv2.GaussianBlur(level, (0, 0), 0.8 if scale == 1 else 1.0)
        for mask, pol in D._masks(g):
            # 10 px: a solid dot wider than the threshold window is hollowed out at
            # full size, so it must be caught small at a coarser level.
            out += _blob_candidates(mask, pol, 8.0 if scale == 1 else 10.0, scale)
        if min(level.shape) < 2 * D._MIN_LEVEL_SIDE or scale >= 4:
            break
        level = cv2.resize(level, (level.shape[1] // 2, level.shape[0] // 2), interpolation=cv2.INTER_AREA)
        scale *= 2
    return D._dedupe_candidates(out)


def _blob_candidates(mask, polarity, min_d, scale):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    out = []
    hierarchy = hierarchy[0]
    for i, (_n, _p, child, parent) in enumerate(hierarchy):
        if parent != -1 or len(contours[i]) < 12:
            continue
        # solid: no holes except noise specks (as the ring stage allows)
        min_hole = max(4.0, 0.02 * cv2.contourArea(contours[i]))
        if any(cv2.contourArea(contours[c]) >= min_hole for c in D._children(hierarchy, child)):
            continue
        e = cv2.fitEllipse(contours[i])
        (cx, cy), (a, b), ang = e
        if min(a, b) < min_d or min(a, b) / max(a, b) < 0.3 or not D._ellipse_like(contours[i], e):
            continue
        off = (scale - 1) / 2
        out.append(D._Candidate(center=(cx * scale + off, cy * scale + off),
                                outer=((cx * scale + off, cy * scale + off), (a * scale, b * scale), ang),
                                inner_ratio=0.0, polarity=polarity))
    return out


def _refine_multi(gray, cand, edges: list[float], design: str, passes: int = 2):
    """Ray-profile refinement with any number of concentric edges.

    Zones alternate bright/dark from the outside in: outside the outer edge
    is dark, then bright, then dark, ... (after flipping for polarity).
    """
    (cx, cy), (A, B), ang = cand.outer
    fits = None
    for _ in range(passes):
        R = max(A, B) / 2
        if not np.isfinite(R) or R < 3:
            return None
        n_rays = int(np.clip(math.pi * R, 32, 180))
        phi = np.arange(n_rays) * (2 * math.pi / n_rays)
        t = math.radians(ang)
        ex = (A / 2) * np.cos(phi) * math.cos(t) - (B / 2) * np.sin(phi) * math.sin(t)
        ey = (A / 2) * np.cos(phi) * math.sin(t) + (B / 2) * np.sin(phi) * math.cos(t)
        s_max = 1.35
        n_s = int(math.ceil(s_max * R / 0.25)) + 1
        s = np.linspace(0.0, s_max, n_s, dtype=np.float32)
        ds = float(s[1] - s[0])
        mx = (cx + s[None, :] * ex[:, None]).astype(np.float32)
        my = (cy + s[None, :] * ey[:, None]).astype(np.float32)
        q = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if cand.polarity < 0:
            q = 255.0 - q

        def idx(v):
            return int(np.clip(round(v / ds), 0, n_s - 1))

        # zone boundaries from the outside in: [1.3, 1.1] outside, then between edges
        bounds = [1.3] + edges + [_FLOOR.get(design, 0.0)]
        levels = []
        for k in range(len(bounds) - 1):
            hi_s, lo_s = bounds[k], bounds[k + 1]
            if k == 0:
                a, b = 1.1, 1.3
            else:
                w = hi_s - lo_s
                a, b = lo_s + 0.25 * w, hi_s - 0.25 * w
            levels.append(np.median(q[:, idx(a):idx(b) + 1], axis=1))
        fits = []
        for k, e in enumerate(edges):
            outside, inside = levels[k], levels[k + 1]
            level = (outside + inside) / 2
            w_out = (bounds[k] - e) if k else 0.3
            w_in = e - bounds[k + 2]
            sc = D._crossings(q, level, idx(e - 0.5 * w_in), idx(e + 0.5 * w_out), e / ds, ds,
                              rising=bool(np.median(inside - outside) < 0))
            contrast = np.abs(outside - inside)
            good = np.isfinite(sc) & (contrast > max(6.0, 0.25 * float(np.median(contrast))))
            pts = np.c_[cx + sc * ex, cy + sc * ey][good]
            if len(pts) < 0.5 * n_rays:
                return None
            f = D._robust_ellipse(pts)
            if f is None or f[2] < 0.5 * n_rays:
                return None
            fits.append(f)
        eo = fits[0][0]
        (cx, cy), (A, B), ang = eo
    w = np.array([n / max(res, 0.05) ** 2 for _, res, n in fits])
    centres = np.array([f[0][0] for f in fits])
    centre = (w[:, None] * centres).sum(axis=0) / w.sum()
    eo = fits[0][0]
    if max(np.linalg.norm(centres - centre, axis=1)) > 0.08 * min(eo[1]) + 0.5:
        return None
    if len(edges) > 1:
        ratio = math.sqrt(fits[1][0][1][0] * fits[1][0][1][1] / (eo[1][0] * eo[1][1]))
        if abs(ratio - edges[1]) > 0.5 * (1 - edges[1]) + 0.1:
            return None
    res = math.sqrt(sum(n * r * r for _, r, n in fits) / sum(n for _, r, n in fits))
    if res > max(0.6, 0.04 * min(eo[1])):
        return None
    surround = float(np.median(levels[0]))
    return {"center": tuple(centre), "axes": (max(eo[1]), min(eo[1])), "outer": eo,
            "polarity": cand.polarity, "residual": res,
            "surround": surround if cand.polarity > 0 else 255.0 - surround}


def _saddle_centre(gray, r):
    """Replace the ellipse centre with the checker corner's saddle point."""
    (_, (A, B), _) = r["outer"]
    win = max(2, int(round(0.2 * min(A, B))))     # ~ the checker disc's radius x 0.8
    pt = np.array([[r["center"]]], np.float32)
    g = cv2.GaussianBlur(gray, (0, 0), 0.7)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    try:
        cv2.cornerSubPix(g, pt, (win, win), (-1, -1), crit)
    except cv2.error:
        return None
    new = pt[0, 0]
    # The ellipse centre is good to a few tenths of a pixel; a saddle further
    # off latched onto another junction (where a quadrant edge meets the gap).
    if np.hypot(*(new - np.array(r["center"]))) > max(0.6, 0.03 * min(A, B)):
        return {**r, "saddle": False}
    return {**r, "center": (float(new[0]), float(new[1])), "ellipse_center": r["center"], "saddle": True}


def _dedupe(found):
    kept = []
    for f in sorted(found, key=lambda f: -f["axes"][1]):
        if all(np.hypot(f["center"][0] - k["center"][0], f["center"][1] - k["center"][1]) > 0.5 * k["axes"][1]
               or f["polarity"] != k["polarity"] for k in kept):
            kept.append(f)
    return kept


def _detect_aruco(img):
    params = cv2.aruco.DetectorParameters()
    params.detectInvertedMarker = True
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO.windows[0].dictionary), params)
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int), []
    centres, sizes, keep = [], [], []
    for c, i in zip(corners, ids.ravel()):
        c = c.reshape(4, 2).astype(float)
        centres.append(_diagonal_intersection(c))
        side = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
        sizes.append((side.max(), side.min()))
        keep.append(int(i))
    return np.array(centres), np.array(sizes), np.array(keep), [(1, 0.0)] * len(keep)


def _diagonal_intersection(c):
    """Centre of a square under perspective: where its diagonals cross."""
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    t = cross(q - p, s) / cross(r, s)
    return p + t * r


# ---------------------------------------------------------------------------

def measure_design(design: str, img, focal_px, image_size):
    """Detect, order and deskew, like splinewire.pipeline.measure does for the ring."""
    centres, sizes, ids, pols = detect(design, img)
    out = {"centres": centres}
    if len(centres) < 3:
        return out, None
    excluded: set[int] = set()
    for _ in range(4):
        chain_idx, links = _order(design, centres, sizes, ids, pols, excluded)
        if chain_idx is None or len(links) < 5:
            return out, None
        rect = rectify_chain(centres[chain_idx], links, SPEC.pitch_mm, image_size, focal_px)
        if ids is not None:
            break                                        # ids leave nothing to second-guess
        rings = [SimpleNamespace(outer_axes_px=tuple(s), surround=p[1]) for s, p in zip(sizes, pols)]
        order = ChainOrder(indices=chain_idx, links=links, gaps=[], rejected=[])
        misfits = _misfits(rings, order, rect, SPEC)
        if not misfits:
            break
        excluded |= misfits
    out["chain_idx"] = chain_idx
    return out, rect


def _order(design, centres, sizes, ids, pols, excluded):
    if ids is not None:                                  # ArUco: order by id
        valid = ids < SPEC.n_pins
        order = np.argsort(ids[valid])
        idx = np.flatnonzero(valid)[order]
        pin_ids = ids[idx]
        links = [(k, k + 1) for k in range(len(idx) - 1) if pin_ids[k + 1] == pin_ids[k] + 1]
        return list(idx), links
    best = None
    for p in (1, -1):
        sel = [i for i, q in enumerate(pols) if q[0] == p and i not in excluded]
        if len(sel) < 3:
            continue
        o = order_chain(centres[sel], axes_px=sizes[sel],
                        pitch_per_diameter=SPEC.pitch_mm / DESIGNS[design].outer_mm)
        if best is None or len(o.indices) > len(best[1].indices):
            best = (sel, o)
    if best is None:
        return None, []
    sel, o = best
    return [sel[i] for i in o.indices], o.links


def evaluate_one(args):
    design, path = args
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    img = cv2.imread(path[:-5] + ".png", cv2.IMREAD_GRAYSCALE)
    truth_px, truth_mm = np.array(meta["pins_px"]), np.array(meta["pins_mm"])
    tol = 0.3 * DESIGNS[design].outer_mm * meta["px_per_mm"]
    t = time.time()
    try:
        out, rect = measure_design(design, img, meta["focal_px"], (img.shape[1], img.shape[0]))
    except Exception as e:                                # noqa: BLE001
        return {**_base(meta, design), "error": repr(e)}
    rec = {**_base(meta, design), "seconds": time.time() - t}
    c = out["centres"]
    d = np.linalg.norm(truth_px[:, None] - c[None], axis=2) if len(c) else np.full((len(truth_px), 1), np.inf)
    near = d.min(axis=1)
    rec["found"] = int(np.sum(near < tol))
    rec["false"] = int(len(c) - rec["found"])
    errs = near[near < tol]
    rec["err_px_median"] = float(np.median(errs)) if len(errs) else None
    rec["err_px_max"] = float(np.max(errs)) if len(errs) else None
    rec["chain_ok"] = False
    rec["err_mm"] = None
    if rect is not None:
        got = c[out["chain_idx"]]
        dd = np.linalg.norm(got[:, None] - truth_px[None], axis=2)
        j = dd.argmin(axis=1)
        ok = dd[np.arange(len(got)), j] < tol
        if ok.all() and (np.all(np.diff(j) > 0) or np.all(np.diff(j) < 0)):
            rec["err_mm"] = compare_to_truth(rect.pins_mm, truth_mm[j])["max_error_mm"]
            rec["chain_ok"] = len(got) == len(truth_mm)
    return rec


def _base(meta, design):
    return {"id": meta["id"], "design": design, "environment": meta["environment"],
            "px_per_mm": meta["px_per_mm"], "filament": meta["env"].get("filament")}


# ---------------------------------------------------------------------------
# Scenes

def _render(args):
    design_name, sid, env, shape, seed, out_dir = args
    from experiments.cv_benchmark import _env_dict, shapes
    img_path, meta_path = out_dir / f"{sid}.png", out_dir / f"{sid}.json"
    if img_path.exists() and meta_path.exists():
        return
    sc = render_scene(shapes()[shape], SPEC, env, design=DESIGNS[design_name], seed=seed)
    cv2.imwrite(str(img_path), sc.image)
    meta_path.write_text(json.dumps({
        "id": sid, "environment": env.name, "shape": shape, "design": design_name,
        "pins_mm": sc.pins_mm.tolist(), "pins_px": sc.pins_px.tolist(), "focal_px": sc.focal_px,
        "px_per_mm": env.px_per_mm, "env": _env_dict(env)}), encoding="utf-8")


def scenes(part: str, n_random: int):
    names = ["s-curve", "pipe", "cove", "tight", "wave"]
    out = []
    if part == "environments":
        skip = {"ideal", "glossy-print", "glare"}       # trivially easy or impossible for all
        for name, env in PRESETS.items():
            if name in skip:
                continue
            for s in range(2):
                e = replace(env, tilt_direction_deg=env.tilt_direction_deg + 97 * s, roll_deg=env.roll_deg + 61 * s)
                out.append((f"{name}-{s}", e, names[(s + len(out)) % 5], s))
        rng = np.random.default_rng(99)
        for k in range(n_random):
            out.append((f"random-{k:03d}", random_environment(rng), names[k % 5], 200 + k))
    elif part == "resolution":
        for ppm in (1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5):
            for s in range(3):
                e = replace(BASE, name=f"{ppm:.1f} px/mm", px_per_mm=ppm, image_size=(2048, 1536),
                            tilt_deg=20.0, tilt_direction_deg=30 + 97 * s, roll_deg=10 + 61 * s)
                out.append((f"res-{ppm:.1f}-{s}", e, names[s % 5], s))
    elif part == "blur":
        for blur in (0.0, 4.0, 8.0, 12.0, 16.0):
            for s in range(3):
                e = replace(BASE, name=f"shake {blur:.0f} px", px_per_mm=8.0, motion_blur_px=blur,
                            motion_angle_deg=20 + 50 * s, tilt_direction_deg=30 + 97 * s)
                out.append((f"blur-{blur:.0f}-{s}", e, names[s % 5], s))
    return out


def summarize(records, title):
    designs = list(DESIGNS)
    envs = list(dict.fromkeys(r["environment"] for r in records))
    lines = [f"### {title}", "", "Chain recovered (photos) / worst pin error, mm (median over recovered photos)", "",
             "| environment | " + " | ".join(designs) + " |", "|---" * (len(designs) + 1) + "|"]
    for env in envs + ["**all**"]:
        cells = []
        for d in designs:
            g = [r for r in records if r["design"] == d and (r["environment"] == env or env == "**all**")]
            if not g:
                cells.append("")
                continue
            ok = [r for r in g if r.get("chain_ok")]
            mm = [r["err_mm"] for r in ok if r.get("err_mm") is not None]
            cells.append(f"{len(ok)}/{len(g)}" + (f" · {np.median(mm):.3f}" if mm else ""))
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    lines += ["", "Pin centre error, px (median / 95th pct over found pins) and false detections per photo", "",
              "| design | centre err px | false/photo | pins found | s/photo |", "|---|---|---|---|---|"]
    for d in designs:
        g = [r for r in records if r["design"] == d and "found" in r]
        med = [r["err_px_median"] for r in g if r["err_px_median"] is not None]
        mx = [r["err_px_max"] for r in g if r["err_px_max"] is not None]
        lines.append(f"| {d} | {np.median(med):.3f} / {np.percentile(mx, 95):.3f} | "
                     f"{np.mean([r['false'] for r in g]):.1f} | {np.mean([r['found'] for r in g]):.1f}/13 | "
                     f"{np.mean([r['seconds'] for r in g]):.2f} |" if med else f"| {d} | – | – | – | – |")
    return "\n".join(lines)


def printability_figure(path: Path, tile_px: int = 180) -> None:
    """Each design as drawn, as a typical 0.4 mm-nozzle print, and as a bad
    print, straight down (black = black filament)."""
    from splinewire.scene import BAD_PRINT, PERFECT_PRINT, PrintQuality, printed_fiducial

    rows = [("design", PERFECT_PRINT), ("typical print", replace(PrintQuality(), relief_mm=0.0)),
            ("bad print", replace(BAD_PRINT, relief_mm=0.0))]
    label_w = 150
    header = 28
    out = np.full((header + len(rows) * tile_px, label_w + len(DESIGNS) * tile_px), 255, np.uint8)
    for j, name in enumerate(DESIGNS):
        cv2.putText(out, name, (label_w + j * tile_px + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    for i, (label, pq) in enumerate(rows):
        y = header + i * tile_px
        cv2.putText(out, label, (8, y + tile_px // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
        for j, d in enumerate(DESIGNS.values()):
            cover, _ = printed_fiducial(SPEC, d, pq, seed=j, half_size_mm=4.5)
            tile = np.round(255 * (0.93 - 0.85 * cover)).astype(np.uint8)
            tile = cv2.resize(tile, (tile_px - 6, tile_px - 6), interpolation=cv2.INTER_AREA)
            x = label_w + j * tile_px
            out[y + 3:y + tile_px - 3, x + 3:x + tile_px - 3] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", type=Path, help="only write the printability figure (PNG) here")
    ap.add_argument("--part", choices=["environments", "resolution", "blur", "all"], default="all")
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--designs", nargs="*", default=list(DESIGNS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.figure:
        printability_figure(args.figure)
        return
    parts = ["environments", "resolution", "blur"] if args.part == "all" else [args.part]
    all_recs = {}
    for part in parts:
        sc = scenes(part, args.random)
        jobs, evals = [], []
        for d in args.designs:
            out_dir = OUT / d
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs += [(d, sid, env, shape, seed, out_dir) for sid, env, shape, seed in sc]
            evals += [(d, str(out_dir / f"{sid}.json")) for sid, *_ in sc]
        t = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(_render, jobs, chunksize=2))
        print(f"[{part}] rendered {len(jobs)} photos in {time.time() - t:.0f} s", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            recs = list(ex.map(evaluate_one, evals, chunksize=2))
        all_recs[part] = recs
        print(summarize(recs, part), "\n", flush=True)
    if args.json:
        args.json.write_text(json.dumps(all_recs, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
