"""Scoring a measurement against a test target (splinewire.targets).

- A fixed-chain plaque: the measured pins against the plaque's pins
  (pipeline.compare_to_truth: after a rigid fit, and after a scale fit).
- A contact gauge: the measured contact points against the gauge's edge.
  The points are placed on the edge by the best rigid fit (no reflection),
  and each point's distance from the edge is its error. A round gauge also
  gets a fitted circle, whose radius is compared with the gauge's.

The chain's end points are scored with and without the ends: until the
chain has its end links, its first and last points are not measured like
the rest (docs/snap-chain.md).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from splinewire.chain import ChainSpec
from splinewire.contact import contact_points, spline_samples
from splinewire.pipeline import compare_to_truth
from splinewire.targets import GAUGES, PLAQUE_POSES, Gauge, plaque_pins

_SEED_MM = 2.0        # spacing of the starting places tried along a gauge's edge
_ICP_ROUNDS = 50
_REFINED = 8          # the best starting places (by their first fit) refined further


def score(target: str, pins_mm: np.ndarray, spec: ChainSpec, sign: int | None = None,
          gaps: Sequence[int] = ()) -> dict:
    """Score a measured chain (its pins, mm) against the named target.

    sign: the object's side of the pins (contact.object_sign). None tries
    both and keeps the better, for a gauge whose side the person set by
    tapping the photo (an S, a straight edge).
    """
    pins = np.asarray(pins_mm, dtype=float)
    if target in PLAQUE_POSES:
        return score_plaque(pins, plaque_pins(target, spec), target)
    gauge = GAUGES.get(target)
    if gauge is None:
        raise ValueError(f"no target {target!r}")
    best = None
    for s in ((sign,) if sign is not None else (1, -1)):
        contacts = contact_points(pins, spec.half_width_mm, s, spec.contact == "round", gaps=gaps)
        result = score_gauge(contacts, gauge)
        result["sign"] = s
        if best is None or result["inner"]["rms_mm"] < best["inner"]["rms_mm"]:
            best = result
    return best


def score_plaque(pins_mm: np.ndarray, truth_mm: np.ndarray, target: str) -> dict:
    """Measured pins against a plaque's. A chain read as cut short (fewer
    pins than the plaque has) is scored against the run of the plaque's pins
    it fits best, so what was found still counts."""
    n, total = len(pins_mm), len(truth_mm)
    if n > total:
        raise ValueError(f"found {n} pins, but {target} has {total}")
    if n < 3:
        raise ValueError(f"found {n} pins: too few to score")
    best = None
    for k in range(total - n + 1):
        result = compare_to_truth(pins_mm, truth_mm[k:k + n])
        if best is None or result["max_error_mm"] < best[1]["max_error_mm"]:
            best = (k, result)
    k, result = best
    return {"target": target, "kind": "plaque", "pins_matched": n, "pins_total": total,
            "first_pin": k, **result}


def score_gauge(contacts_mm: np.ndarray, gauge: Gauge) -> dict:
    """Contact points (in chain order) against a gauge: every point ("all")
    and without the two end points ("inner")."""
    pts = np.asarray(contacts_mm, dtype=float)
    if len(pts) < 5:
        raise ValueError(f"only {len(pts)} contact points: too few to score")
    edge = gauge.edge()
    tree = cKDTree(edge)
    out = {"target": gauge.name, "kind": "gauge"}
    for key, sel in (("all", pts), ("inner", pts[1:-1])):
        rot, shift, dist = _register(sel, edge, tree, gauge.closed)
        curve = spline_samples(sel) @ rot.T + shift
        part = {
            "points": len(sel),
            "max_mm": float(dist.max()),
            "rms_mm": float(np.sqrt(np.mean(dist ** 2))),
            "curve_max_mm": float(_nearest(curve, edge, tree)[0].max()),
        }
        if gauge.radius_mm is not None:
            center, radius = fit_circle(sel)
            residual = np.abs(np.linalg.norm(sel - center, axis=1) - radius)
            part.update(radius_mm=radius, radius_error_mm=radius - gauge.radius_mm,
                        circle_rms_mm=float(np.sqrt(np.mean(residual ** 2))))
        out[key] = part
    return out


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through points (geometric distances): (center, radius)."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = np.sum(pts ** 2, axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    guess = np.array([cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 1e-12))])
    fit = least_squares(lambda p: np.linalg.norm(pts - p[:2], axis=1) - p[2], guess)
    return fit.x[:2], float(fit.x[2])


def _register(pts: np.ndarray, edge: np.ndarray, tree: cKDTree, closed: bool
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rigid placement of the points on the edge that brings them closest
    to it: (rotation, shift, each point's distance from the edge).

    The chain has no labels and could lie anywhere along the edge, either way
    round, so the search starts from every place along the edge (every
    _SEED_MM, both directions), lays the points along it by their spacing,
    and refines each start by iterative closest points.
    """
    s_edge = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))]
    length = s_edge[-1]
    s_pts = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    starts = []
    for s0 in np.arange(0.0, length + 1e-9, _SEED_MM):
        for direction in (1, -1):
            s = s0 + direction * s_pts
            s = np.mod(s, length) if closed else np.clip(s, 0.0, length)
            target = np.c_[np.interp(s, s_edge, edge[:, 0]), np.interp(s, s_edge, edge[:, 1])]
            rot, shift = _procrustes(pts, target)
            rough = float(np.mean(tree.query(pts @ rot.T + shift)[0] ** 2))
            starts.append((rough, rot, shift))
    starts.sort(key=lambda x: x[0])
    best = None
    for _, rot, shift in starts[:_REFINED]:
        for _ in range(_ICP_ROUNDS):
            _, near = _nearest(pts @ rot.T + shift, edge, tree)
            new_rot, new_shift = _procrustes(pts, near)
            done = np.allclose(new_rot, rot, atol=1e-10) and np.allclose(new_shift, shift, atol=1e-9)
            rot, shift = new_rot, new_shift
            if done:
                break
        dist, _ = _nearest(pts @ rot.T + shift, edge, tree)
        rms = float(np.sqrt(np.mean(dist ** 2)))
        if best is None or rms < best[0]:
            best = (rms, rot, shift, dist)
    return best[1], best[2], best[3]


def _nearest(pts: np.ndarray, edge: np.ndarray, tree: cKDTree) -> tuple[np.ndarray, np.ndarray]:
    """Each point's distance from the edge (a dense polyline) and the nearest
    point on it: on the segments either side of the nearest vertex."""
    _, idx = tree.query(pts)
    best_d = np.full(len(pts), np.inf)
    best_p = np.zeros_like(pts)
    for lo in (idx - 1, idx):
        lo = np.clip(lo, 0, len(edge) - 2)
        a, b = edge[lo], edge[lo + 1]
        u = b - a
        t = np.clip(np.sum((pts - a) * u, axis=1) / np.maximum(np.sum(u * u, axis=1), 1e-18), 0.0, 1.0)
        q = a + t[:, None] * u
        d = np.linalg.norm(pts - q, axis=1)
        better = d < best_d
        best_d[better], best_p[better] = d[better], q[better]
    return best_d, best_p


def _procrustes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and shift taking a onto b, least squares, no reflection: b ≈ a @ R.T + t."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    rot = u @ np.diag([1.0, d]) @ vt
    return rot, cb - ca @ rot.T


# ---- photos ----

REPORT_SCHEMA = "spline-wire/test-report@1"


def target_side(target: str) -> str | None:
    """The object's side the target implies: a disc is inside the chain's
    bend, a hollow outside it; None for a plaque (no object) or a gauge either
    side of which the chain may lie (the S, the straight edge)."""
    gauge = GAUGES.get(target)
    if gauge is None or gauge.concave is None:
        return None
    return "outside" if gauge.concave else "inside"


def score_photo(data: bytes, name: str, target: str, spec: ChainSpec, *,
                saved_focal_35mm: float | None = None, focal_override_35mm: float | None = None,
                fit_focal: bool = False) -> dict:
    """Measure a test photo and score it against its target, with the focal
    length the page would use (the override, else EXIF, else the saved
    setting), estimated from the chain, and (fit_focal) the one that fits the
    target best.

    {"photo", "target", "pins_found", "tilt_deg", "warnings",
     "focal": {"exif": {...}, "estimated": {...}, "fitted": {...}}}
    with each focal entry {"source", "focal_35mm", "score"} (or {"error"}).
    """
    from splinewire.camera import focal_35mm_from_px
    from splinewire.session import Options, PhotoSession

    session = PhotoSession.open(data, name)
    side = target_side(target) or "inside"
    exif_px = session.exif_focal_px

    def run(override: float | None, use_exif: bool) -> tuple[dict, object]:
        session.exif_focal_px = exif_px if use_exif else None
        opts = Options(spec=spec, side=side, focal_override_35mm=override,
                       saved_focal_35mm=saved_focal_35mm if use_exif else None)
        r = session.run(opts)
        m = r.measurement
        sign = m.object_sign if target_side(target) else None
        s = score(target, m.pins_mm, spec, sign=sign, gaps=m.order.gaps)
        f35 = focal_35mm_from_px(m.rectification.focal_px, session.size)
        return {"source": r.focal_source, "focal_35mm": round(f35, 2), "score": s}, r

    out: dict = {"photo": name, "target": target, "focal": {}}
    try:
        entry, r = run(focal_override_35mm, True)
    except ValueError as err:
        return {**out, "error": str(err)}
    m = r.measurement
    out.update(pins_found=len(m.order.indices), tilt_deg=round(m.rectification.tilt_deg, 1),
               warnings=list(m.warnings))
    out["focal"]["exif"] = entry
    if entry["source"] != "estimated":
        out["focal"]["estimated"] = _safely(lambda: run(None, False)[0])
    if fit_focal:
        out["focal"]["fitted"] = _safely(lambda: _fit_focal(run, entry["focal_35mm"]))
    session.exif_focal_px = exif_px
    return out


def headline(s: dict) -> float:
    """The one number a score is judged by: a plaque's worst pin, a gauge's
    worst contact point without the ends."""
    return s["max_error_mm"] if s["kind"] == "plaque" else s["inner"]["max_mm"]


def _fit_focal(run, start_35mm: float) -> dict:
    from scipy.optimize import minimize_scalar

    def cost(f35: float) -> float:
        try:
            s = run(f35, True)[0]["score"]
        except ValueError:
            return 1e3
        return s["rms_error_mm"] if s["kind"] == "plaque" else s["inner"]["rms_mm"]

    best = minimize_scalar(cost, bounds=(0.6 * start_35mm, 1.6 * start_35mm), method="bounded",
                           options={"xatol": 0.01})
    entry = run(float(best.x), True)[0]
    entry["source"] = "fitted"
    return entry


def _safely(fn) -> dict:
    try:
        return fn()
    except ValueError as err:
        return {"error": str(err)}
