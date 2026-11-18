"""3D-printable contact gauges for the real-world tests (docs/real-world-tests.md).

Each gauge is a flat block whose edge has a known shape (splinewire.targets
GAUGES): a disc or an arc to wrap the chain round, a hollow to press it
into, an S, a straight edge. The chain lies on the table against the edge,
and its sleeves touch it.

- 6 mm tall: the snap-fit chain's sleeves are widest from 0.45 to 2.15 mm
  above the table and again at 7.1 mm, so the edge meets them low down,
  where a real object low on the table would.
- The bottom 0.6 mm is set in by 0.4 mm all round, so the first layers'
  squash (elephant's foot) does not stick out past the edge.
- Any one colour. Printed outer edges usually come out a little large and
  holes a little small: measure the discs and the hole with calipers
  (PRINTING.txt) and read the scores' radius against that.

Needs shapely and trimesh (the dev dependency group).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from splinewire.targets import GAUGES, Arc, Gauge, Line

HEIGHT_MM = 6.0
FOOT_MM = 0.6               # height of the set-in bottom
FOOT_INSET_MM = 0.4
WALL_MM = 15.0              # thickness behind an edge that is not a disc
_STEP_MM = 0.2              # polygon spacing along the edge


def gauge_polygon(gauge: Gauge):
    """The gauge's outline seen from above (a shapely Polygon), in the
    gauge's own coordinates, with its edge where targets.GAUGES puts it."""
    from shapely.geometry import LineString, Point, Polygon, box

    def arc_points(arc: Arc, radius: float | None = None) -> np.ndarray:
        r = radius or arc.radius
        n = max(8, int(math.ceil(arc.length / _STEP_MM)))
        a = np.radians(arc.start_deg) + np.radians(arc.sweep_deg) * np.linspace(0, 1, n + 1)
        return np.c_[arc.center[0] + r * np.cos(a), arc.center[1] + r * np.sin(a)]

    seg = gauge.segments[0]
    if gauge.radius_mm is not None:
        r = gauge.radius_mm
        if gauge.closed and not gauge.concave:      # a ring: the disc's edge without its middle
            disc = Point(0, 0).buffer(r, quad_segs=int(math.ceil(math.pi * r / 2 / _STEP_MM)))
            return disc.difference(Point(0, 0).buffer(r - WALL_MM, quad_segs=64)) if r > WALL_MM + 5 else disc
        if gauge.closed:
            disc = Point(0, 0).buffer(r, quad_segs=int(math.ceil(math.pi * r / 2 / _STEP_MM)))
            return box(-r - WALL_MM, -r - WALL_MM, r + WALL_MM, r + WALL_MM).difference(disc)
        inner = arc_points(seg)
        if not gauge.concave:                       # a segment of a disc: the arc and its chord
            return Polygon(inner)
        outer = arc_points(seg, r + WALL_MM)[::-1]  # a band behind the hollow
        return Polygon(np.vstack([inner, outer]))
    # An open edge (the S, the straight edge): a band on the edge's right,
    # leaving its left free for the chain (targets.chain_on_gauge side +1).
    edge = gauge.edge(_STEP_MM)
    return LineString(edge).buffer(-WALL_MM, single_sided=True, quad_segs=16)


def gauge_mesh(gauge: Gauge):
    """The gauge as one trimesh: the set-in foot under the full outline."""
    import trimesh

    poly = gauge_polygon(gauge)
    foot = poly.buffer(-FOOT_INSET_MM, join_style="mitre")
    top = trimesh.creation.extrude_polygon(poly, HEIGHT_MM - FOOT_MM)
    top.apply_translation([0, 0, FOOT_MM])
    return trimesh.util.concatenate([trimesh.creation.extrude_polygon(foot, FOOT_MM), top])


def chain_side(gauge: Gauge) -> int:
    """Which side of the gauge's edge the chain lies on (targets.chain_on_gauge)."""
    if gauge.concave is None:
        return 1
    return 1 if gauge.concave else -1


def write_gauges(out: Path, names: list[str] | None = None) -> dict[str, Path]:
    """STLs of the gauges, their truth (gauges.json) and printing notes."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [GAUGES[n] for n in (names or GAUGES)]
    paths: dict[str, Path] = {}
    for g in chosen:
        path = out / f"gauge-{g.name}.stl"
        gauge_mesh(g).export(path)
        paths[g.name] = path
    truth = out / "gauges.json"
    truth.write_text(json.dumps({"gauges": [g.to_json() for g in chosen]}, indent=1) + "\n", encoding="utf-8")
    paths["truth"] = truth
    notes = out / "PRINTING.txt"
    notes.write_text(printing_notes(chosen), encoding="utf-8")
    paths["instructions"] = notes
    return paths


def printing_notes(gauges: list[Gauge]) -> str:
    lines = [
        "Contact gauges for the real-world tests (docs/real-world-tests.md)",
        "",
        f"Any one colour; PLA is fine. {HEIGHT_MM:g} mm tall, flat on the bed, no supports.",
        "0.2 mm layers, 3 or more perimeters, so the edge is solid and smooth.",
        "Write each gauge's name on it after printing.",
        "",
        "Before testing, measure with calipers and note the numbers:",
    ]
    for g in gauges:
        if g.radius_mm is not None and g.closed:
            what = "outside diameter" if not g.concave else "hole diameter"
            lines.append(f"  {g.name}: {what}, should be {2 * g.radius_mm:.2f} mm "
                         "(several places; the difference is the print's edge offset)")
    lines += [
        "",
        "Using a gauge: lay it flat, wrap or press the chain against its edge with the rims up,",
        "with every link touching and neither end hanging past the edge.",
    ]
    for g in gauges:
        seg = g.segments[0]
        size = (f"edge {g.length:.0f} mm" if isinstance(seg, Line) or g.radius_mm is None
                else f"R{g.radius_mm:g}, {abs(seg.sweep_deg):g}°")
        lines.append(f"  gauge-{g.name}.stl  {g.label}  ({size})")
    return "\n".join(lines) + "\n"
