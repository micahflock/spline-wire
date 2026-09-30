"""3D-printable test plaque: a chain shape with exactly known pin positions.

Designed for one manual filament swap with black and white filament:

- White base, printed first, trimmed to the chain's outline plus a narrow
  rim, with a short bridge to the scale bar. A full rectangular plate used
  about three times the material for nothing: only the white under the
  ring windows is ever measured. White PLA is translucent, so the base is
  thick enough (8 layers) to look white on any table.
- One swap to black at z = plate thickness.
- A thin black layer shaped like the chain (8 mm wide links with round
  ends), with a ring-shaped window over every pin that shows the white
  plate through it. Black is opaque, so two layers are enough, which
  keeps the window walls shallow (see experiments/relief_bias.py).
- A black 50.0 mm bar to check the printer's XY scale with calipers.

Needs shapely and trimesh (the dev dependency group).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec

PLATE_MM = 1.6          # white base; a multiple of 0.2 and 0.16 mm layers
BLACK_MM = 0.4          # black pattern on top (two 0.2 mm layers)
RIM_MM = 1.5            # white rim around the black pattern
SCALE_BAR_MM = (50.0, 2.0)
BAR_GAP_MM = 5.0        # black chain to black scale bar
BRIDGE_MM = 6.0         # width of the white strip joining the bar to the chain
_ARC = 32               # segments per quarter circle


@dataclass(frozen=True)
class Plaque:
    plate: object       # shapely Polygon: plate outline
    pattern: object     # shapely (Multi)Polygon: black areas
    pins_mm: np.ndarray


def plaque_geometry(pins_mm: np.ndarray, spec: ChainSpec) -> Plaque:
    from shapely.geometry import LineString, Point, box
    from shapely.ops import nearest_points, unary_union

    body = LineString(pins_mm).buffer(spec.half_width_mm, quad_segs=_ARC)  # round ends and joints
    windows = unary_union([
        Point(p).buffer(spec.ring_outer_mm / 2, quad_segs=_ARC)
        .difference(Point(p).buffer(spec.ring_inner_mm / 2, quad_segs=_ARC))
        for p in pins_mm
    ])
    # Scale bar under the chain's lowest point, kept within the chain's width.
    x0, y0, x1, _ = body.bounds
    bar_w, bar_h = SCALE_BAR_MM
    lowest = min(body.exterior.coords, key=lambda c: c[1])[0]
    bx0 = float(np.clip(lowest - bar_w / 2, x0, max(x0, x1 - bar_w)))
    bar = box(bx0, y0 - BAR_GAP_MM - bar_h, bx0 + bar_w, y0 - BAR_GAP_MM)
    pattern = unary_union([body.difference(windows), bar])

    bridge = LineString(nearest_points(body, bar)).buffer(BRIDGE_MM / 2, quad_segs=_ARC)
    plate = unary_union([body.buffer(RIM_MM, quad_segs=_ARC), bar.buffer(RIM_MM, quad_segs=_ARC), bridge])
    return Plaque(plate=plate, pattern=pattern, pins_mm=np.asarray(pins_mm, dtype=float))


def plaque_meshes(plaque: Plaque) -> dict[str, object]:
    """trimesh meshes: "white" (plate), "black" (pattern), and "combined".

    The combined mesh is for single-extruder printing with a filament swap:
    its black part reaches 0.2 mm down into the plate so the two shells
    overlap instead of merely touching, which slicers merge cleanly.
    """
    import trimesh

    white = trimesh.creation.extrude_polygon(plaque.plate, PLATE_MM)
    black = _extrude(plaque.pattern, BLACK_MM)
    black.apply_translation([0, 0, PLATE_MM])
    sunk = _extrude(plaque.pattern, BLACK_MM + 0.2)
    sunk.apply_translation([0, 0, PLATE_MM - 0.2])
    return {"white": white, "black": black, "combined": trimesh.util.concatenate([white, sunk])}


def write_plaque(out_dir: Path, name: str, pins_mm: np.ndarray, spec: ChainSpec) -> dict[str, Path]:
    plaque = plaque_geometry(pins_mm, spec)
    meshes = plaque_meshes(plaque)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": out_dir / f"{name}-plaque.stl",
        "white": out_dir / f"{name}-plaque-white.stl",
        "black": out_dir / f"{name}-plaque-black.stl",
        "instructions": out_dir / f"{name}-plaque-PRINTING.txt",
    }
    for key in ("combined", "white", "black"):
        meshes[key].export(paths[key])
    paths["instructions"].write_text(print_instructions(name, plaque), encoding="utf-8")
    return paths


def print_instructions(name: str, plaque: Plaque) -> str:
    x0, y0, x1, y1 = plaque.plate.bounds
    return f"""Spline Wire test plaque: {name}
Size {x1 - x0:.0f} x {y1 - y0:.0f} mm, {PLATE_MM + BLACK_MM:.1f} mm thick, about {plaque_volume_cm3(plaque):.1f} cm3.

Single extruder, manual filament swap (black + white PLA):
  1. Load {name}-plaque.stl. Print flat, pattern side up. 0.2 mm layers
     (0.2 mm first layer too). No ironing: a glossy top causes glare.
  2. Start with WHITE filament.
  3. Add a filament change / pause at the first layer above {PLATE_MM:.1f} mm
     (the layer that starts at {PLATE_MM:.1f} mm, i.e. Z = {PLATE_MM + 0.2:.1f} mm with
     0.2 mm layers). Switch to BLACK there.
     PrusaSlicer/Orca/Bambu Studio: right-click the "+" on the layer
     slider at that height -> Add color change. Cura: "Filament Change"
     post-processing script at that layer.
  4. Use MATTE black filament if you can ("PLA Matte" and the like). A lamp
     reflected in ordinary PLA can wash the black out until the rings
     vanish; in simulated photos matte never failed, standard PLA failed
     in about 1 in 10 setups and glossy/silk in about 1 in 4
     (docs/cv-robustness.md). Avoid silk.
  5. Solid infill (100%), or 4+ top/bottom layers: the part is thin, and
     sparse infill can show through the white under the rings.

Multi-material printer instead: load {name}-plaque-white.stl and
{name}-plaque-black.stl together as one object with two parts, and
assign the colours.

Before testing:
  - Measure the black 50.0 mm bar with calipers. More than ~0.2 mm off
    means the printer's XY scale is off; the app's "after scale fit"
    error removes that effect.
  - Photograph the plaque on a plain, even surface (any colour), whole
    chain in view, and process the photos with the truth file
    {name}-truth.json.
  - Keep lamps out of the reflection: if the black looks grey or shiny on
    the phone screen, move the light or tilt the phone a little until it
    looks black again.
"""


def plaque_volume_cm3(plaque: Plaque) -> float:
    return (plaque.plate.area * PLATE_MM + plaque.pattern.area * BLACK_MM) / 1000.0


def _extrude(geom, height: float):
    import trimesh

    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return trimesh.util.concatenate([trimesh.creation.extrude_polygon(p, height) for p in polys])
