"""Print-in-place measuring chain with working joints.

The plaque (plaque.py) is a chain frozen in one pose. This is the real
thing: a chain matching data/chain.yaml that comes off the printer
assembled, bends at every pin, holds its pose by friction, and keeps every
link exactly one pitch long. Design notes: docs/printed-chain.md.

- Links alternate like a bicycle chain. Outer links are a top plate with a
  pin hanging from each end; every dot is on an outer link, centred on its
  pin. Inner links are a bottom plate with a socket at each end. Where a
  link would print on top of another, the surfaces between them are 45°
  cones, so nothing needs support and no flat surface prints onto a gap.
- Clearance would let each pin wander in its socket by up to 0.3 mm. The
  deskew relies on every link being exactly one pitch long, and in
  simulation that much play costs ~0.25 mm worst pin, against 0.1 mm for
  the first real photo of the plaque. A loose joint would not hold a pose
  either. So each socket holds its pin in a rigid 90° V, pressed
  there by a spring beam in the inner link. The V faces across the link,
  so the pitch does not depend on the V's printed size to first order.
- As printed, the V and the spring face a thin neck on the pin, with
  clearance. After printing, each inner link is pressed once toward the
  outer plates (about 1.5 mm, until it stops). That pulls the pin's barrel
  into the V and bends the spring by `preload_mm`, which takes up the
  play and gives the joint its friction.
- Colours by layer, for one extruder: black, then white from the inner
  links' top up to just under the top, then black again. The dot is a
  window through the top black layers, as on the plaque, and every
  surface seen from above is black.

Needs shapely, trimesh and manifold3d (the dev dependency group).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec

LAYER_MM = 0.2
E_PLA_MPA = 3500.0
MU_PLA = 0.35


@dataclass(frozen=True)
class JointParams:
    clearance_mm: float = 0.3     # gap between parts as printed (normal to the surfaces)
    preload_mm: float = 0.15      # how far the barrel bends the spring once set
    barrel_mm: float = 2.1        # radius of the pin where the V and spring hold it
    land_mm: float = 0.6          # height of the V and the spring's contact
    flange_mm: float = 0.8        # how far the pin's foot reaches past its neck
    plate_mm: float = 0.8         # outer link top plate
    black_mm: float = 0.4         # black on top of the plate: the dot windows' depth
    shoulder_mm: float = 0.3      # flat stop on the pin that ends the set travel
    beam_mm: float = 0.8          # spring beam thickness (two 0.4 mm lines)
    slot_mm: float = 0.35         # gaps either side of the spring beam
    top_gap_mm: float = 0.2       # inner link to outer plate once set
    max_turn_deg: float = 90.0    # bend at a joint before links touch
    segments: int = 96            # facets per circle


DEFAULT_PARAMS = JointParams()


@dataclass(frozen=True)
class Joint:
    """Radii and heights (mm, z up from the bed) shared by every joint."""
    p: JointParams
    pitch: float
    R: float          # link half-width: plate, socket and link ends
    r_flange: float
    r_neck: float
    r_barrel: float
    chamfer: float    # height of the 45° chamfers above and below the land
    z_flange: float   # top of the flange's cylinder
    z_neck: float     # flange cone meets the neck
    z_land: float     # bottom of the land (V and spring contact)
    z_land_top: float
    z_ramp: float     # neck starts widening to the barrel
    z_barrel: float   # barrel starts
    z_shoulder: float  # barrel ends: flat stop, then the cap cone
    z_plate: float    # underside of the outer plate
    top: float
    inner_top: float  # top of the inner links as printed
    travel: float     # set travel that seats the barrel in the land
    full_travel: float  # until the shoulder stops it

    @property
    def white_from(self) -> float:
        """First height printed in white (black below)."""
        return self.inner_top

    @property
    def black_from(self) -> float:
        """Back to black from here to the top."""
        return round(self.top - self.p.black_mm, 6)


def joint_geometry(spec: ChainSpec, p: JointParams = DEFAULT_PARAMS) -> Joint:
    c, d = p.clearance_mm, p.preload_mm
    R, r_b = spec.half_width_mm, p.barrel_mm
    r_n = r_b - d - c                       # neck: clears the spring as printed
    r_f = r_n + p.flange_mm
    chamfer = c + d
    z_flange = 2 * LAYER_MM
    z_neck = z_flange + (r_f - r_n)
    # The land sits a 45° chamfer plus a clearance above the flange's cone.
    z_land = _snap_up(z_neck + chamfer + c * math.sqrt(2))
    z_land_top = z_land + p.land_mm
    z_ramp = z_land_top + c * (math.sqrt(2) - 1) + 0.03
    z_barrel = z_ramp + (r_b - r_n)
    travel = z_barrel - z_land               # barrel reaches the land
    full_travel = travel + c
    # Inner link top at the hole: above the land's upper chamfer as printed
    # and up against the shoulder after the full travel.
    z_shoulder = _snap_up(z_land_top + chamfer + LAYER_MM + full_travel)
    z_plate = z_shoulder + (R - r_b - p.shoulder_mm)
    top = z_plate + p.plate_mm
    inner_top = _snap_down(z_plate - full_travel - p.top_gap_mm)
    j = Joint(p=p, pitch=spec.pitch_mm, R=R, r_flange=r_f, r_neck=r_n, r_barrel=r_b, chamfer=chamfer,
              z_flange=z_flange, z_neck=z_neck, z_land=z_land, z_land_top=z_land_top, z_ramp=z_ramp,
              z_barrel=z_barrel, z_shoulder=z_shoulder, z_plate=z_plate, top=top,
              inner_top=inner_top, travel=travel, full_travel=full_travel)
    _check(j, spec)
    return j


def _snap_up(z: float) -> float:
    return round(math.ceil(z / LAYER_MM - 1e-6) * LAYER_MM, 6)


def _snap_down(z: float) -> float:
    return round(math.floor(z / LAYER_MM + 1e-6) * LAYER_MM, 6)


def _check(j: Joint, spec: ChainSpec) -> None:
    p = j.p
    beam_out = _beam_y(j)[1] + p.slot_mm
    if j.R - beam_out < 0.75:
        raise ValueError(f"half_width_mm {j.R} leaves {j.R - beam_out:.2f} mm of wall outside the spring "
                         f"beam; need 0.75 (reduce barrel_mm)")
    if spec.pitch_mm < 2 * j.R + 1.0:
        raise ValueError("pitch_mm must exceed 2 * half_width_mm + 1 for the outer links' ribs")
    if j.white_from >= j.black_from - LAYER_MM:
        raise ValueError("no white left under the dots; plate_mm is too thin")


# ---------------------------------------------------------------------------
# Profiles (r, z) of the pin and of the socket cavity around it

def pin_profile(j: Joint, plate_r: float | None = None):
    """Outer link's pin, cap cone and plate disc, as a polygon in (r, z)."""
    from shapely.geometry import Polygon

    s = j.r_barrel + j.p.shoulder_mm
    r = j.R if plate_r is None else plate_r
    return Polygon([
        (0, 0), (j.r_flange, 0), (j.r_flange, j.z_flange), (j.r_neck, j.z_neck),
        (j.r_neck, j.z_ramp), (j.r_barrel, j.z_barrel), (j.r_barrel, j.z_shoulder),
        (s, j.z_shoulder), (r, j.z_plate - (j.R - r)), (r, j.top), (0, j.top),
    ])


def cavity_profile(j: Joint):
    """Socket cavity in (r, z): the pin swept down by the set travel (the
    inner link moves up past it), grown by the clearance."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    pin = pin_profile(j)
    v = (0.0, -j.travel)
    coords = list(pin.exterior.coords)
    quads = [Polygon([a, b, (b[0] + v[0], b[1] + v[1]), (a[0] + v[0], a[1] + v[1])])
             for a, b in itertools.pairwise(coords)]
    shifted = Polygon([(x + v[0], y + v[1]) for x, y in coords])
    swept = unary_union([pin, shifted] + [q.buffer(0) for q in quads if q.area > 1e-9])
    grown = swept.buffer(j.p.clearance_mm, quad_segs=8)
    return grown.intersection(box(0, -10, j.R + 5, j.top + 10)).simplify(0.002)


# ---------------------------------------------------------------------------
# Solids (manifold3d), built in each link's own frame

def _cs(geom):
    import manifold3d as m
    from shapely.geometry.polygon import orient

    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    rings = []
    for poly in polys:
        poly = orient(poly, 1.0)
        rings.append(np.asarray(poly.exterior.coords)[:-1])
        rings.extend(np.asarray(r.coords)[:-1] for r in poly.interiors)
    return m.CrossSection(rings, m.FillRule.EvenOdd)


def _extrude(geom, z0: float, z1: float):
    return _cs(geom).extrude(z1 - z0).translate((0, 0, z0))


def _revolve(profile, segments: int, at=(0.0, 0.0)):
    return _cs(profile).revolve(segments).translate((at[0], at[1], 0))


def _stadium(a, b, r: float, segments: int):
    from shapely.geometry import LineString, Point

    if np.allclose(a, b):
        return Point(a).buffer(r, quad_segs=segments // 4)
    return LineString([a, b]).buffer(r, quad_segs=segments // 4)


def _hull(points):
    import manifold3d as m

    return m.Manifold.hull_points(np.asarray(points, dtype=float))


def _beam_y(j: Joint) -> tuple[float, float]:
    """The spring beam's inner and outer faces, in its socket's frame."""
    y0 = j.r_barrel - j.p.preload_mm
    return y0, y0 + j.p.beam_mm


def _window(spec: ChainSpec, segments: int, at):
    from shapely.geometry import Point

    c = Point(at)
    w = c.buffer(spec.fiducial_mm / 2, quad_segs=segments // 4)
    if spec.fiducial == "ring":
        w = w.difference(c.buffer(spec.ring_inner_mm / 2, quad_segs=segments // 4))
    return w


def outer_link(j: Joint, spec: ChainSpec, sockets: tuple[bool, bool], n_pins: int = 2):
    """Outer link with pins at (0, 0) and (pitch, 0): plate, pins, rib, dots.

    sockets says which pins sit in an inner link's socket; a pin that does
    not (a chain end) is filled solid to the full link outline. n_pins=1
    makes the end button, a lone pin for an inner link's last socket.
    """
    seg = j.p.segments
    pins = [(0.0, 0.0), (j.pitch, 0.0)][:n_pins]
    body = _extrude(_stadium(pins[0], pins[-1], j.R, seg), j.z_plate, j.top)
    # The pins' plate discs stop just inside the plate's edge, so no two
    # surfaces of the union coincide.
    profile = pin_profile(j, plate_r=j.R - 0.05)
    for at in pins:
        body += _revolve(profile, seg, at)
    if n_pins == 2:
        body += _extrude(_rib(j, sockets), 0.0, j.z_plate)
    for at in pins:
        body -= _extrude(_window(spec, seg, at), j.black_from, j.top + 1.0)
    return body


def _rib(j: Joint, sockets: tuple[bool, bool]):
    """Web under an outer link's plate, holding it up while printing, cut
    back so the inner links can turn by max_turn_deg. At a chain end it
    fills the link's rounded end to the full outline."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    c, P, R, seg = j.p.clearance_mm, j.pitch, j.R, j.p.segments
    rib = _stadium((0, 0), (P, 0), R, seg)
    swept = []
    for in_socket, at, away in zip(sockets, ((0.0, 0.0), (P, 0.0)), (math.pi, 0.0)):
        if in_socket:
            rib = rib.difference(Point(at).buffer(R + c, quad_segs=seg // 4))
            for t in np.radians(np.linspace(-j.p.max_turn_deg, j.p.max_turn_deg, 91)):
                end = (at[0] + P * math.cos(away + t), at[1] + P * math.sin(away + t))
                swept.append(_stadium(at, end, R + c, 16))
    if swept:
        rib = rib.difference(unary_union(swept))
    if rib.is_empty or rib.area < 1.0:
        raise ValueError("no room for the outer links' rib: lower max_turn_deg or raise the pitch")
    return rib


def inner_link(j: Joint):
    """Inner link with sockets at (0, 0) and (pitch, 0)."""
    import manifold3d as m

    seg, P = j.p.segments, j.pitch
    outline = _extrude(_stadium((0, 0), (P, 0), j.R, seg), 0.0, j.inner_top)
    body = outline
    cavity = _revolve(cavity_profile(j), seg)
    socket = _socket_parts(j)

    def flip(solid):   # the same, for the socket at the far end
        return solid.rotate((0, 0, 180)).translate((P, 0, 0))

    for piece in (cavity, socket["slots"]):
        body = body - piece - flip(piece)
    grip = socket["grip"]
    body = body + ((grip + flip(grip)) ^ outline)
    assert body.status() == m.Error.NoError
    return body


def _socket_parts(j: Joint) -> dict:
    """Spring beam slots and the land (V flanks and spring ridge) for the
    socket at the origin. The spring is on +y and pushes the pin into the
    V on -y; the socket at the far end is the same turned 180°."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    p = j.p
    c, d, sl = p.clearance_mm, p.preload_mm, p.slot_mm
    y0, y1 = _beam_y(j)
    tip = -0.5                       # beam tip, just past the contact at x = 0
    anchor = j.pitch / 2
    slots = unary_union([
        box(tip - sl, y1, anchor, y1 + sl),           # outside the beam
        box(tip - sl, y0 - sl, anchor, y0),           # inside it
        box(tip - sl, y0 - sl, tip, y1 + sl),         # round its tip
    ])
    # The land starts a hair below z_land, where the cavity's wall bends:
    # edges of the two must not coincide.
    z = [j.z_land - j.chamfer, j.z_land - 0.02, j.z_land_top, j.z_land_top + j.chamfer]
    # How far each level is from full intrusion; the chamfers end a little
    # inside the cavity wall, again so no surfaces coincide.
    grow = [1.15, 0.0, 0.0, 1.15]
    # Spring ridge: the beam's face reaches in to y0 on the land. It ends
    # inside the cavity (r_barrel + clearance) so it does not run on along
    # the beam's own face.
    end = 0.95 * math.sqrt((j.r_barrel + c) ** 2 - y0 ** 2)
    ridge = [(x, y0 + g * (c + d), zz) for zz, g in zip(z, grow) for x in (tip, end)]
    ridge += [(x, y1, zz) for zz in z for x in (tip, end)]
    parts = [_hull(ridge)]
    # V flanks at -45° and -135°, tangent to the barrel.
    for ang in (-math.pi / 4, -3 * math.pi / 4):
        u = np.array([math.cos(ang), math.sin(ang)])
        t = np.array([-u[1], u[0]])
        pts = []
        for zz, g in zip(z, grow):
            rho = j.r_barrel + g * c
            for a in (-1.6, 1.6):
                for b in (0.0, 1.0):
                    q = (rho + b) * u + a * t
                    pts.append((q[0], q[1], zz))
        parts.append(_hull(pts))
    grip = parts[0] + parts[1] + parts[2]
    return {"slots": _extrude(slots, -1.0, j.top + 1.0), "grip": grip}


# ---------------------------------------------------------------------------
# The whole chain

@dataclass(frozen=True)
class Body:
    kind: str           # "outer", "inner" or "button"
    pins: tuple[int, ...]   # chain pin indices it carries (outer, button) or holds (inner)
    solid: object       # manifold3d.Manifold, in the chain frame as printed


def chain_bodies(spec: ChainSpec, p: JointParams = DEFAULT_PARAMS, n_pins: int | None = None) -> list[Body]:
    """Every part of the chain as printed: straight along +x, pin i at
    (i * pitch, 0). The first link is an outer one and they alternate from
    there; an end button when the last link is an inner one."""
    j = joint_geometry(spec, p)
    n = spec.n_pins if n_pins is None else n_pins
    if n < 2:
        raise ValueError("need at least 2 pins")
    P = j.pitch
    cache: dict = {}
    bodies = []
    for k in range(n - 1):
        x = k * P
        if k % 2 == 0:
            sockets = (k > 0, k + 1 < n - 1)
            if sockets not in cache:
                cache[sockets] = outer_link(j, spec, sockets)
            bodies.append(Body("outer", (k, k + 1), cache[sockets].translate((x, 0, 0))))
        else:
            if "inner" not in cache:
                cache["inner"] = inner_link(j)
            bodies.append(Body("inner", (k, k + 1), cache["inner"].translate((x, 0, 0))))
    if (n - 2) % 2 == 1:
        button = outer_link(j, spec, (True,), n_pins=1)
        bodies.append(Body("button", (n - 1,), button.translate(((n - 1) * P, 0, 0))))
    return bodies


def set_pose(bodies: list[Body], j: Joint, turns_deg, travel: float | None = None) -> list:
    """Solids of the set chain bent by turns_deg (one per joint, pins 1..n-2):
    inner links raised by the travel, every link turned about its pins."""
    travel = j.full_travel if travel is None else travel
    turns = np.radians(np.asarray(turns_deg, dtype=float))
    n = max(max(b.pins) for b in bodies) + 1
    if len(turns) != n - 2:
        raise ValueError(f"need {n - 2} turns, got {len(turns)}")
    heading = np.concatenate([[0.0], np.cumsum(turns)])
    pins = np.vstack([[0.0, 0.0], np.cumsum(j.pitch * np.c_[np.cos(heading), np.sin(heading)], axis=0)])
    out = []
    for b in bodies:
        first = b.pins[0]
        h = heading[first] if first < len(heading) else heading[-1]
        s = b.solid.translate((-first * j.pitch, 0, travel if b.kind == "inner" else 0.0))
        out.append(s.rotate((0, 0, math.degrees(h))).translate((pins[first][0], pins[first][1], 0)))
    return out


def to_trimesh(solid):
    import trimesh

    mesh = solid.to_mesh()
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3], faces=np.asarray(mesh.tri_verts))
    out.merge_vertices(digits_vertex=6)
    return out


def colour_split(solid, j: Joint) -> dict:
    """The solid cut at the colour changes: "black" (bottom and top) and "white"."""
    lower, rest = solid.trim_by_plane((0, 0, -1), -j.white_from), solid.trim_by_plane((0, 0, 1), j.white_from)
    white, top = rest.trim_by_plane((0, 0, -1), -j.black_from), rest.trim_by_plane((0, 0, 1), j.black_from)
    return {"black": lower + top, "white": white}


def spring_estimate(j: Joint) -> dict:
    """Rough numbers for the spring: a cantilever from the inner link's
    middle to the pin, as tall as the inner link."""
    t, L, h = j.p.beam_mm, j.pitch / 2, j.inner_top
    k = 3 * E_PLA_MPA * (h * t ** 3 / 12) / L ** 3          # N/mm
    force = k * j.p.preload_mm
    stress = 6 * force * L / (h * t ** 2)
    # The spring and the V's two flanks press on the barrel: F + 2 F/sqrt(2).
    torque = MU_PLA * (1 + math.sqrt(2)) * force * j.r_barrel
    return {"force_N": force, "stress_MPa": stress, "torque_Nmm": torque}


def write_printed_chain(out_dir: Path, spec: ChainSpec, p: JointParams = DEFAULT_PARAMS,
                        n_pins: int | None = None, name: str = "chain") -> dict[str, Path]:
    import manifold3d as m

    j = joint_geometry(spec, p)
    bodies = chain_bodies(spec, p, n_pins)
    whole = m.Manifold.batch_boolean([b.solid for b in bodies], m.OpType.Add)
    split = colour_split(whole, j)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": out_dir / f"{name}.stl",
        "black": out_dir / f"{name}-black.stl",
        "white": out_dir / f"{name}-white.stl",
        "instructions": out_dir / f"{name}-PRINTING.txt",
        "drawing": out_dir / f"{name}-joint.svg",
    }
    to_trimesh(whole).export(paths["combined"])
    to_trimesh(split["black"]).export(paths["black"])
    to_trimesh(split["white"]).export(paths["white"])
    n = len({i for b in bodies for i in b.pins})
    paths["instructions"].write_text(print_instructions(name, j, bodies, n), encoding="utf-8")
    paths["drawing"].write_text(joint_svg(spec, p), encoding="utf-8")
    return paths


def print_instructions(name: str, j: Joint, bodies: list[Body], n_pins: int) -> str:
    n_inner = sum(b.kind == "inner" for b in bodies)
    length = (n_pins - 1) * j.pitch + 2 * j.R
    sp = spring_estimate(j)
    button = any(b.kind == "button" for b in bodies)
    return f"""Spline Wire print-in-place chain: {name}
{n_pins} pins, pitch {j.pitch:g} mm, {length:.1f} x {2 * j.R:.1f} x {j.top:.1f} mm as printed.
Comes off the printer assembled; no supports.

Single extruder, two filament changes (matte black + white PLA):
  1. Load {name}.stl. Print it flat as it comes, dots up. 0.2 mm layers,
     0.2 mm first layer. No supports, no ironing, no brim.
  2. Start with BLACK.
  3. Change to WHITE at the first layer above {j.white_from:.1f} mm
     (Z = {j.white_from + LAYER_MM:.1f} mm with 0.2 mm layers).
  4. Change back to BLACK at the first layer above {j.black_from:.1f} mm
     (Z = {j.black_from + LAYER_MM:.1f} mm).
     PrusaSlicer/Orca/Bambu Studio: right-click the "+" on the layer slider
     at each height -> Add color change. Cura: "Filament Change"
     post-processing script, once per height.
  5. Matte black if you can: a lamp reflected in shiny black washes the
     dots out (docs/cv-robustness.md).
  6. The joints are {j.p.clearance_mm:.2f} mm apart:
     - Elephant-foot compensation on (0.1-0.2 mm): the pins' feet sit
       {j.p.clearance_mm:.2f} mm from the inner links on the first layer.
     - XY size/hole compensation off: the pitch is the dimension that matters.
     - 3+ perimeters, 100% infill or close to it (the parts are small).
Multi-material printer instead: load {name}-black.stl and {name}-white.stl
together as one object with two parts, and assign the colours.

After printing:
  1. Work every joint gently left and right a few times until it turns freely.
  2. Set the joints. Lay the chain dots DOWN on a flat, hard table. From the
     back, the {n_inner} inner links are the long plain links; the outer links
     show a round foot at each pin{" (and the last pin is a loose round button)" if button else ""}.
     Press the middle of each inner link firmly toward the table until it
     stops: it moves about {j.full_travel:.1f} mm. That pulls each pin into its V and bends its spring {j.p.preload_mm:.2f} mm, which
     removes the play and gives the joint its friction.
  3. Turn the chain over: it rests on the outer links, and the inner links
     now sit {j.full_travel:.1f} mm higher than before, still below the outer
     links' top. Pressing on the dots or the outer links can't undo this.
  4. Check: every joint turns with even friction and has no play you can feel
     when you push the links together or pull them apart. A loose joint was
     not pressed home: press that inner link again.
  5. Pull the chain straight along a ruler: from end to end it is {length:.1f} mm
     long ({n_pins - 1} x {j.pitch:g} mm pitch + 2 x {j.R:g} mm). More than ~0.3 mm off
     means the printer's XY scale is off; measure the actual pitch and put
     it in chain.yaml.

Estimated, for PLA: spring force {sp["force_N"]:.1f} N, friction torque
{sp["torque_Nmm"]:.1f} N mm per joint (bending stress {sp["stress_MPa"]:.0f} MPa). Too stiff or too
loose: regenerate with a different --preload (default {JointParams().preload_mm:g} mm). Joints that
fuse or bind: regenerate with a larger --clearance (default {JointParams().clearance_mm:g} mm).
Try a short piece first: splinewire print-chain --pins 3.
"""


# ---------------------------------------------------------------------------
# Drawing of one joint, for the docs and the output folder

def joint_svg(spec: ChainSpec, p: JointParams = DEFAULT_PARAMS) -> str:
    """Sections through one joint: across the chain at the pin, as printed
    and once set, and a top view at the land."""
    from shapely.geometry import box

    j = joint_geometry(spec, p)
    bodies = chain_bodies(spec, p, n_pins=3)
    x0 = j.pitch                                   # the joint at pin 1
    printed = [b.solid for b in bodies]
    seated = set_pose(bodies, j, [0.0])
    zones = [(0.0, j.white_from, "#3a3a3a"), (j.white_from, j.black_from, "#f4f1ea"),
             (j.black_from, j.top + 5, "#3a3a3a")]
    stroke = {"outer": "#2f6db5", "inner": "#d0782a", "button": "#2f6db5"}
    scale, pad = 18.0, 14.0
    panels, x_at = [], pad

    def across(solids, kinds, title, lift):
        nonlocal x_at
        w, h = 2 * j.R + 2, j.top + j.full_travel + 1
        g = [f'<g transform="translate({x_at:.1f},{pad + 16:.1f})">',
             f'<text x="0" y="-6" class="t">{title}</text>']
        for solid, kind in zip(solids, kinds):
            cut = solid.translate((-x0, 0, 0)).rotate((0, 90, 0)).slice(0.0)
            geom = _polys_to_shapely(cut.to_polygons(), swap=True)
            if geom.is_empty:
                continue
            z_off = lift if kind == "inner" else 0.0
            for z0, z1, fill in zones:
                part = geom.intersection(box(-50, z0 + z_off, 50, z1 + z_off))
                g += _svg_paths(part, fill, "none", scale, w / 2, h)
            g += _svg_paths(geom, "none", stroke[kind], scale, w / 2, h)
        g.append("</g>")
        x_at += w * scale + pad
        panels.extend(g)

    kinds = [b.kind for b in bodies]
    across(printed, kinds, "As printed, section at a pin", 0.0)
    across(seated, kinds, f"Set: inner link up {j.full_travel:.1f} mm", j.full_travel)
    # Top view at the land, set.
    zl = (j.z_land + j.z_land_top) / 2
    w = 2 * j.R + 2
    g = [f'<g transform="translate({x_at:.1f},{pad + 16:.1f})">',
         '<text x="0" y="-6" class="t">Set: top view at the V</text>']
    for solid, kind in zip(seated, kinds):
        cut = solid.slice(zl + j.full_travel)
        geom = _polys_to_shapely(cut.to_polygons()).intersection(box(x0 - w / 2, -w / 2, x0 + w / 2, w / 2))
        moved = _shift(geom, -x0 + w / 2, w / 2)
        g += _svg_paths(moved, "#3a3a3a" if kind == "inner" else "#8a8a8a", stroke[kind], scale, 0, w,
                        flip=True)
    g.append("</g>")
    x_at += w * scale + pad
    panels.extend(g)
    height = (j.top + j.full_travel + 1) * scale + 2 * pad + 16
    legend = (("Blue outline: outer link (pin, dot). Orange: inner link (socket; "
               "V on one side, spring beam on the other)."),
              "Fill: filament colour. Dimensions: docs/printed-chain.md.")
    for i, line in enumerate(legend):
        panels.append(f'<text x="{pad}" y="{height + 4 + 16 * i:.0f}" class="t">{line}</text>')
    height += 32
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{x_at:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {x_at:.0f} {height:.0f}">\n'
            '<style>.t{font:12px sans-serif;fill:#333}</style>\n'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(panels) + "\n</svg>\n")


def _polys_to_shapely(contours, swap=False):
    from shapely.geometry import Polygon

    polys = []
    for c in contours:
        c = np.asarray(c)
        if len(c) < 3:
            continue
        if swap:   # slice of a solid turned 90° about y: (x, y) is (z, y) of the original
            c = np.c_[c[:, 1], c[:, 0]]
        polys.append(Polygon(c))
    if not polys:
        return Polygon()
    # Even-odd: holes come out as separate contours.
    out = Polygon()
    for poly in polys:
        out = out.symmetric_difference(poly.buffer(0))
    return out


def _shift(geom, dx, dy):
    from shapely import affinity

    return affinity.translate(geom, dx, dy)


def _svg_paths(geom, fill, stroke, scale, x_mid, height, flip=True):
    if geom.is_empty:
        return []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for poly in polys:
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        d = ""
        for ring in [poly.exterior, *poly.interiors]:
            pts = np.asarray(ring.coords)
            xs = (pts[:, 0] + x_mid) * scale
            ys = (height - pts[:, 1]) * scale if flip else pts[:, 1] * scale
            d += "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + "Z"
        sw = 1.2 if stroke != "none" else 0
        out.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="{stroke}" stroke-width="{sw}"/>')
    return out
