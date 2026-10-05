"""Assembled measuring chain for PETG: printed links, O-rings, M3 screws.

The print-in-place chain (printed_chain.py) needs 0.3 mm gaps that PETG
strings across and fuses. This one prints its links apart and is screwed
together: one M3 countersunk screw, one O-ring and one small steel washer
per joint. It matches data/chain.yaml (pitch, link width, dot); the
software only has to accept sharper bends (max_bend_deg). Design notes:
docs/printed-chain.md.

- Links alternate like a bicycle chain. Outer links: a top plate with the
  dots and a sleeve under each pin. Inner links: a bottom plate with a
  countersunk hole at each end.
- At each joint the screw comes up from below, through the inner link's
  countersink, and cuts its thread into the outer link's sleeve until the
  sleeve's flat end stops on the screw head. The O-ring, squeezed between a
  45° seat under the outer plate and a washer resting on the inner link,
  presses the inner link's countersink down onto the head's cone, which
  centres it without play whatever the hole's printed size. The outer link
  is centred by the screw's thread, squared up by the head it is clamped to.
- No rubber surface slips. The O-ring grips its seat and the washer; the
  joint turns where the steel washer slides on a narrow ring on the inner
  link and where the countersink slides on the screw head. Rubber that
  slipped would first twist and then spring back when let go. Some twist is
  still stored in the O-ring (bounded by the washer's friction), so the
  countersink's friction is kept well above the washer's: then it holds
  the pose against what the O-ring stores.
- A post under the middle of each outer link stops both of its joints
  where the pins either side of a joint are (1 + margin) pitches apart, so
  the ordering never sees a neighbour's neighbour closer than a pitch.
- Everything prints as used, dots up, on one bed and without supports: the
  outer links stand on their sleeves' ends, posts and feet, and grow from
  them at 45° around the parts that move under them. Colours by layer: the
  dots are windows through a black top onto white, as on the plaque, and
  the inner links are lower than the white so they come out black.

All heights are in the chain's frame as used: z up from the table, dots up.
Needs shapely, trimesh and manifold3d (the dev dependency group).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec, pins_from_turns
from splinewire.printed_chain import (
    LAYER_MM,
    _extrude,
    _polys_to_shapely,
    _revolve,
    _stadium,
    _svg_paths,
    _window,
    to_trimesh,
)

MU_STEEL_PETG = 0.25
ORING_N_PER_MM = 0.6    # NBR 70A at 20% squeeze, per mm of circumference (rough mid-range)


@dataclass(frozen=True)
class OringParams:
    margin: float = 0.20          # pins either side of a joint stay >= (1 + margin) pitches apart
    squeeze_mm: float = 0.25      # O-ring pressed into its 45° seat by this much once screwed home
    oring_id_mm: float = 4.0
    oring_cs_mm: float = 1.5
    screw_d_mm: float = 3.0       # M3 countersunk, 90° head (ISO 10642 / DIN 7991 / DIN 965)
    head_d_mm: float = 6.0
    screw_len_mm: float = 6.0     # overall length, head included
    pilot_mm: float = 2.5         # hole the screw cuts its thread into
    sleeve_mm: float = 4.0        # outer link's sleeve
    washer_id_mm: float = 4.3     # ISO 7092 M4 small washer: 4.3 x 8 x 0.5
    washer_od_mm: float = 8.0
    washer_mm: float = 0.5
    inner_mm: float = 3.0         # inner link plate
    ring_mm: float = 0.4          # raised ring the washer slides on
    ring_od_mm: float = 5.5
    hole_gap_mm: float = 0.25     # sleeve to inner link hole, radial
    head_recess_mm: float = 0.2   # screw head below the inner link's underside
    clearance_mm: float = 0.35    # between parts that move past each other
    black_mm: float = 0.4         # black over the dots: the windows' depth
    white_mm: float = 1.2         # white under the windows
    white_over_bore_mm: float = 0.8
    segments: int = 96


DEFAULT_PARAMS = OringParams()


@dataclass(frozen=True)
class Joint:
    """Radii and heights (mm, z up from the table as used) shared by every joint."""
    p: OringParams
    pitch: float
    R: float            # link half-width
    stop_deg: float     # smallest angle between two links at a joint
    r_cs: float         # countersink radius at the inner link's underside
    r_hole: float       # inner link hole
    r_ring: float       # outer edge of the washer's ring
    r_keep: float       # nothing of an outer link within this of a pin below z_seat_top
    z_cs_top: float     # countersink meets the hole
    z_head: float       # screw head's flat face
    z_head_top: float   # head cone meets the shank: the sleeve's end stops here
    z_washer: float     # washer underside (top of the ring)
    z_washer_top: float
    z_seat: float       # the O-ring's 45° seat meets the sleeve
    z_bed: float        # outer links' lowest point: their print bed
    z_clear: float      # outer links spread out above this (inner links pass below)
    z_bore: float       # top of the screw's pilot hole
    z_tip: float
    top: float
    post_r: float       # stop post: a stadium across the link,
    post_a: float       # radius post_r, straight part 2 * post_a long

    @property
    def white_from(self) -> float:
        """First height (as used) printed in white."""
        return round(self.top - self.p.black_mm - self.p.white_mm, 6)

    @property
    def black_from(self) -> float:
        """Back to black from here to the top."""
        return round(self.top - self.p.black_mm, 6)

    @property
    def oring_centre(self) -> tuple[float, float]:
        """(r, z) of the O-ring's cross-section, resting on the washer and the sleeve."""
        a = self.p.oring_cs_mm / 2
        return self.p.oring_id_mm / 2 + a, self.z_washer_top + a


def stop_angle_deg(margin: float) -> float:
    """Smallest angle between two links that keeps the pins either side of
    the joint (1 + margin) pitches apart: 2 p sin(θ/2) = (1 + margin) p."""
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin}")
    return math.degrees(2 * math.asin((1 + margin) / 2))


def joint_geometry(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS) -> Joint:
    P, R = spec.pitch_mm, spec.half_width_mm
    r_cs = p.head_d_mm / 2 + p.head_recess_mm         # 90° countersink: r = r_cs - z
    r_hole = p.sleeve_mm / 2 + p.hole_gap_mm
    z_head_top = r_cs - p.screw_d_mm / 2
    z_washer = p.inner_mm + p.ring_mm
    z_washer_top = z_washer + p.washer_mm
    # 45° seat over the O-ring: z = z_seat + (r - r_sleeve), pressed into the
    # O-ring's cross-section by squeeze_mm along its normal.
    a = p.oring_cs_mm / 2
    rc, zc = p.oring_id_mm / 2 + a, z_washer_top + a
    z_seat = zc - (rc - p.sleeve_mm / 2) + math.sqrt(2) * (a - p.squeeze_mm)
    z_bed = z_head_top
    z_tip = p.head_recess_mm + p.screw_len_mm
    z_bore = z_tip + 0.3
    # The bore ends under the white, with white_over_bore_mm of it above.
    top = z_bed + _snap_up(z_bore + p.white_over_bore_mm + p.black_mm - z_bed)
    stop = stop_angle_deg(p.margin)
    # The post sits midway between the pins, clear of both inner links' round
    # ends, and its edge meets an inner link's side at the stop:
    # (P/2) sin θ = R + post_r + post_a |cos θ|.
    post_r = P / 2 - R - p.clearance_mm
    th = math.radians(stop)
    post_a = (P / 2 * math.sin(th) - R - post_r) / abs(math.cos(th)) if abs(math.cos(th)) > 1e-9 else 0.0
    float_r = (p.washer_id_mm - p.sleeve_mm) / 2      # how far a washer can slide off centre
    j = Joint(p=p, pitch=P, R=R, stop_deg=stop, r_cs=r_cs, r_hole=r_hole, r_ring=p.ring_od_mm / 2,
              r_keep=p.washer_od_mm / 2 + float_r + 0.1, z_cs_top=r_cs - r_hole, z_head=r_cs - p.head_d_mm / 2,
              z_head_top=z_head_top, z_washer=z_washer, z_washer_top=z_washer_top, z_seat=z_seat,
              z_bed=z_bed, z_clear=p.inner_mm + p.clearance_mm, z_bore=z_bore, z_tip=z_tip, top=top,
              post_r=post_r, post_a=post_a)
    _check(j, spec)
    return j


def _snap_up(z: float) -> float:
    return round(math.ceil(z / LAYER_MM - 1e-6) * LAYER_MM, 6)


def _check(j: Joint, spec: ChainSpec) -> None:
    p = j.p
    if j.post_r < 0.5:
        raise ValueError(f"pitch_mm {j.pitch} leaves no room for the stop post between the inner links "
                         f"(post radius {j.post_r:.2f} mm)")
    if j.post_a < 0:
        raise ValueError(f"the links collide before the {j.stop_deg:.0f}° stop; raise --margin")
    if j.stop_deg > 85.0 or j.post_a + j.post_r > j.R - 0.5:
        raise ValueError(f"a {j.stop_deg:.0f}° stop is too wide for the post (it stops the inner links by "
                         f"their sides, below 85°); lower --margin")
    if j.pitch / 2 - j.post_r < j.r_keep + 0.05:
        raise ValueError("the stop post would touch the washers")
    if p.washer_od_mm / 2 > j.R + 1e-9 or j.r_cs > j.R - 0.5:
        raise ValueError("washer or screw head too large for the link width")
    if j.z_cs_top >= p.inner_mm - 0.6 or j.z_head_top >= p.inner_mm - 1.0:
        raise ValueError("inner link too thin for the screw head")
    if not j.r_hole < j.r_ring < p.washer_od_mm / 2:
        raise ValueError("the washer's ring must lie between the hole and the washer's edge")
    if p.washer_id_mm < p.sleeve_mm or abs(p.oring_id_mm - p.sleeve_mm) > 0.4:
        raise ValueError("washer or O-ring does not fit the sleeve")
    if not 0 < p.squeeze_mm < 0.4 * p.oring_cs_mm:
        raise ValueError("squeeze_mm must be between 0 and 40% of the O-ring's cross-section")
    if j.z_seat < j.z_washer_top + 0.2:
        raise ValueError("the O-ring's seat runs into the washer")
    if spec.fiducial_mm / 2 + 1.0 > j.R:
        raise ValueError("dot too large for the link")


# ---------------------------------------------------------------------------
# Parts, each in its own frame: first pin at the origin, link along +x

def outer_link(j: Joint, spec: ChainSpec, sockets: tuple[bool, ...]):
    """Outer link with pins at (0, 0) and (pitch, 0), or a button with one.

    sockets says which pins sit in an inner link; a pin that does not (a
    chain end) gets a solid foot instead of a sleeve. A button (one pin)
    caps the last inner link of a chain with an odd pin count.

    It prints the way it is used, dots up, from z_bed: the sleeves' ends, the
    stop post and the feet stand on the bed, and everything above grows
    from them at 45°, layer by layer, around the parts of other links that
    move under the plate: the inner links (below z_clear), and around each
    pin the washer and the O-ring under its 45° seat.
    """
    import shapely
    from shapely.geometry import Point
    from shapely.ops import unary_union

    p, seg, P = j.p, j.p.segments, j.pitch
    q = seg // 4
    pins = [(0.0, 0.0), (P, 0.0)][:len(sockets)]
    outline = _stadium(pins[0], pins[-1], j.R, seg)
    rs = p.sleeve_mm / 2
    base = [Point(at).buffer(rs if s else j.R, quad_segs=q) for at, s in zip(pins, sockets)]
    if len(pins) == 2:
        base.append(post_outline(j))
    base = unary_union(base)
    socket_pins = [at for at, s in zip(pins, sockets) if s]
    n = round((j.top - j.z_bed) / LAYER_MM)
    layers, region = [], base
    for k in range(n):
        z0 = j.z_bed + k * LAYER_MM
        if k > 0:
            grown = region.buffer(LAYER_MM, quad_segs=q)
            allowed = outline if z0 >= j.z_clear - 1e-6 else base
            for at in socket_pins:
                # Around a pin: only the 45° seat over the O-ring, and above the washer.
                if z0 < j.z_washer_top + 0.2 - 1e-6:
                    allowed = allowed.difference(Point(at).buffer(j.r_keep, quad_segs=q).difference(
                        Point(at).buffer(rs, quad_segs=q)))
                seat = rs + max(0.0, z0 - j.z_seat)
                if seat < j.r_keep:
                    ring = Point(at).buffer(j.r_keep, quad_segs=q).difference(Point(at).buffer(seat, quad_segs=q))
                    allowed = allowed.difference(ring)
            region = unary_union([grown.intersection(allowed), base]).intersection(outline)
        layers.append((z0, shapely.set_precision(region.simplify(0.005), 1e-4)))
    body = None
    for z0, geom in layers:
        slab = _extrude(geom, z0, z0 + LAYER_MM)
        body = slab if body is None else body + slab
    bore = _revolve(_bore_profile(j), seg)
    for at in socket_pins:
        body -= bore.translate((at[0], at[1], 0))
    for at in pins:
        body -= _extrude(_window(spec, seg, at), j.black_from, j.top + 1.0)
    return body


def post_outline(j: Joint):
    from shapely.geometry import LineString, Point

    c = (j.pitch / 2, 0.0)
    if j.post_a < 1e-6:
        return Point(c).buffer(j.post_r, quad_segs=16)
    return LineString([(c[0], -j.post_a), (c[0], j.post_a)]).buffer(j.post_r, quad_segs=16)


def _bore_profile(j: Joint):
    from shapely.geometry import box

    return box(0.0, j.z_bed - 1.0, j.p.pilot_mm / 2, j.z_bore)


def inner_link(j: Joint):
    """Inner link with countersunk holes at (0, 0) and (pitch, 0)."""
    from shapely.geometry import Point, Polygon

    seg, P, p = j.p.segments, j.pitch, j.p
    body = _extrude(_stadium((0, 0), (P, 0), j.R, seg), 0.0, p.inner_mm)
    ring = Point(0, 0).buffer(j.r_ring, quad_segs=seg // 4)
    hole = Polygon([(0, -1.0), (j.r_cs + 1.0, -1.0), (j.r_hole, j.z_cs_top), (j.r_hole, j.top),
                    (0, j.top)])
    cut = _revolve(hole, seg)
    for x in (0.0, P):
        body += _extrude(ring, p.inner_mm - 0.01, j.z_washer).translate((x, 0, 0))
    for x in (0.0, P):
        body -= cut.translate((x, 0, 0))
    return body


def screw(j: Joint):
    """M3 countersunk screw as seated, its core (pilot radius) for the thread."""
    from shapely.geometry import Polygon

    p = j.p
    core = p.pilot_mm / 2 - 0.05
    return _revolve(Polygon([(0, j.z_head), (p.head_d_mm / 2, j.z_head), (p.screw_d_mm / 2, j.z_head_top),
                             (core, j.z_head_top), (core, j.z_tip), (0, j.z_tip)]), j.p.segments)


def washer(j: Joint):
    from shapely.geometry import box

    p = j.p
    return _revolve(box(p.washer_id_mm / 2, j.z_washer, p.washer_od_mm / 2, j.z_washer_top), p.segments)


def oring(j: Joint):
    """The O-ring as it rests on the washer against the sleeve, before the
    seat squeezes it: it overlaps the outer link's seat by the squeeze."""
    from shapely.geometry import Point

    rc, zc = j.oring_centre
    return _revolve(Point(rc, zc).buffer(j.p.oring_cs_mm / 2, quad_segs=16), j.p.segments)


# ---------------------------------------------------------------------------
# The whole chain

@dataclass(frozen=True)
class Part:
    kind: str              # "outer", "inner", "button", "screw", "washer" or "oring"
    pins: tuple[int, ...]  # chain pins it carries (outer, button, hardware) or holds (inner)
    solid: object          # manifold3d.Manifold in the part's own frame (first pin at the origin)

    @property
    def printed(self) -> bool:
        return self.kind in ("outer", "inner", "button")


def chain_parts(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS, n_pins: int | None = None) -> list[Part]:
    """Every part of the chain. The first link is an outer one and they
    alternate; a button caps the last inner link when the pin count is odd.
    Every pin held by an inner link gets a screw, a washer and an O-ring."""
    j = joint_geometry(spec, p)
    n = spec.n_pins if n_pins is None else n_pins
    if n < 2:
        raise ValueError("need at least 2 pins")
    cache: dict = {}

    def cached(key, make):
        if key not in cache:
            cache[key] = make()
        return cache[key]

    parts = []
    for k in range(n - 1):
        if k % 2 == 0:
            sockets = (k > 0, k + 1 < n - 1)
            parts.append(Part("outer", (k, k + 1), cached(sockets, lambda: outer_link(j, spec, sockets))))
        else:
            parts.append(Part("inner", (k, k + 1), cached("inner", lambda: inner_link(j))))
    if (n - 2) % 2 == 1:
        parts.append(Part("button", (n - 1,), cached("button", lambda: outer_link(j, spec, (True,)))))
    for k in range(1, n):
        if k < n - 1 or (n - 2) % 2 == 1:
            for kind, make in (("screw", screw), ("washer", washer), ("oring", oring)):
                parts.append(Part(kind, (k,), cached(kind, lambda make=make: make(j))))
    return parts


def posed(parts: list[Part], j: Joint, turns_deg=None) -> list:
    """Solids of the assembled chain bent by turns_deg (one per interior pin,
    degrees, counter-clockwise positive); straight if None."""
    n = max(max(pt.pins) for pt in parts) + 1
    turns = np.zeros(n - 2) if turns_deg is None else np.radians(np.asarray(turns_deg, dtype=float))
    if len(turns) != n - 2:
        raise ValueError(f"need {n - 2} turns, got {len(turns)}")
    pins = pins_from_turns(j.pitch, turns)
    heading = np.concatenate([[0.0], np.cumsum(turns)])
    out = []
    for pt in parts:
        k = pt.pins[0]
        h = heading[min(k, len(heading) - 1)]
        out.append(pt.solid.rotate((0, 0, math.degrees(h))).translate((pins[k][0], pins[k][1], 0)))
    return out


def turns_at_stop(n_pins: int, j: Joint, sign: int = 1, slack_deg: float = 0.0) -> np.ndarray:
    """A zigzag with every joint bent to its stop (less slack_deg)."""
    bend = 180.0 - j.stop_deg - slack_deg
    return sign * bend * np.array([(-1) ** k for k in range(n_pins - 2)], dtype=float)


def print_layout(parts: list[Part], j: Joint, gap: float = 3.0):
    """All printed parts on one bed, as they are used (dots up): outer links
    and buttons lowered onto the bed by z_bed, in rows, then the inner links."""
    import manifold3d as m

    solids = ([pt.solid.translate((0, 0, -j.z_bed)) for pt in parts if pt.kind in ("outer", "button")]
              + [pt.solid for pt in parts if pt.kind == "inner"])
    per_row, placed, y = 4, [], 0.0
    for r in range(0, len(solids), per_row):
        x = 0.0
        for s in solids[r:r + per_row]:
            x0, y0, _, x1, _, _ = s.bounding_box()
            placed.append(s.translate((x - x0, y - y0, 0)))
            x += (x1 - x0) + gap
        y += 2 * j.R + gap
    return m.Manifold.batch_boolean(placed, m.OpType.Add)


def print_heights(j: Joint) -> dict:
    """Colour changes as print heights (bed at 0): white from, black from."""
    return {"white": round(j.white_from - j.z_bed, 6), "black": round(j.black_from - j.z_bed, 6)}


def colour_split(solid, j: Joint) -> dict:
    """Printed parts on the bed cut at the colour changes: "black" (below
    the white and above it) and "white"."""
    h = print_heights(j)
    lower, rest = solid.trim_by_plane((0, 0, -1), -h["white"]), solid.trim_by_plane((0, 0, 1), h["white"])
    white, upper = rest.trim_by_plane((0, 0, -1), -h["black"]), rest.trim_by_plane((0, 0, 1), h["black"])
    return {"black": lower + upper, "white": white}


def friction_estimate(j: Joint) -> dict:
    """Rough joint numbers: O-ring force, the two friction torques, and
    their ratio (the countersink must out-hold the washer, see the module
    docstring)."""
    p = j.p
    force = ORING_N_PER_MM * (p.squeeze_mm / p.oring_cs_mm / 0.2) * math.pi * (p.oring_id_mm + p.oring_cs_mm)
    r_cone = (j.r_hole + p.head_d_mm / 2) / 2
    r_ring = (j.r_hole + j.r_ring) / 2
    countersink = MU_STEEL_PETG * force / math.sin(math.pi / 4) * r_cone
    washer_t = MU_STEEL_PETG * force * r_ring
    return {"force_N": force, "countersink_Nmm": countersink, "washer_Nmm": washer_t,
            "torque_Nmm": countersink + washer_t, "hold_ratio": countersink / washer_t}


def counts(parts: list[Part]) -> dict:
    out: dict = {}
    for pt in parts:
        out[pt.kind] = out.get(pt.kind, 0) + 1
    return out


def write_oring_chain(out_dir: Path, spec: ChainSpec, p: OringParams = DEFAULT_PARAMS,
                      n_pins: int | None = None, name: str = "oring-chain") -> dict[str, Path]:
    import manifold3d as m

    j = joint_geometry(spec, p)
    parts = chain_parts(spec, p, n_pins)
    bed = print_layout(parts, j)
    split = colour_split(bed, j)
    assembled = m.Manifold.batch_boolean(posed(parts, j), m.OpType.Add)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": out_dir / f"{name}.stl",
        "black": out_dir / f"{name}-black.stl",
        "white": out_dir / f"{name}-white.stl",
        "assembled": out_dir / f"{name}-assembled.stl",
        "instructions": out_dir / f"{name}-PRINTING.txt",
        "drawing": out_dir / f"{name}-joint.svg",
    }
    to_trimesh(bed).export(paths["combined"])
    to_trimesh(split["black"]).export(paths["black"])
    to_trimesh(split["white"]).export(paths["white"])
    to_trimesh(assembled).export(paths["assembled"])
    n = max(max(pt.pins) for pt in parts) + 1
    paths["instructions"].write_text(print_instructions(name, j, parts, n), encoding="utf-8")
    paths["drawing"].write_text(joint_svg(spec, p), encoding="utf-8")
    return paths


def print_instructions(name: str, j: Joint, parts: list[Part], n_pins: int) -> str:
    p, c, h = j.p, counts(parts), print_heights(j)
    fe = friction_estimate(j)
    length = (n_pins - 1) * j.pitch + 2 * j.R
    n_joints = c.get("screw", 0)
    n_outer = c.get("outer", 0) + c.get("button", 0)
    button = ", one of them the round button for the last pin" if c.get("button") else ""
    return f"""Spline Wire O-ring chain: {name}
{n_pins} pins, pitch {j.pitch:g} mm, {length:.1f} x {2 * j.R:.1f} x {j.top:.1f} mm assembled.
Each joint bends up to {180 - j.stop_deg:.0f} deg either way: the links close to {j.stop_deg:.0f} deg at the
least, so the pins either side of a joint stay {1 + p.margin:.2f} pitches apart.

Print {name}.stl: {n_outer} outer links (dots up{button}) and {c.get("inner", 0)} inner links.
Hardware for {n_joints} joints, from assortment boxes:
  {n_joints} x M3 x {p.screw_len_mm:g} countersunk screw, 90 deg head (ISO 10642, DIN 7991 or DIN 965)
  {n_joints} x O-ring {p.oring_id_mm:g} x {p.oring_cs_mm:g} mm (ID x cross-section), NBR 70A
  {n_joints} x M4 small washer, {p.washer_id_mm:g} x {p.washer_od_mm:g} x {p.washer_mm:g} mm (ISO 7092 / DIN 433), steel

Printing (PETG; matte if you can: a lamp reflected in shiny filament washes
the dots out, docs/cv-robustness.md):
  1. Print it as it comes, everything dots up / flat side down. 0.2 mm
     layers, no supports, no brim.
  2. Start with BLACK.
  3. Change to WHITE at the first layer above {h["white"]:.1f} mm (Z = {h["white"] + LAYER_MM:.1f} mm).
  4. Change back to BLACK at the first layer above {h["black"]:.1f} mm (Z = {h["black"] + LAYER_MM:.1f} mm).
     The inner links are lower than both, so they come out all black.
     PrusaSlicer/Orca/Bambu Studio: right-click the "+" on the layer slider
     -> Add color change. Multi-material: load {name}-black.stl and
     {name}-white.stl as one object with two parts.
  5. XY size/hole compensation off: the pitch is the dimension that matters.
     Elephant-foot compensation as usual. 3+ perimeters, 100% infill or
     close to it (the parts are small). The stop posts under the outer
     links are small: keep the minimum layer time at ~8 s so they cool.
Each dot is a {p.black_mm:g} mm deep window in the black top onto the white.

Assembly, per joint (outer link dots down on the table):
  1. Cut the thread first: drive a screw into the sleeve until its head
     touches the sleeve's end, then take it out again.
  2. Slide an O-ring over the sleeve up into its seat, then a washer.
  3. Put an inner link's hole over the sleeve, raised ring towards the
     washer, countersink facing you.
  4. Drive the screw through the countersink into the sleeve. It gets
     firmer as the O-ring squeezes, then stops hard when the head reaches the
     sleeve's end: stop there. That sets the squeeze, and so the friction,
     the same at every joint.
  5. Turn the joint: it should move smoothly, hold its angle, and stop
     against the post under the outer link at about {j.stop_deg:.0f} deg between links.

Checks:
  - No play you can feel when you push the links together or pull them apart.
  - Bend a joint 90 deg, let go, and look along it: it should not creep back.
  - Lay the chain straight along a ruler: {length:.1f} mm end to end
    ({n_pins - 1} x {j.pitch:g} mm pitch + 2 x {j.R:g} mm). More than ~0.3 mm off means the
    printer's XY scale is off; measure the pitch and put it in chain.yaml.

Estimated: O-ring force {fe["force_N"]:.0f} N, friction {fe["torque_Nmm"]:.0f} N mm per joint (countersink
{fe["countersink_Nmm"]:.0f}, washer {fe["washer_Nmm"]:.0f}). The countersink holds {fe["hold_ratio"]:.1f}x the twist the O-ring can
store, so a joint does not spring back. Too stiff or too loose: regenerate
with a different --squeeze (default {OringParams().squeeze_mm:g} mm).
Try a short piece first: splinewire oring-chain --pins 4 (two joints).
To measure photos of this chain, set max_bend_deg: 125 in chain.yaml.
"""


# ---------------------------------------------------------------------------
# Drawing of one joint, for the docs and the output folder

def joint_svg(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS) -> str:
    """Section along the chain through two joints, straight, and a plan of
    a joint at its stop."""
    from shapely import affinity
    from shapely.geometry import box

    j = joint_geometry(spec, p)
    parts = chain_parts(spec, p, n_pins=4)
    scale, s2, pad, title = 28.0, 11.0, 14.0, 16.0
    fill = {"screw": "#9aa3ad", "washer": "#c4cad1", "oring": "#c0392b", "inner": "#3a3a3a"}
    stroke = {"outer": "#2f6db5", "inner": "#d0782a", "screw": "#555", "washer": "#555", "oring": "#7b241c"}
    zones = [(-1.0, j.white_from, "#3a3a3a"), (j.white_from, j.black_from, "#f4f1ea"),
             (j.black_from, j.top + 1, "#3a3a3a")]

    # Section along the chain (y = 0) through pins 1 and 2.
    x_lo, x_hi = 0.5 * j.pitch - 1.0, 2.5 * j.pitch + 1.0
    w1, h1 = x_hi - x_lo, j.top + 0.5
    g = [f'<g transform="translate({pad:.1f},{pad + title:.1f})">',
         '<text x="0" y="-6" class="t">Section along the chain through two joints</text>']
    for pt, solid in zip(parts, posed(parts, j)):
        geom = _polys_to_shapely(solid.rotate((-90, 0, 0)).slice(0.0).to_polygons())
        geom = geom.intersection(box(x_lo, -1, x_hi, j.top + 1))
        if geom.is_empty:
            continue
        if pt.kind == "outer":
            for z0, z1, colour in zones:
                g += _svg_paths(geom.intersection(box(x_lo, z0, x_hi, z1)), colour, "none", scale, -x_lo, h1)
            g += _svg_paths(geom, "none", stroke["outer"], scale, -x_lo, h1)
        else:
            g += _svg_paths(geom, fill[pt.kind], stroke[pt.kind], scale, -x_lo, h1)
    g.append("</g>")
    x_at = pad + w1 * scale + 2 * pad

    # Plan of pin 1 at its stop: a cut through the inner links and the posts,
    # with the outer plates outlined.
    z_cut = (j.z_bed + j.p.inner_mm) / 2
    plan = []
    for pt, solid in zip(parts, posed(parts, j, [180.0 - j.stop_deg, 0.0])):
        if pt.kind in ("outer", "inner"):
            cut = _polys_to_shapely(solid.slice(z_cut).to_polygons())
            plate = _polys_to_shapely(solid.slice(j.top - 0.1).to_polygons()) if pt.kind == "outer" else None
            plan.append((pt.kind, cut, plate))
    shapes = [geom for _, geom, _ in plan] + [plate for _, _, plate in plan if plate is not None]
    bx0 = min(s.bounds[0] for s in shapes) - 1
    by0 = min(s.bounds[1] for s in shapes) - 1
    bx1 = max(s.bounds[2] for s in shapes) + 1
    by1 = max(s.bounds[3] for s in shapes) + 1
    h2 = by1 - by0
    g2 = [f'<g transform="translate({x_at:.1f},{pad + title:.1f})">',
          f'<text x="0" y="-6" class="t">Plan at the stop: {j.stop_deg:.0f}° between links</text>']
    for kind, geom, plate in plan:
        colour = "#3a3a3a" if kind == "inner" else "#8a8a8a"
        g2 += _svg_paths(affinity.translate(geom, -bx0, -by0), colour, stroke[kind], s2, 0, h2)
        if plate is not None:
            g2 += _svg_paths(affinity.translate(plate, -bx0, -by0), "none", stroke["outer"], s2, 0, h2)
    g2.append("</g>")
    width = x_at + (bx1 - bx0) * s2 + pad
    height = max(h1 * scale, h2 * s2) + 2 * pad + title
    legend = ("Section: blue outline = outer link (fill = filament colour, dots on top); orange = inner link; "
              "grey = screw and washer; red = O-ring.",
              f"Plan: cut {z_cut:.1f} mm above the table through the inner links and the stop posts; "
              "outer plates outlined. Dimensions: docs/printed-chain.md.")
    text = [f'<text x="{pad}" y="{height + 16 * i:.0f}" class="t">{line}</text>' for i, line in enumerate(legend)]
    height += 32
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}">\n'
            '<style>.t{font:12px sans-serif;fill:#333}</style>\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(g + g2 + text) + "\n</svg>\n")
