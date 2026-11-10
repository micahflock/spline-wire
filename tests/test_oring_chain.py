import itertools
import math

import numpy as np
import pytest

pytest.importorskip("manifold3d")
pytest.importorskip("trimesh")
shapely = pytest.importorskip("shapely")

from splinewire.chain import pins_from_turns
from splinewire.oring_chain import (
    OringParams,
    chain_parts,
    colour_split,
    counts,
    friction_estimate,
    joint_geometry,
    posed,
    print_heights,
    print_layout,
    screw_gauge,
    stop_angle_deg,
    to_trimesh,
    turns_at_stop,
    write_oring_chain,
)
from splinewire.printed_chain import LAYER_MM, _polys_to_shapely

TOUCH = 0.01        # mm³: facets of two revolved parts turned to different angles


@pytest.fixture(scope="module")
def chain():
    from splinewire.chain import default_chain_path, load_chain_spec

    spec = load_chain_spec(default_chain_path())
    j = joint_geometry(spec)
    return spec, j, chain_parts(spec, n_pins=4)


def _overlaps(parts, solids):
    """Volume shared by each pair of parts that overlap."""
    out = {}
    for (a, sa), (b, sb) in itertools.combinations(zip(parts, solids), 2):
        v = (sa ^ sb).volume()
        if v > TOUCH:
            out[(a.kind, a.pins, b.kind, b.pins)] = v
    return out


def _squeezed(overlaps):
    """Only the O-rings overlap anything: their seats, by the squeeze."""
    return {k: v for k, v in overlaps.items() if not (k[0] == "outer" and k[2] == "oring")}


def test_stop_angle_keeps_neighbours_a_margin_apart():
    assert stop_angle_deg(0.0) == pytest.approx(60.0)
    assert stop_angle_deg(0.2) == pytest.approx(73.74, abs=0.01)
    th = math.radians(stop_angle_deg(0.2))
    pins = pins_from_turns(10.0, np.array([math.pi - th]))
    assert np.linalg.norm(pins[2] - pins[0]) == pytest.approx(12.0)


def test_parts_and_hardware(chain):
    spec, _, _ = chain
    full = chain_parts(spec)
    assert spec.n_pins == 13
    assert counts(full) == {"outer": 6, "inner": 6, "button": 1, "screw": 12, "washer": 12, "oring": 12}
    even = chain_parts(spec, n_pins=14)
    assert counts(even)["screw"] == 12 and "button" not in counts(even)
    for pt in full:
        assert to_trimesh(pt.solid).is_watertight


def test_straight_chain_fits_together(chain):
    _, j, parts = chain
    over = _overlaps(parts, posed(parts, j))
    assert _squeezed(over) == {}
    rings = [v for k, v in over.items() if k[2] == "oring"]
    assert len(rings) == 2 and all(v > 0.5 for v in rings)      # each O-ring is squeezed by its seat


def test_both_links_seat_on_the_screw_head(chain):
    """The sleeve's end and the countersink touch the head: either one moved
    0.05 mm towards it overlaps it."""
    _, j, parts = chain
    solids = posed(parts, j)
    by = {(pt.kind, pt.pins): s for pt, s in zip(parts, solids)}
    head = by[("screw", (1,))]
    outer, inner = by[("outer", (0, 1))], by[("inner", (1, 2))]
    assert (head ^ outer).volume() < TOUCH and (head ^ inner).volume() < TOUCH
    assert (head ^ outer.translate((0, 0, -0.05))).volume() > TOUCH
    assert (head ^ inner.translate((0, 0, -0.05))).volume() > TOUCH


@pytest.mark.parametrize("sign", [1, -1])
def test_joints_turn_to_the_stop_and_no_further(chain, sign):
    _, j, parts = chain
    assert _squeezed(_overlaps(parts, posed(parts, j, turns_at_stop(4, j, sign, slack_deg=1.0)))) == {}
    past = _squeezed(_overlaps(parts, posed(parts, j, turns_at_stop(4, j, sign, slack_deg=-1.0))))
    # Past the stop, each inner link runs into the post of the outer link it turns against.
    assert set(past) == {("outer", (0, 1), "inner", (1, 2)), ("inner", (1, 2), "outer", (2, 3))}


def test_outer_plates_clear_each_other_at_the_stop(chain):
    _, j, parts = chain
    solids = posed(parts, j, turns_at_stop(4, j, 1))
    a, c = solids[0], solids[2]
    assert a.min_gap(c, 3.0) > 1.0


def test_parts_print_without_support(chain):
    """Per 0.2 mm layer, on the bed as printed: nothing starts in mid-air and
    nothing overhangs by more than 45°, except the bridge that closes each
    screw hole at its top."""
    _, j, parts = chain
    for pt in parts:
        if pt.printed:
            solid = pt.solid.translate((0, 0, -j.z_bed)) if pt.kind != "inner" else pt.solid
            _assert_prints_without_support(solid, j, [(0.0, j.p.pilot_mm), (j.pitch, j.p.pilot_mm)], pt.kind)


def _assert_prints_without_support(solid, j, bores, what):
    """solid on the bed; bores: (x, diameter) of the screw holes' bridged tops."""
    assert solid.bounding_box()[2] == pytest.approx(0.0, abs=1e-6)
    n = round(solid.bounding_box()[5] / LAYER_MM)
    assert n * LAYER_MM == pytest.approx(solid.bounding_box()[5], abs=1e-6)
    bridges = shapely.ops.unary_union([shapely.geometry.Point(x, 0).buffer(d / 2 + 0.01) for x, d in bores])
    prev = None
    for i in range(n):
        here = _polys_to_shapely(solid.slice((i + 0.5) * LAYER_MM).to_polygons())
        if prev is not None:
            for piece in getattr(here, "geoms", [here]):
                assert piece.area < 1e-3 or piece.intersects(prev)
            hanging = here.difference(prev.buffer(LAYER_MM + 0.02)).difference(bridges)
            assert hanging.area < 1e-3, (what, i)
        prev = here


def _hole_diameter(solid, x, z):
    """Diameter of the round hole around (x, 0) in the solid's section at height z."""
    section = _polys_to_shapely(solid.slice(z).to_polygons())
    holes = [ring for poly in getattr(section, "geoms", [section]) for ring in poly.interiors
             if shapely.geometry.Polygon(ring).contains(shapely.geometry.Point(x, 0))]
    assert len(holes) == 1
    return 2 * math.sqrt(shapely.geometry.Polygon(holes[0]).area / math.pi)


def test_screw_hole_tapers_from_the_screw_to_the_pilot(chain):
    """The screw has to start by hand in a printed hole and then bite: the
    hole is the screw's diameter at the sleeve's end, the pilot at its tip."""
    _, j, parts = chain
    outer, p = parts[0].solid, j.p
    assert j.lead_mm == pytest.approx(p.screw_d_mm) and p.pilot_mm < j.lead_mm
    assert _hole_diameter(outer, j.pitch, j.z_bed + 0.01) == pytest.approx(j.lead_mm, abs=0.01)
    assert _hole_diameter(outer, j.pitch, j.z_tip - 0.01) == pytest.approx(p.pilot_mm, abs=0.01)
    assert _hole_diameter(outer, j.pitch, j.z_bore - 0.01) == pytest.approx(p.pilot_mm, abs=0.01)
    mid = _hole_diameter(outer, j.pitch, (j.z_bed + j.z_tip) / 2)
    assert mid == pytest.approx((j.lead_mm + p.pilot_mm) / 2, abs=0.01)
    assert (p.sleeve_mm - j.lead_mm) / 2 >= 0.5 - 1e-9           # a wall the slicer still prints


def test_screw_gauge_has_the_chains_sleeves_in_five_sizes(chain):
    spec, j, parts = chain
    sizes = (2.6, 2.7, 2.8, 2.9, 3.0)
    gauge = screw_gauge(j)
    assert to_trimesh(gauge).is_watertight
    assert j.p.pilot_mm in sizes
    for k, d in enumerate(sizes, start=1):
        assert _hole_diameter(gauge, k * j.pitch, j.z_tip - 0.01) == pytest.approx(d, abs=0.01)
        assert _hole_diameter(gauge, k * j.pitch, j.z_bed + 0.01) == pytest.approx(j.lead_mm, abs=0.01)
    # A sleeve of the gauge is a sleeve of the chain: same section through the seat, hole aside.
    outer = parts[2].solid                                      # pins 2 and 3: a sleeve at its first pin
    k = sizes.index(j.p.pilot_mm) + 1
    near = shapely.geometry.Point(0, 0).buffer(3.0)
    for z in (j.z_bed + 0.1, j.z_seat - 0.1, j.z_seat + 0.5):
        a = _polys_to_shapely(outer.slice(z).to_polygons()).intersection(near)
        b = _polys_to_shapely(gauge.translate((-k * j.pitch, 0, 0)).slice(z).to_polygons()).intersection(near)
        assert a.area > 5.0 and a.symmetric_difference(b).area < 0.05     # the outlines are simplified to 0.005
    # The foot marks the end with the smallest hole: solid down to the bed.
    foot = _polys_to_shapely(gauge.slice(j.z_bed + 0.1).to_polygons())
    assert foot.contains(shapely.geometry.Point(0, 0)) and not foot.contains(shapely.geometry.Point(j.pitch, 0))
    _assert_prints_without_support(gauge.translate((0, 0, -j.z_bed)), j,
                                   [(k * j.pitch, d) for k, d in enumerate(sizes, start=1)], "gauge")
    with pytest.raises(ValueError):
        screw_gauge(j, (2.8, 3.2))


def test_one_bed_and_colours(chain):
    spec, j, parts = chain
    h = print_heights(j)
    assert all(abs(z / LAYER_MM - round(z / LAYER_MM)) < 1e-6 for z in h.values())
    assert j.p.inner_mm + j.p.ring_mm < h["white"]               # inner links come out all black
    bed = print_layout(parts, j)
    split = colour_split(bed, j)
    assert split["black"].volume() + split["white"].volume() == pytest.approx(bed.volume(), rel=1e-3)
    # Under every dot: a window through the black onto white, black around it.
    outer = parts[0].solid
    top = _polys_to_shapely(outer.slice(j.top - 0.1).to_polygons())
    below = _polys_to_shapely(outer.slice(j.black_from - 0.1).to_polygons())
    for x in (0.0, spec.pitch_mm):
        for r in (0.0, 0.45 * spec.fiducial_mm):
            for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
                pt = shapely.geometry.Point(x + r * math.cos(a), r * math.sin(a))
                assert not top.contains(pt) and below.contains(pt)
        assert top.contains(shapely.geometry.Point(x + spec.fiducial_mm / 2 + 0.3, 0))
    # White over the screw's hole, under the dot.
    assert j.white_from <= j.z_bore + 0.001 and j.black_from - j.z_bore >= j.p.white_over_bore_mm - 1e-6


def test_countersink_outholds_the_washer():
    """The O-ring stores at most the washer's friction as twist; the
    countersink must hold more than that or the joint springs back."""
    j = joint_geometry(__import__("splinewire.chain", fromlist=["x"]).load_chain_spec(
        __import__("splinewire.chain", fromlist=["x"]).default_chain_path()))
    f = friction_estimate(j)
    assert f["hold_ratio"] > 1.3
    assert 5.0 < f["torque_Nmm"] < 30.0


def test_bad_parameters_are_refused(chain):
    spec, _, _ = chain
    with pytest.raises(ValueError):
        joint_geometry(spec, OringParams(margin=-0.1))
    with pytest.raises(ValueError):
        joint_geometry(spec, OringParams(squeeze_mm=0.8))
    with pytest.raises(ValueError):
        joint_geometry(spec, OringParams(margin=0.5))
    with pytest.raises(ValueError):
        joint_geometry(spec, OringParams(pilot_mm=3.1))     # wider than the screw: no thread
    with pytest.raises(ValueError):
        joint_geometry(spec, OringParams(pilot_mm=2.3))     # under the screw's core: it cannot go in
    joint_geometry(spec, OringParams(pilot_mm=3.0))         # a straight hole, for printers whose holes shrink most


def test_screw_hole_is_drawn_open_for_a_printed_hole():
    # Small printed holes shrink 0.1-0.3 mm; a 2.5 mm (tap drill) hole came out too tight for an M3.
    assert 2.7 <= OringParams().pilot_mm < 3.0


def test_write_files(tmp_path, chain):
    spec, _, _ = chain
    paths = write_oring_chain(tmp_path, spec, n_pins=4, name="test")
    for p in paths.values():
        assert p.is_file() and p.stat().st_size > 0
    text = paths["instructions"].read_text(encoding="utf-8")
    h = print_heights(joint_geometry(spec))
    assert f"{h['white']:.1f} mm" in text and f"{h['black']:.1f} mm" in text
    assert "2 x M3 x 6 countersunk" in text
    assert "test-screw-gauge.stl" in text and "2.6, 2.7, 2.8, 2.9 and 3.0 mm" in text
    import trimesh
    gauge = trimesh.load(paths["gauge"])
    assert gauge.is_watertight and gauge.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert paths["drawing"].read_text(encoding="utf-8").startswith("<svg")
    page = paths["viewer"].read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "{{" not in page and "<!--FIT_ROWS-->" not in page
    assert "window.CHAIN=" in page and "106.3" in page and "Ø4.5 on Ø4.0" in page
    model = trimesh.load(paths["model"])
    assert len(model.geometry) == 9                      # 2 outer, 1 inner, 2 x (screw, washer, O-ring)


def test_viewer_data_matches_the_parts(chain):
    import base64

    from splinewire.oring_viewer import fit_rows, viewer_data

    spec, j, parts = chain
    data = viewer_data(spec)
    assert [it["kind"] for it in data["items"]] == [pt.kind for pt in parts]
    for it, pt in zip(data["items"], parts):
        g = data["geoms"][it["geom"]]
        v = np.frombuffer(base64.b64decode(g["pos"]), dtype=np.float32).reshape(-1, 3)
        assert np.allclose(v.min(axis=0), pt.solid.bounding_box()[:3], atol=1e-4)
    rows = dict((a, c) for a, _, c in fit_rows(spec))
    assert rows["Washer on the sleeve"] == "0.15 mm play, free"
    assert rows["Outer plates at the stop"] == "plates 1.6 mm apart"
