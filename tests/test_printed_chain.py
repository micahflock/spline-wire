import itertools
import math

import numpy as np
import pytest

pytest.importorskip("manifold3d")
pytest.importorskip("trimesh")
shapely = pytest.importorskip("shapely")

from splinewire.printed_chain import (
    LAYER_MM,
    JointParams,
    _polys_to_shapely,
    chain_bodies,
    colour_split,
    joint_geometry,
    set_pose,
    to_trimesh,
    write_printed_chain,
)


@pytest.fixture(scope="module")
def chain(request):
    from splinewire.chain import default_chain_path, load_chain_spec

    spec = load_chain_spec(default_chain_path())
    return spec, joint_geometry(spec), chain_bodies(spec)


def _on_layer(z):
    return abs(z / LAYER_MM - round(z / LAYER_MM)) < 1e-6


def test_colour_changes_on_layer_boundaries(chain):
    _, j, _ = chain
    assert _on_layer(j.white_from) and _on_layer(j.black_from) and _on_layer(j.top)
    assert j.black_from - j.white_from >= 1.0          # enough white under the dots to look white
    assert j.inner_top == j.white_from                 # inner links are black all through


def test_links_alternate_and_end_in_a_button(chain):
    spec, _, bodies = chain
    assert spec.n_pins == 13
    assert [b.kind for b in bodies] == ["outer", "inner"] * 6 + ["button"]
    assert sorted({i for b in bodies for i in b.pins}) == list(range(13))
    even = chain_bodies(spec, n_pins=4)
    assert [b.kind for b in even] == ["outer", "inner", "outer"]


def test_every_part_is_a_closed_solid(chain):
    _, _, bodies = chain
    for b in bodies:
        assert to_trimesh(b.solid).is_watertight


def test_parts_are_apart_as_printed(chain):
    _, j, bodies = chain
    c = j.p.clearance_mm
    for a, b in itertools.combinations(bodies, 2):
        assert a.solid.min_gap(b.solid, 1.0) > 0.98 * c


def _grip(j, pair, travel):
    """Where an outer and an inner link sharing a pin overlap once set."""
    a, b = set_pose(pair, j, [0.0], travel=travel)
    return a ^ b


def test_setting_seats_the_pin_in_the_v_against_the_spring(chain):
    spec, j, _ = chain
    pair = chain_bodies(spec, n_pins=3)[:2]           # outer link, inner link sharing pin 1
    assert _grip(j, pair, 0.0).is_empty()
    for travel in (j.travel, j.full_travel):
        grip = _grip(j, pair, travel)
        x0, _, z0, x1, y1, z1 = grip.bounding_box()
        # Only at the land, raised by the travel, around pin 1 (x = pitch).
        assert z0 >= j.z_land - j.chamfer + travel - 1e-3
        assert z1 <= j.z_land_top + j.chamfer + travel + 1e-3 or z1 <= j.z_shoulder + 1e-3
        assert j.pitch - 2.0 < x0 and x1 < j.pitch + 2.0
        # The spring side bites by the preload; the V side just touches the barrel.
        assert y1 == pytest.approx(j.r_barrel, abs=0.01)
        spring = grip ^ grip.trim_by_plane((0, 1, 0), 0.0)
        assert spring.bounding_box()[1] == pytest.approx(j.r_barrel - j.p.preload_mm, abs=0.01)
        v_side = grip.trim_by_plane((0, -1, 0), 0.0)
        assert -v_side.bounding_box()[1] < j.r_barrel        # flanks, not the whole half-circle


@pytest.mark.parametrize("sign", [1, -1])
def test_every_joint_turns_to_the_limit_without_collisions(chain, sign):
    """Zigzag at nearly the full bend: neighbours meet only where the pin is
    held, others not at all."""
    spec, j, bodies = chain
    turns = sign * (j.p.max_turn_deg - 2) * np.array([(-1) ** k for k in range(spec.n_pins - 2)])
    held = _grip(j, bodies[:2], j.full_travel).volume()
    posed = set_pose(bodies, j, turns)
    for (a, sa), (b, sb) in itertools.combinations(zip(bodies, posed), 2):
        overlap = (sa ^ sb).volume()
        if set(a.pins) & set(b.pins):
            assert overlap == pytest.approx(held, rel=0.05)
        else:
            assert overlap < 1e-6


def test_nothing_prints_onto_another_part(chain):
    """Per 0.2 mm layer: whatever a part adds beyond a 45° overhang of the
    layer under it hangs over air (a bridge), never over another part
    within 1 mm, and no part starts in mid-air."""
    spec, j, _ = chain
    bodies = chain_bodies(spec, n_pins=4)
    n = round(j.top / LAYER_MM)
    layers = [[_polys_to_shapely(b.solid.slice((i + 0.5) * LAYER_MM).to_polygons()) for i in range(n)]
              for b in bodies]
    for k in range(len(bodies)):
        for i in range(1, n):
            here, below = layers[k][i], layers[k][i - 1]
            for piece in getattr(here, "geoms", [here]):
                assert piece.area < 1e-3 or piece.intersects(below)
            hanging = here.difference(below.buffer(LAYER_MM + 0.02))
            if hanging.area < 1e-3:
                continue
            others = shapely.ops.unary_union([layers[o][ii] for o in range(len(bodies)) if o != k
                                              for ii in range(max(0, i - 5), i)])
            assert hanging.buffer(-0.01).intersection(others).area < 0.01


def test_dots_are_windows_onto_white_over_each_pin(chain):
    spec, j, bodies = chain
    outer = bodies[0].solid
    split = colour_split(outer, j)
    top = _polys_to_shapely(outer.slice(j.top - 0.1).to_polygons())
    under = _polys_to_shapely(split["white"].slice(j.black_from - 0.1).to_polygons())
    for x in (0.0, spec.pitch_mm):
        for r in (0.0, 0.45 * spec.fiducial_mm):
            for a in np.linspace(0, 2 * math.pi, 6, endpoint=False):
                pt = shapely.geometry.Point(x + r * math.cos(a), r * math.sin(a))
                assert not top.contains(pt) and under.contains(pt)
        assert top.contains(shapely.geometry.Point(x + spec.fiducial_mm / 2 + 0.3, 0))


def test_write_files(tmp_path, chain):
    spec, _, _ = chain
    paths = write_printed_chain(tmp_path, spec, JointParams(clearance_mm=0.35), n_pins=3, name="joint")
    for p in paths.values():
        assert p.is_file() and p.stat().st_size > 0
    text = paths["instructions"].read_text()
    j = joint_geometry(spec, JointParams(clearance_mm=0.35))
    assert f"{j.white_from:.1f} mm" in text and f"{j.black_from:.1f} mm" in text
    assert "dots DOWN" in text
    assert paths["drawing"].read_text().startswith("<svg")
