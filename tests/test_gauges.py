"""The printable contact gauges match the shapes they are scored against."""
import json

import pytest

pytest.importorskip("shapely")
pytest.importorskip("trimesh")

from shapely.geometry import Point  # noqa: E402

from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.cli import main  # noqa: E402
from splinewire.gauges import FOOT_INSET_MM, HEIGHT_MM, chain_side, gauge_mesh, gauge_polygon  # noqa: E402
from splinewire.targets import GAUGES, chain_on_gauge  # noqa: E402


@pytest.mark.parametrize("name", list(GAUGES))
def test_the_printed_edge_is_the_scored_edge_and_the_chain_fits_against_it(name):
    """Every point of the gauge's edge (targets) lies on the printed outline,
    and a chain laid against it touches without any sleeve inside the block."""
    g = GAUGES[name]
    poly = gauge_polygon(g)
    boundary = poly.boundary
    assert max(boundary.distance(Point(p)) for p in g.edge(1.0)) < 0.005
    spec = load_chain_spec(chain_presets()["snap"])
    pins = chain_on_gauge(g, spec, chain_side(g), start_mm=6.0)
    gaps = [poly.distance(Point(p)) for p in pins]
    assert min(gaps) == pytest.approx(spec.half_width_mm, abs=0.005)
    mesh = gauge_mesh(g)
    assert mesh.bounds[1][2] == pytest.approx(HEIGHT_MM)
    assert max(mesh.extents[:2]) < 210            # fits a 220 mm bed


def test_the_foot_is_set_in_from_the_edge():
    mesh = gauge_mesh(GAUGES["convex-r25"])
    bottom = mesh.vertices[mesh.vertices[:, 2] < 1e-9]
    assert abs(bottom[:, :2]).max() == pytest.approx(25.0 - FOOT_INSET_MM, abs=0.01)


def test_test_gauges_command_writes_stls_truth_and_notes(tmp_path, capsys):
    assert main(["test-gauges", "convex-r40", "s-r60", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "gauge-convex-r40.stl").stat().st_size > 0
    doc = json.loads((tmp_path / "gauges.json").read_text())
    assert [g["name"] for g in doc["gauges"]] == ["convex-r40", "s-r60"]
    assert "outside diameter, should be 80.00 mm" in capsys.readouterr().out
    assert main(["test-gauges", "nope", "--out", str(tmp_path)]) == 1
