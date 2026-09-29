import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
shapely = pytest.importorskip("shapely")

from splinewire.plaque import BLACK_MM, PLATE_MM, plaque_geometry, plaque_meshes, write_plaque
from splinewire.synthetic import circle_wrap_pins, s_curve_pins


@pytest.mark.parametrize("shape", ["s-curve", "pipe", "cove"])
def test_meshes_are_printable(spec, shape):
    pins = s_curve_pins(spec) if shape == "s-curve" else circle_wrap_pins(spec, 30.0, concave=shape == "cove")
    meshes = plaque_meshes(plaque_geometry(pins, spec))
    for key in ("white", "black"):
        assert meshes[key].is_watertight
    assert meshes["white"].bounds[:, 2].tolist() == pytest.approx([0.0, PLATE_MM])
    assert meshes["black"].bounds[:, 2].tolist() == pytest.approx([PLATE_MM, PLATE_MM + BLACK_MM])
    # black sits inside the plate outline
    assert np.all(meshes["black"].bounds[0, :2] > meshes["white"].bounds[0, :2])
    assert np.all(meshes["black"].bounds[1, :2] < meshes["white"].bounds[1, :2])


def test_each_pin_has_a_white_ring_window_with_a_black_center(spec):
    from shapely.geometry import Point

    pins = s_curve_pins(spec)
    pattern = plaque_geometry(pins, spec).pattern
    band = (spec.ring_outer_mm + spec.ring_inner_mm) / 4     # middle of the ring band
    for p in pins:
        assert pattern.contains(Point(p))                     # black center disc
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            q = p + band * np.array([np.cos(angle), np.sin(angle)])
            assert not pattern.contains(Point(q))             # white window
        assert pattern.contains(Point(p + [spec.ring_outer_mm / 2 + 0.3, 0]))  # black link around it


def test_scale_bar_is_50_mm(spec):
    pattern = plaque_geometry(s_curve_pins(spec), spec).pattern
    bars = [g for g in pattern.geoms if abs((g.bounds[2] - g.bounds[0]) - 50.0) < 1e-6]
    assert len(bars) == 1


def test_write_plaque_files(spec, tmp_path):
    paths = write_plaque(tmp_path, "s-curve", s_curve_pins(spec), spec)
    for p in paths.values():
        assert p.is_file() and p.stat().st_size > 0
    text = paths["instructions"].read_text()
    assert f"{PLATE_MM:.1f} mm" in text and "WHITE" in text and "BLACK" in text
