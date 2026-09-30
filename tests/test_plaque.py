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


def _photo_of_plaque(plaque, camera, table_gray, tau=20.0, ss=3, rng=None):
    """Grayscale photo of the printed plaque lying on a table of the given shade."""
    import cv2

    rng = rng or np.random.default_rng(0)
    x0, y0, x1, y1 = plaque.plate.bounds
    xs = x0 - 5 + np.arange(int((x1 - x0 + 10) * tau)) / tau
    ys = y1 + 5 - np.arange(int((y1 - y0 + 10) * tau)) / tau
    gx, gy = np.meshgrid(xs, ys)
    tex = np.full(gx.shape, float(table_gray))
    tex[shapely.contains_xy(plaque.plate, gx, gy)] = 225.0      # white base
    tex[shapely.contains_xy(plaque.pattern, gx, gy)] = 40.0     # black top
    T = np.array([[1 / tau, 0, xs[0]], [0, -1 / tau, ys[0]], [0, 0, 1]])
    S = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])
    w, h = camera.image_size
    hi = cv2.warpPerspective(tex.astype(np.float32), S @ camera.plane_homography @ T, (w * ss, h * ss),
                             flags=cv2.INTER_LINEAR, borderValue=float(table_gray))
    img = cv2.GaussianBlur(cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA).astype(float), (0, 0), 0.7)
    return np.clip(np.round(img + rng.normal(0, 3.0, img.shape)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape", ["s-curve", "pipe", "cove"])
@pytest.mark.parametrize("table_gray", [25, 128, 225])
def test_trimmed_plaque_measures_on_any_table(spec, shape, table_gray):
    """The plate follows the chain's outline, so the table shows around it:
    dark, mid-grey and white tables must all measure correctly."""
    from splinewire.camera import focal_px_from_35mm, look_at_plane
    from splinewire.pipeline import compare_to_truth, measure

    pins = s_curve_pins(spec) if shape == "s-curve" else circle_wrap_pins(spec, 30.0, concave=shape == "cove")
    plaque = plaque_geometry(pins, spec)
    size = (2000, 1500)
    f = focal_px_from_35mm(26, size)
    target = tuple(np.mean(np.array(plaque.plate.bounds).reshape(2, 2), axis=0))
    cam = look_at_plane(f, size, 200.0, tilt_deg=25.0, tilt_direction_deg=60.0, target_mm=target)
    m = measure(_photo_of_plaque(plaque, cam, table_gray), spec, f)
    assert len(m.order.indices) == len(pins) and not m.order.gaps
    assert compare_to_truth(m.rectification.pins_mm, pins)["max_error_mm"] < 0.1


def test_plate_is_trimmed_to_the_chain(spec):
    """No rectangular slab: the white base hugs the chain and the bar."""
    plaque = plaque_geometry(s_curve_pins(spec), spec)
    from splinewire.plaque import RIM_MM, plaque_volume_cm3
    assert plaque.plate.geom_type == "Polygon"                 # one piece, bar attached
    assert plaque.plate.contains(plaque.pattern.buffer(RIM_MM * 0.9))
    x0, y0, x1, y1 = plaque.plate.bounds
    assert plaque.plate.area < 0.4 * (x1 - x0) * (y1 - y0)
    assert plaque_volume_cm3(plaque) < 4.0
