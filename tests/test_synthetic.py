from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from validation.fiducial_geometry import strip_primitives
from validation.ground_truth import build_strip_geometry, load_strip_spec
from validation.synthetic import render_png, render_svg


PX_PER_MM = 20  # tests run faster at 20 than at 40


@pytest.fixture
def primitives_and_spec(data_dir):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    geom = build_strip_geometry(spec)
    prims = strip_primitives(geom)
    return prims, spec


def test_render_png_produces_rgb_image(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = Image.open(out)
    assert img.mode == "RGB"


def test_render_png_dimensions_match_strip(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out))
    expected_w = int(round(spec.strip_width_mm * PX_PER_MM))
    expected_h = int(round(spec.strip_height_mm * PX_PER_MM))
    assert img.shape[1] == expected_w
    assert img.shape[0] == expected_h


def test_render_png_ring_is_hollow(tmp_path, primitives_and_spec):
    """A pixel at the very center of a ring should be background (the hole),
    while a pixel halfway between center and outer edge should be inked."""
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out).convert("L"))
    h_px = img.shape[0]

    ring_center_mm = spec.tiles[0].center_mm
    cx_px = int(round(ring_center_mm[0] * PX_PER_MM))
    cy_px = h_px - int(round(ring_center_mm[1] * PX_PER_MM))
    # center pixel: light (background, inside the hole)
    assert img[cy_px, cx_px] > 200
    # halfway between center and outer edge (outer radius 2.5 mm)
    half_radius_mm = 1.25
    edge_x_px = cx_px + int(round(half_radius_mm * PX_PER_MM))
    assert img[cy_px, edge_x_px] < 80


def test_render_png_inks_each_ring(tmp_path, primitives_and_spec):
    """Every tile's ring should leave dark pixels in the image."""
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out).convert("L"))
    h_px = img.shape[0]
    # Halfway from ring center toward outer edge along +x.
    half_r_mm = 1.25
    for tile in spec.tiles:
        cx, cy = tile.center_mm
        px = int(round((cx + half_r_mm) * PX_PER_MM))
        py = h_px - int(round(cy * PX_PER_MM))
        assert img[py, px] < 80, f"tile {tile.tile_id}: expected dark pixel at ({px},{py})"


def test_render_svg_is_well_formed(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    # Parses without error.
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag.endswith("svg")


def test_render_svg_dimensions_in_mm_match_strip(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    tree = ET.parse(out)
    root = tree.getroot()
    width_str = root.attrib["width"]
    height_str = root.attrib["height"]
    assert width_str.endswith("mm")
    assert height_str.endswith("mm")
    assert float(width_str[:-2]) == pytest.approx(spec.strip_width_mm)
    assert float(height_str[:-2]) == pytest.approx(spec.strip_height_mm)


def test_render_svg_contains_one_ring_path_and_one_digit_path_per_tile(
    tmp_path, primitives_and_spec
):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    tree = ET.parse(out)
    root = tree.getroot()
    # Both rings and digits are <path> elements (ring uses two-circle path
    # with evenodd; digit uses traced exterior+interiors).
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall("svg:path", ns) + root.findall("path")
    # 16 tiles × 2 paths each = 32
    assert len(paths) == 2 * len(spec.tiles)


def test_render_svg_and_png_use_same_primitives(tmp_path, primitives_and_spec):
    """Cross-backend smoke test: both backends consume the same primitive
    list and produce non-empty output. The deeper agreement between them
    (e.g. ring positions match) is asserted indirectly by the per-backend
    tests above plus the geometry tests in test_fiducial_geometry.py."""
    prims, spec = primitives_and_spec
    png_out = tmp_path / "strip.png"
    svg_out = tmp_path / "strip.svg"
    render_png(prims, png_out, spec, px_per_mm=PX_PER_MM)
    render_svg(prims, svg_out, spec)

    assert png_out.stat().st_size > 0
    assert svg_out.stat().st_size > 0

    # SVG references the first ring's leftmost-edge x coordinate (rx - outer_radius).
    # rx = 7.5, outer_radius = 2.5 → 5.0. render_svg formats with %.4f → "5.0000".
    svg_text = svg_out.read_text(encoding="utf-8")
    rx = spec.tiles[0].center_mm[0]
    leftmost_edge = rx - spec.ring.outer_diameter_mm / 2.0
    assert f"{leftmost_edge:.4f}" in svg_text
