from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from validation.fiducial_geometry import strip_primitives
from validation.ground_truth import build_strip_geometry, load_strip_spec
from validation.synthetic import render_png


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
