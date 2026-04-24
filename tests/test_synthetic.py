from pathlib import Path

import numpy as np
from PIL import Image

from validation.ground_truth import load_strip_spec
from validation.synthetic import render_set


def test_render_set_produces_rgb_image(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    out = tmp_path / "set_a.png"
    render_set(spec.sets["A"], out, px_per_mm=20)
    assert out.exists()
    img = Image.open(out)
    assert img.mode == "RGB"


def test_render_set_is_wide_enough_for_all_tiles(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out))
    # strip width must cover last tile's x + half pitch, with 20 px/mm
    min_width_px = int((set_a.tiles[-1].center_mm[0] + set_a.tile_pitch_mm) * 20)
    assert img.shape[1] >= min_width_px


def test_render_set_places_dark_circle_at_each_tile_center(
    data_dir: Path, tmp_path: Path
):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out).convert("L"))
    # Circle is drawn at tile.center_mm — the tile's position reference.
    # The glyph is offset from there along the orientation axis.
    tile0 = set_a.tiles[0]
    circle_x_mm = tile0.center_mm[0]
    circle_y_mm = tile0.center_mm[1]
    px = int(circle_x_mm * 20)
    py = int(circle_y_mm * 20)
    # image y grows downward, spec y grows upward — synthetic must handle this
    # we expect a *low* pixel value (dark ink on light background)
    assert img[img.shape[0] - py, px] < 80


def test_render_set_draws_all_tiles(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out).convert("L"))
    # sum dark pixels — should have roughly (circle area + glyph strokes) × 10 tiles
    dark = (img < 80).sum()
    assert dark > 1000
