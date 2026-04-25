from pathlib import Path

import pytest

from validation.ground_truth import build_strip_geometry, load_strip_spec


def test_load_strip_spec_returns_flat_strip_with_16_tiles(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.units == "mm"
    assert len(spec.tiles) == 16
    assert spec.tiles[0].tile_id == "0"
    assert spec.tiles[15].tile_id == "F"


def test_load_strip_spec_first_tile_at_expected_position(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    # margin.left + ring.outer/2 = 5 + 2.5 = 7.5
    # margin.bottom + max(ring.outer, digit.height)/2 = 3 + 2.5 = 5.5
    assert spec.tiles[0].center_mm == pytest.approx((7.5, 5.5))


def test_load_strip_spec_pitch_steps_ring_centers(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.tiles[1].center_mm[0] - spec.tiles[0].center_mm[0] == pytest.approx(11.02)
    assert spec.tiles[15].center_mm[0] == pytest.approx(7.5 + 15 * 11.02)


def test_load_strip_spec_orientation_propagated_to_tiles(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    for tile in spec.tiles:
        assert tile.orientation_deg == 0.0


def test_load_strip_spec_derived_glyph_offset(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.glyph_offset_mm == pytest.approx(5.51)


def test_build_strip_geometry_matches_spec(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    geom = build_strip_geometry(spec)
    assert geom.ring_outer_diameter_mm == spec.ring.outer_diameter_mm
    assert geom.alphabet == "0123456789AbCdEF"
    assert geom.count == 16


def test_load_rejects_bad_alphabet_char(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(alphabet="0123456789AbcdEF"), encoding="utf-8")
    # lowercase 'c' is not in SEGMENTS_FOR (we use uppercase 'C' for hex)
    with pytest.raises(ValueError, match="character 'c'"):
        load_strip_spec(bad_yaml)


def test_load_rejects_alphabet_count_mismatch(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(alphabet="012345"), encoding="utf-8")
    with pytest.raises(ValueError, match="alphabet"):
        load_strip_spec(bad_yaml)


def test_load_rejects_inverted_ring_diameters(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(ring_inner=6.0, ring_outer=5.0), encoding="utf-8")
    with pytest.raises(ValueError, match="inner_diameter"):
        load_strip_spec(bad_yaml)


def test_load_rejects_oversized_corner_radius(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(corner_radius=0.5), encoding="utf-8")
    # corner_radius * 2 > stroke (0.86) → invalid
    with pytest.raises(ValueError, match="corner_radius"):
        load_strip_spec(bad_yaml)


def test_load_rejects_pitch_too_small(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(pitch=8.0), encoding="utf-8")
    # min pitch = 9.30; 8.0 causes tile overlap
    with pytest.raises(ValueError, match="tile_pitch"):
        load_strip_spec(bad_yaml)


def _yaml_with(
    ring_outer=5.00, ring_inner=1.66,
    digit_w=2.58, digit_h=5.00, stroke=0.86, corner_radius=0.20,
    gap=1.72, count=16, alphabet="0123456789AbCdEF",
    pitch=11.02,
):
    return f"""
units: mm
tile:
  ring:
    outer_diameter_mm: {ring_outer}
    inner_diameter_mm: {ring_inner}
  digit:
    width_mm: {digit_w}
    height_mm: {digit_h}
    stroke_mm: {stroke}
    corner_radius_mm: {corner_radius}
  ring_to_digit_gap_mm: {gap}
strip:
  count: {count}
  alphabet: "{alphabet}"
  tile_pitch_mm: {pitch}
  orientation_deg: 0.0
  margin_mm:
    left: 5.0
    right: 5.0
    top: 3.0
    bottom: 3.0
"""
