from pathlib import Path

import pytest

from validation.ground_truth import load_strip_spec
from validation.schemas import FiducialType


def test_load_strip_spec_parses_three_sets(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.units == "mm"
    assert set(spec.sets.keys()) == {"A", "B", "C"}


def test_set_a_generates_10_tiles_with_correct_pitch(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    assert len(set_a.tiles) == 10
    assert set_a.tiles[0].center_mm == (15.0, 20.0)
    assert set_a.tiles[1].center_mm == (25.0, 20.0)
    assert set_a.tiles[9].center_mm == (105.0, 20.0)
    assert set_a.fiducial_type is FiducialType.CIRCLE_GLYPH


def test_set_a_tile_ids_match_digit_alphabet(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    ids = [t.tile_id for t in set_a.tiles]
    assert ids == list("0123456789")


def test_set_c_uses_aruco_type(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_c = spec.sets["C"]
    assert set_c.fiducial_type is FiducialType.ARUCO_4X4


def test_load_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_strip_spec(tmp_path / "does-not-exist.yaml")
