import pytest

from validation.schemas import (
    DigitDims,
    DetectedTile,
    GroundTruthTile,
    ImageDetection,
    RingDims,
    ScoringResult,
    StripSpec,
)


def test_ground_truth_tile_holds_position_and_id():
    tile = GroundTruthTile(tile_id="0", center_mm=(7.5, 5.5), orientation_deg=0.0)
    assert tile.tile_id == "0"
    assert tile.center_mm == (7.5, 5.5)
    assert tile.orientation_deg == 0.0


def test_ring_dims_holds_outer_and_inner_diameter():
    r = RingDims(outer_diameter_mm=5.00, inner_diameter_mm=1.66)
    assert r.outer_diameter_mm == 5.00
    assert r.inner_diameter_mm == 1.66


def test_digit_dims_holds_size_stroke_and_radius():
    d = DigitDims(width_mm=2.58, height_mm=5.00, stroke_mm=0.86, corner_radius_mm=0.20)
    assert d.width_mm == 2.58
    assert d.corner_radius_mm == 0.20


def _build_spec(count=16, alphabet="0123456789AbCdEF"):
    tiles = [
        GroundTruthTile(tile_id=alphabet[i], center_mm=(7.5 + i * 11.02, 5.5), orientation_deg=0.0)
        for i in range(count)
    ]
    return StripSpec(
        units="mm",
        ring=RingDims(outer_diameter_mm=5.00, inner_diameter_mm=1.66),
        digit=DigitDims(width_mm=2.58, height_mm=5.00, stroke_mm=0.86, corner_radius_mm=0.20),
        ring_to_digit_gap_mm=1.72,
        orientation_deg=0.0,
        margin_mm={"left": 5.0, "right": 5.0, "top": 3.0, "bottom": 3.0},
        tile_pitch_mm=11.02,
        tiles=tiles,
    )


def test_strip_spec_glyph_offset_is_derived():
    spec = _build_spec()
    # 2.5 + 1.72 + 1.29 = 5.51
    assert spec.glyph_offset_mm == pytest.approx(5.51)


def test_strip_spec_strip_width_is_derived():
    spec = _build_spec()
    # 5.0 + 15*11.02 + 5.51 + 1.29 + 5.0 = 184.60
    assert spec.strip_width_mm == pytest.approx(184.60, abs=0.001)


def test_strip_spec_strip_height_is_derived():
    spec = _build_spec()
    # bottom 3 + max(5,5) + top 3 = 11
    assert spec.strip_height_mm == pytest.approx(11.0)


def test_strip_spec_has_count_tiles():
    spec = _build_spec()
    assert len(spec.tiles) == 16


def test_detected_tile_stores_pixel_positions_and_confidence():
    d = DetectedTile(
        tile_id="3",
        circle_xy_px=(120.5, 200.0),
        glyph_xy_px=(160.0, 200.0),
        confidence=0.92,
    )
    assert d.tile_id == "3"
    assert d.confidence == 0.92


def test_image_detection_drops_set_name():
    det = DetectedTile(
        tile_id="0",
        circle_xy_px=(10.0, 10.0),
        glyph_xy_px=(20.0, 10.0),
        confidence=1.0,
    )
    img = ImageDetection(
        image_path="data/photos/strip.png",
        model="claude-opus-4-7",
        prompt_version="v1",
        detections=[det],
        raw_response="...",
        latency_seconds=3.1,
    )
    assert img.model == "claude-opus-4-7"
    assert img.detections[0].tile_id == "0"


def test_scoring_result_captures_mean_and_per_point_errors():
    r = ScoringResult(
        image_path="data/photos/strip.png",
        per_point_error_mm={"0": 0.4, "1": 0.6},
        mean_error_mm=0.5,
        homography_rmse_px=1.2,
    )
    assert r.mean_error_mm == 0.5
    assert r.per_point_error_mm["1"] == 0.6
