import numpy as np

from validation.homography import fit_homography
from validation.schemas import (
    DetectedTile,
    DigitDims,
    GroundTruthTile,
    ImageDetection,
    RingDims,
    ScoringResult,
    StripSpec,
)
from validation.scoring import score_detection


def _make_detection(tile_id: str, px: tuple[float, float]) -> DetectedTile:
    return DetectedTile(
        tile_id=tile_id,
        circle_xy_px=px,
        glyph_xy_px=None,
        confidence=1.0,
    )


def _spec_for(tiles: list[GroundTruthTile]) -> StripSpec:
    """Build a minimal StripSpec wrapping the given tiles. Geometry values
    are the canonical ring+digit dimensions from data/test-strip-spec.yaml,
    but the tests below use only spec.tiles and spec.glyph_offset_mm — they
    don't depend on the surrounding strip layout."""
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


def _make_img(detections: list[DetectedTile]) -> ImageDetection:
    return ImageDetection(
        image_path="x", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )


def test_score_perfect_detection_has_zero_mean_error():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert result.mean_error_mm < 1e-6


def test_score_reports_per_point_error_when_detection_is_offset():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
        GroundTruthTile("4", (5.0, 5.0), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("4", (55.0, 55.0)),  # would be mm (5.5, 5.5) — 0.5 mm off
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert "4" in result.per_point_error_mm
    assert 0.2 < result.per_point_error_mm["4"] < 1.0
    assert result.mean_error_mm > 0.0


def test_score_only_uses_tiles_detected_and_in_truth():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("99", (0.0, 0.0)),  # bogus detection not in truth
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert "99" not in result.per_point_error_mm
