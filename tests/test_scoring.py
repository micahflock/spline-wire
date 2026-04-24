import numpy as np

from validation.homography import fit_homography
from validation.schemas import (
    DetectedTile,
    GroundTruthTile,
    ImageDetection,
    ScoringResult,
)
from validation.scoring import score_detection


def _make_detection(tile_id: str, px: tuple[float, float]) -> DetectedTile:
    return DetectedTile(
        tile_id=tile_id,
        circle_xy_px=px,
        glyph_xy_px=None,
        confidence=1.0,
    )


def test_score_perfect_detection_has_zero_mean_error():
    # build 4-point identity mapping at 10 px/mm
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
    ]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles)
    assert result.mean_error_mm < 1e-6


def test_score_reports_per_point_error_when_detection_is_offset():
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    # perfect first 4, then a 5th tile detected 0.5mm off
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("4", (55.0, 55.0)),  # would be mm (5.5, 5.5)
    ]
    tiles_with_5th = tiles + [GroundTruthTile("4", (5.0, 5.0), 90)]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles_with_5th)
    # tile 4 is detected ~0.5mm off in x and y; exact per-point error depends
    # on the least-squares homography fit distributing residuals across all 5
    # points. Verify the error is reported and is non-trivial but bounded.
    assert "4" in result.per_point_error_mm
    assert 0.2 < result.per_point_error_mm["4"] < 1.0
    assert result.mean_error_mm > 0.0


def test_score_only_uses_tiles_detected_and_in_truth():
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("99", (0.0, 0.0)),  # bogus detection not in truth
    ]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles)
    assert "99" not in result.per_point_error_mm
