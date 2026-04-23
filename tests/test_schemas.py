from validation.schemas import (
    FiducialType,
    GroundTruthTile,
    TileSet,
    StripSpec,
    DetectedTile,
    ImageDetection,
    ScoringResult,
)


def test_ground_truth_tile_holds_position_and_id():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    assert tile.tile_id == "0"
    assert tile.center_mm == (10.0, 10.0)
    assert tile.orientation_deg == 90.0


def test_tile_set_has_name_type_and_tiles():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    ts = TileSet(
        name="A",
        fiducial_type=FiducialType.CIRCLE_GLYPH,
        tile_face_mm=8.0,
        tile_pitch_mm=10.0,
        tiles=[tile],
    )
    assert ts.name == "A"
    assert ts.fiducial_type is FiducialType.CIRCLE_GLYPH
    assert len(ts.tiles) == 1


def test_strip_spec_indexes_sets_by_name():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    ts = TileSet(
        name="A",
        fiducial_type=FiducialType.CIRCLE_GLYPH,
        tile_face_mm=8.0,
        tile_pitch_mm=10.0,
        tiles=[tile],
    )
    spec = StripSpec(units="mm", sets={"A": ts})
    assert spec.sets["A"].tiles[0].tile_id == "0"


def test_detected_tile_stores_pixel_positions_and_confidence():
    d = DetectedTile(
        tile_id="3",
        circle_xy_px=(120.5, 200.0),
        glyph_xy_px=(160.0, 200.0),
        confidence=0.92,
    )
    assert d.tile_id == "3"
    assert d.confidence == 0.92


def test_image_detection_carries_metadata():
    det = DetectedTile(
        tile_id="0",
        circle_xy_px=(10.0, 10.0),
        glyph_xy_px=(20.0, 10.0),
        confidence=1.0,
    )
    img = ImageDetection(
        image_path="data/photos/a_top_day_01.jpg",
        set_name="A",
        model="claude-opus-4-7",
        prompt_version="v1",
        detections=[det],
        raw_response="...",
        latency_seconds=3.1,
    )
    assert img.set_name == "A"
    assert img.model == "claude-opus-4-7"
    assert img.detections[0].tile_id == "0"


def test_scoring_result_captures_mean_and_per_point_errors():
    r = ScoringResult(
        image_path="data/photos/a_top_day_01.jpg",
        per_point_error_mm={"0": 0.4, "1": 0.6},
        mean_error_mm=0.5,
        homography_rmse_px=1.2,
    )
    assert r.mean_error_mm == 0.5
    assert r.per_point_error_mm["1"] == 0.6
