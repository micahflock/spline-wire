from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FiducialType(str, Enum):
    CIRCLE_GLYPH = "circle_glyph"
    ARUCO_4X4 = "aruco_4x4"


@dataclass(frozen=True)
class GroundTruthTile:
    tile_id: str
    center_mm: tuple[float, float]
    orientation_deg: float


@dataclass(frozen=True)
class TileSet:
    name: str
    fiducial_type: FiducialType
    tile_face_mm: float
    tile_pitch_mm: float
    tiles: list[GroundTruthTile]


@dataclass(frozen=True)
class StripSpec:
    units: str
    sets: dict[str, TileSet]


@dataclass(frozen=True)
class DetectedTile:
    tile_id: str
    circle_xy_px: tuple[float, float]
    glyph_xy_px: tuple[float, float] | None
    confidence: float


@dataclass
class ImageDetection:
    image_path: str
    set_name: str
    model: str
    prompt_version: str
    detections: list[DetectedTile]
    raw_response: str
    latency_seconds: float


@dataclass
class ScoringResult:
    image_path: str
    per_point_error_mm: dict[str, float]
    mean_error_mm: float
    homography_rmse_px: float
