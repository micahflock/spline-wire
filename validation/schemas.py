from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruthTile:
    tile_id: str
    center_mm: tuple[float, float]
    orientation_deg: float


@dataclass(frozen=True)
class RingDims:
    outer_diameter_mm: float
    inner_diameter_mm: float


@dataclass(frozen=True)
class DigitDims:
    width_mm: float
    height_mm: float
    stroke_mm: float
    corner_radius_mm: float


@dataclass(frozen=True)
class StripSpec:
    units: str
    ring: RingDims
    digit: DigitDims
    ring_to_digit_gap_mm: float
    orientation_deg: float
    margin_mm: dict[str, float]   # keys: left, right, top, bottom
    tile_pitch_mm: float
    tiles: list[GroundTruthTile]

    @property
    def glyph_offset_mm(self) -> float:
        return (
            self.ring.outer_diameter_mm / 2.0
            + self.ring_to_digit_gap_mm
            + self.digit.width_mm / 2.0
        )

    @property
    def strip_width_mm(self) -> float:
        # margin.left + (count-1)*pitch + outer_diameter + gap + digit.width + margin.right
        count = len(self.tiles)
        return (
            self.margin_mm["left"]
            + (count - 1) * self.tile_pitch_mm
            + self.ring.outer_diameter_mm
            + self.ring_to_digit_gap_mm
            + self.digit.width_mm
            + self.margin_mm["right"]
        )

    @property
    def strip_height_mm(self) -> float:
        tile_h = max(self.ring.outer_diameter_mm, self.digit.height_mm)
        return self.margin_mm["bottom"] + tile_h + self.margin_mm["top"]


@dataclass(frozen=True)
class DetectedTile:
    tile_id: str
    circle_xy_px: tuple[float, float]
    glyph_xy_px: tuple[float, float] | None
    confidence: float


@dataclass
class ImageDetection:
    image_path: str
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
