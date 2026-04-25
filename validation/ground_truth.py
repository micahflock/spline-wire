from __future__ import annotations

from pathlib import Path

import yaml

from validation.fiducial_geometry import SEGMENTS_FOR, StripGeometry
from validation.schemas import (
    DigitDims,
    GroundTruthTile,
    RingDims,
    StripSpec,
)


def load_strip_spec(path: Path) -> StripSpec:
    """Load and validate the test strip YAML spec."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ring = RingDims(
        outer_diameter_mm=raw["tile"]["ring"]["outer_diameter_mm"],
        inner_diameter_mm=raw["tile"]["ring"]["inner_diameter_mm"],
    )
    digit = DigitDims(
        width_mm=raw["tile"]["digit"]["width_mm"],
        height_mm=raw["tile"]["digit"]["height_mm"],
        stroke_mm=raw["tile"]["digit"]["stroke_mm"],
        corner_radius_mm=raw["tile"]["digit"]["corner_radius_mm"],
    )
    ring_to_digit_gap_mm = raw["tile"]["ring_to_digit_gap_mm"]

    strip = raw["strip"]
    count = strip["count"]
    alphabet = strip["alphabet"]
    pitch = strip["tile_pitch_mm"]
    orientation = strip["orientation_deg"]
    margin = {k: float(v) for k, v in strip["margin_mm"].items()}

    _validate_ring(ring)
    _validate_digit(digit)
    _validate_alphabet(alphabet, count)
    _validate_pitch(pitch, ring, digit, ring_to_digit_gap_mm)

    first_x = margin["left"] + ring.outer_diameter_mm / 2.0
    tile_y = margin["bottom"] + max(ring.outer_diameter_mm, digit.height_mm) / 2.0
    tiles = [
        GroundTruthTile(
            tile_id=alphabet[i],
            center_mm=(first_x + i * pitch, tile_y),
            orientation_deg=orientation,
        )
        for i in range(count)
    ]

    return StripSpec(
        units=raw["units"],
        ring=ring,
        digit=digit,
        ring_to_digit_gap_mm=ring_to_digit_gap_mm,
        orientation_deg=orientation,
        margin_mm=margin,
        tile_pitch_mm=pitch,
        tiles=tiles,
    )


def build_strip_geometry(spec: StripSpec) -> StripGeometry:
    """Build the geometry-layer view from a StripSpec for renderers."""
    return StripGeometry(
        ring_outer_diameter_mm=spec.ring.outer_diameter_mm,
        ring_inner_diameter_mm=spec.ring.inner_diameter_mm,
        digit_width_mm=spec.digit.width_mm,
        digit_height_mm=spec.digit.height_mm,
        digit_stroke_mm=spec.digit.stroke_mm,
        digit_corner_radius_mm=spec.digit.corner_radius_mm,
        ring_to_digit_gap_mm=spec.ring_to_digit_gap_mm,
        tile_pitch_mm=spec.tile_pitch_mm,
        margin_left_mm=spec.margin_mm["left"],
        margin_bottom_mm=spec.margin_mm["bottom"],
        count=len(spec.tiles),
        alphabet="".join(t.tile_id for t in spec.tiles),
    )


def _validate_ring(ring: RingDims) -> None:
    if ring.inner_diameter_mm >= ring.outer_diameter_mm:
        raise ValueError(
            f"ring inner_diameter_mm ({ring.inner_diameter_mm}) must be less than "
            f"outer_diameter_mm ({ring.outer_diameter_mm})"
        )


def _validate_digit(digit: DigitDims) -> None:
    if digit.corner_radius_mm * 2 > digit.stroke_mm:
        raise ValueError(
            f"digit corner_radius_mm ({digit.corner_radius_mm}) * 2 must not exceed "
            f"stroke_mm ({digit.stroke_mm}); rounding would eat the stroke"
        )
    if digit.width_mm < 3 * digit.stroke_mm:
        raise ValueError(
            f"digit width_mm ({digit.width_mm}) must be at least 3 * stroke_mm "
            f"({3 * digit.stroke_mm}) to fit segments horizontally"
        )
    if digit.height_mm < 5 * digit.stroke_mm:
        raise ValueError(
            f"digit height_mm ({digit.height_mm}) must be at least 5 * stroke_mm "
            f"({5 * digit.stroke_mm}) to fit segments vertically"
        )


def _validate_alphabet(alphabet: str, count: int) -> None:
    if len(alphabet) != count:
        raise ValueError(
            f"alphabet length {len(alphabet)} does not match strip count {count}"
        )
    for ch in alphabet:
        if ch not in SEGMENTS_FOR:
            raise ValueError(
                f"alphabet contains character {ch!r} with no 7-segment glyph defined"
            )


def _validate_pitch(
    pitch: float,
    ring: RingDims,
    digit: DigitDims,
    gap: float,
) -> None:
    min_pitch = ring.outer_diameter_mm + gap + digit.width_mm
    if pitch < min_pitch:
        raise ValueError(
            f"tile_pitch_mm ({pitch}) is below minimum {min_pitch:.3f} (tiles would overlap)"
        )
