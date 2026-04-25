from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Annulus:
    """Ring fiducial: filled outer disc with a concentric inner cutout."""
    center_mm: tuple[float, float]
    outer_radius_mm: float
    inner_radius_mm: float


@dataclass(frozen=True)
class Polygon:
    """A planar polygon in mm, possibly with holes (interiors)."""
    exterior_mm: list[tuple[float, float]]
    interiors_mm: list[list[tuple[float, float]]]


# 7-segment glyph layout. Each entry maps a segment label to a function
# (W, H, S) -> (x0, y0, x1, y1) returning the segment's bounding box in
# mm relative to the digit's bottom-left corner.
#
# Vertical segments split at y=H/2 (not y=H/2 ± S/2) so f+e and b+c fuse
# seamlessly when g is unlit. See the spec ("7-segment glyph layout") for
# the rationale.
SegmentBox = Callable[[float, float, float], tuple[float, float, float, float]]

SEGMENT_LAYOUT: dict[str, SegmentBox] = {
    "a": lambda W, H, S: (0.0,   H - S,         W,   H),                # top horizontal
    "b": lambda W, H, S: (W - S, H / 2.0,       W,   H),                # upper right vertical
    "c": lambda W, H, S: (W - S, 0.0,           W,   H / 2.0),          # lower right vertical
    "d": lambda W, H, S: (0.0,   0.0,           W,   S),                # bottom horizontal
    "e": lambda W, H, S: (0.0,   0.0,           S,   H / 2.0),          # lower left vertical
    "f": lambda W, H, S: (0.0,   H / 2.0,       S,   H),                # upper left vertical
    "g": lambda W, H, S: (0.0,   (H - S) / 2.0, W,   (H + S) / 2.0),    # middle horizontal
}


# Lit segments per character (mixed-case hex; standard 7-seg convention).
SEGMENTS_FOR: dict[str, str] = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    "A": "abcefg",
    "b": "cdefg",
    "C": "adef",
    "d": "bcdeg",
    "E": "adefg",
    "F": "aefg",
}
