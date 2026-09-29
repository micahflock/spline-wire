"""Physical chain model: rigid links of fixed pitch joined at pins."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ChainSpec:
    pitch_mm: float       # pin-to-pin distance
    half_width_mm: float  # pin line to the contact edge of a link
    ring_outer_mm: float  # ring fiducial centered on each pin
    ring_inner_mm: float
    n_pins: int

    def __post_init__(self) -> None:
        if self.pitch_mm <= 0:
            raise ValueError(f"pitch_mm must be positive, got {self.pitch_mm}")
        if self.half_width_mm < 0:
            raise ValueError(f"half_width_mm must be >= 0, got {self.half_width_mm}")
        if not 0 < self.ring_inner_mm < self.ring_outer_mm:
            raise ValueError(
                f"need 0 < ring_inner_mm ({self.ring_inner_mm}) "
                f"< ring_outer_mm ({self.ring_outer_mm})"
            )
        if self.ring_outer_mm >= self.pitch_mm:
            raise ValueError(
                f"ring_outer_mm ({self.ring_outer_mm}) must be smaller than "
                f"pitch_mm ({self.pitch_mm}) or neighbouring rings touch"
            )
        if self.n_pins < 3:
            raise ValueError(f"n_pins must be >= 3, got {self.n_pins}")


def load_chain_spec(path: Path) -> ChainSpec:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ChainSpec(
        pitch_mm=float(raw["pitch_mm"]),
        half_width_mm=float(raw["half_width_mm"]),
        ring_outer_mm=float(raw["ring_outer_mm"]),
        ring_inner_mm=float(raw["ring_inner_mm"]),
        n_pins=int(raw["n_pins"]),
    )


def pins_from_turns(
    pitch_mm: float,
    turns_rad: np.ndarray,
    start_mm: tuple[float, float] = (0.0, 0.0),
    heading_rad: float = 0.0,
) -> np.ndarray:
    """Pin positions of a chain posed by its joint angles.

    turns_rad[k] is the signed turn (counter-clockwise positive) at interior
    pin k+1, so a chain with N pins takes N-2 turns.
    """
    headings = heading_rad + np.concatenate([[0.0], np.cumsum(turns_rad)])
    steps = pitch_mm * np.c_[np.cos(headings), np.sin(headings)]
    return np.vstack([start_mm, start_mm + np.cumsum(steps, axis=0)])


def link_lengths(pins_mm: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(pins_mm, axis=0), axis=1)
