"""Physical chain model: rigid links of fixed pitch joined at pins."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


FIDUCIALS = ("dot", "ring")


@dataclass(frozen=True)
class ChainSpec:
    pitch_mm: float       # pin-to-pin distance
    half_width_mm: float  # pin line to the contact edge of a link
    fiducial_mm: float    # outer diameter of the fiducial centred on each pin
    n_pins: int
    fiducial: str = "dot"         # "dot" (solid light disc) or "ring" (light annulus)
    ring_inner_mm: float = 0.0    # a ring's hole diameter; 0 for a dot

    def __post_init__(self) -> None:
        if self.pitch_mm <= 0:
            raise ValueError(f"pitch_mm must be positive, got {self.pitch_mm}")
        if self.half_width_mm < 0:
            raise ValueError(f"half_width_mm must be >= 0, got {self.half_width_mm}")
        if self.fiducial not in FIDUCIALS:
            raise ValueError(f"fiducial must be one of {FIDUCIALS}, got {self.fiducial!r}")
        if self.fiducial_mm <= 0:
            raise ValueError(f"fiducial_mm must be positive, got {self.fiducial_mm}")
        if self.fiducial == "ring" and not 0 < self.ring_inner_mm < self.fiducial_mm:
            raise ValueError(
                f"need 0 < ring_inner_mm ({self.ring_inner_mm}) "
                f"< fiducial_mm ({self.fiducial_mm}) for a ring"
            )
        if self.fiducial_mm >= self.pitch_mm:
            raise ValueError(
                f"fiducial_mm ({self.fiducial_mm}) must be smaller than "
                f"pitch_mm ({self.pitch_mm}) or neighbouring fiducials touch"
            )
        if self.n_pins < 3:
            raise ValueError(f"n_pins must be >= 3, got {self.n_pins}")


def default_chain_path() -> Path:
    """data/chain.yaml in the repo, or its bundled copy in a packaged app."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / "data" / "chain.yaml"


def load_chain_spec(path: Path) -> ChainSpec:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return spec_from_dict(raw)


def spec_from_dict(raw: dict) -> ChainSpec:
    """ChainSpec from chain.yaml or saved settings. Older files describe a
    ring as ring_outer_mm + ring_inner_mm, with no `fiducial` key."""
    if "fiducial_mm" in raw:
        size, kind = raw["fiducial_mm"], raw.get("fiducial", "dot")
    else:
        size, kind = raw["ring_outer_mm"], raw.get("fiducial", "ring")
    return ChainSpec(
        pitch_mm=float(raw["pitch_mm"]),
        half_width_mm=float(raw["half_width_mm"]),
        fiducial_mm=float(size),
        n_pins=int(raw["n_pins"]),
        fiducial=str(kind),
        ring_inner_mm=float(raw.get("ring_inner_mm") or 0.0) if kind == "ring" else 0.0,
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
