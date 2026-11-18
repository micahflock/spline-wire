"""Physical chain model: rigid links of fixed pitch joined at pins."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


FIDUCIALS = ("dot", "ring")
# What touches the target: "edge", a link's straight edge between round
# ends, or "round", a round sleeve about every pin with the link between
# them narrower (the snap-fit chain).
CONTACTS = ("edge", "round")


@dataclass(frozen=True)
class ChainSpec:
    pitch_mm: float       # pin-to-pin distance
    half_width_mm: float  # pin line to the contact edge of a link (round contact: the sleeve's radius)
    fiducial_mm: float    # outer diameter of the fiducial centred on each pin
    n_pins: int
    fiducial: str = "dot"         # "dot" (solid light disc) or "ring" (light annulus)
    ring_inner_mm: float = 0.0    # a ring's hole diameter; 0 for a dot
    max_bend_deg: float = 80.0    # sharpest turn at a pin the ordering accepts, as seen in the photo
    contact: str = "edge"         # what touches the target: see CONTACTS

    def __post_init__(self) -> None:
        if self.pitch_mm <= 0:
            raise ValueError(f"pitch_mm must be positive, got {self.pitch_mm}")
        if self.half_width_mm < 0:
            raise ValueError(f"half_width_mm must be >= 0, got {self.half_width_mm}")
        if self.fiducial not in FIDUCIALS:
            raise ValueError(f"fiducial must be one of {FIDUCIALS}, got {self.fiducial!r}")
        if self.contact not in CONTACTS:
            raise ValueError(f"contact must be one of {CONTACTS}, got {self.contact!r}")
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
        if not 30.0 <= self.max_bend_deg <= 150.0:
            raise ValueError(f"max_bend_deg must be between 30 and 150, got {self.max_bend_deg}")


# The chain everything starts with: the CLI's measure and synth, the app, the
# web pages and their tests (the owner's choice, 2026-11-18: the snap-fit
# chain is the one being tested from now on). data/chain.yaml stays the
# O-ring dot chain, which the CAD generators, the printed test plaque, the
# realistic scenes, the benchmark and the regression tests are built around:
# oring_chain_path().
DEFAULT_PRESET = "snap"


def _data_dir() -> Path:
    """data/ in the repo, or its bundled copy in a packaged app."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "data"


def oring_chain_path() -> Path:
    """data/chain.yaml: the O-ring dot chain."""
    return _data_dir() / "chain.yaml"


def default_chain_path() -> Path:
    """The default chain's file (DEFAULT_PRESET: the snap-fit chain)."""
    return chain_presets()[DEFAULT_PRESET]


def chain_presets() -> dict[str, Path]:
    """The chains that ship with the app, by name, the default first:
    data/chain-<name>.yaml is <name>, data/chain.yaml is "oring"."""
    folder = _data_dir()
    found = {p.stem[len("chain-"):]: p for p in sorted(folder.glob("chain-*.yaml"))}
    found["oring"] = oring_chain_path()
    return {DEFAULT_PRESET: found.pop(DEFAULT_PRESET), **found}


def load_chain_spec(path: Path) -> ChainSpec:
    """The chain a YAML file describes. Raises ValueError if the file is not
    YAML settings or a value is out of range, KeyError if one is missing."""
    with Path(path).open("r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as err:
            raise ValueError(f"{Path(path).name} is not valid YAML: {err}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{Path(path).name} must hold the chain's settings (pitch_mm: 10.0, ...)")
    return spec_from_dict(raw)


def spec_from_dict(raw: dict) -> ChainSpec:
    """ChainSpec from chain.yaml or saved settings. Older files describe a
    ring as ring_outer_mm + ring_inner_mm, with no `fiducial` key."""
    if "fiducial_mm" in raw or "ring_outer_mm" not in raw:     # with neither, it is fiducial_mm that is missing
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
        max_bend_deg=float(raw.get("max_bend_deg") or ChainSpec.max_bend_deg),
        contact=str(raw.get("contact") or "edge"),
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
