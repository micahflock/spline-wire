"""What the real-world tests measure against (docs/real-world-tests.md).

Two kinds of target, each with a truth the measurement is scored against:

- **Contact gauges**: printed blocks whose edge the chain is wrapped round
  or pressed into. Their truth is the edge's shape (arcs and lines), and the
  chain's contact points should lie on it.
- **Fixed-chain plaques**: the chain's own link, posed and fused into one
  rigid print. Their truth is the pins' positions.

Each target has a name ("convex-r40", "plaque-s"), which the test page and
`splinewire score` use to find its truth, and a JSON form for the page.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from splinewire.chain import ChainSpec, pins_from_turns

GAUGE_SCHEMA = "spline-wire/gauge@1"
PLAQUE_SCHEMA = "spline-wire/plaque-pose@1"


@dataclass(frozen=True)
class Arc:
    center: tuple[float, float]
    radius: float
    start_deg: float
    sweep_deg: float            # counter-clockwise positive

    @property
    def length(self) -> float:
        return abs(math.radians(self.sweep_deg)) * self.radius

    def at(self, s: np.ndarray) -> np.ndarray:
        """Points at arc lengths s from the start."""
        a = math.radians(self.start_deg) + np.sign(self.sweep_deg) * np.asarray(s) / self.radius
        return np.c_[self.center[0] + self.radius * np.cos(a), self.center[1] + self.radius * np.sin(a)]

    def end(self) -> tuple[np.ndarray, float]:
        """End point and the heading there (degrees)."""
        p = self.at(np.array([self.length]))[0]
        heading = self.start_deg + self.sweep_deg + (90.0 if self.sweep_deg > 0 else -90.0)
        return p, heading

    def to_json(self) -> dict:
        return {"arc": {"center": list(self.center), "radius": self.radius,
                        "start_deg": self.start_deg, "sweep_deg": self.sweep_deg}}


@dataclass(frozen=True)
class Line:
    start: tuple[float, float]
    end_: tuple[float, float]

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end_)

    def at(self, s: np.ndarray) -> np.ndarray:
        a, b = np.asarray(self.start), np.asarray(self.end_)
        t = np.asarray(s)[:, None] / self.length
        return a + t * (b - a)

    def to_json(self) -> dict:
        return {"line": {"start": list(self.start), "end": list(self.end_)}}


Segment = Arc | Line


@dataclass(frozen=True)
class Gauge:
    """A contact gauge: the edge the chain touches, in mm.

    radius_mm and concave are set for a round gauge (one arc), where the
    score also fits a circle. A closed gauge's edge is a full circle.
    """
    name: str
    label: str
    segments: tuple[Segment, ...]
    radius_mm: float | None = None
    concave: bool | None = None

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def closed(self) -> bool:
        return (len(self.segments) == 1 and isinstance(self.segments[0], Arc)
                and abs(abs(self.segments[0].sweep_deg) - 360.0) < 1e-9)

    def edge(self, spacing_mm: float = 0.02) -> np.ndarray:
        """Dense points along the edge, in order (spacing at most spacing_mm)."""
        parts = []
        for seg in self.segments:
            n = max(2, int(math.ceil(seg.length / spacing_mm)) + 1)
            parts.append(seg.at(np.linspace(0.0, seg.length, n)))
        return np.vstack(parts)

    def to_json(self) -> dict:
        doc = {"schema": GAUGE_SCHEMA, "name": self.name, "label": self.label, "units": "mm",
               "closed": self.closed, "segments": [s.to_json() for s in self.segments]}
        if self.radius_mm is not None:
            doc["radius_mm"] = self.radius_mm
            doc["concave"] = self.concave
        return doc


def _round(name: str, radius: float, concave: bool, sweep: float) -> Gauge:
    kind = "Concave" if concave else "Convex"
    span = "full circle" if sweep >= 360 else f"{sweep:g}° arc"
    return Gauge(name=name, label=f"{kind} R{radius:g} ({span})",
                 segments=(Arc((0.0, 0.0), radius, -sweep / 2 if sweep < 360 else 0.0, sweep),),
                 radius_mm=radius, concave=concave)


def _s_gauge(radius: float, sweep: float) -> Gauge:
    """An S: an arc bending one way into an arc bending the other, tangent
    where they meet (the inflection)."""
    first = Arc((0.0, 0.0), radius, -90.0 - sweep, sweep)          # ends at (0, -radius), heading +x
    p, _ = first.end()
    second = Arc((float(p[0]), float(p[1]) - radius), radius, 90.0, -sweep)
    return Gauge(name=f"s-r{radius:g}", label=f"S, R{radius:g} each way ({sweep:g}° + {sweep:g}°)",
                 segments=(first, second))


# The gauges of docs/real-world-tests.md: each covers the arc a 13-pin chain
# of 13 mm pitch wraps on it, with a margin, so the whole chain is in contact.
GAUGES: dict[str, Gauge] = {g.name: g for g in (
    _round("convex-r25", 25.0, False, 360.0),
    _round("convex-r40", 40.0, False, 360.0),
    _round("convex-r60", 60.0, False, 180.0),
    _round("convex-r100", 100.0, False, 110.0),
    _round("concave-r40", 40.0, True, 360.0),
    _round("concave-r60", 60.0, True, 180.0),
    _round("concave-r100", 100.0, True, 110.0),
    _s_gauge(60.0, 90.0),
    Gauge(name="straight", label="Straight edge, 200 mm", segments=(Line((0.0, 0.0), (200.0, 0.0)),)),
)}


# ---- fixed-chain plaques ----

PLAQUE_POSES = {
    "plaque-s": "S: bends ease in to about 45° and out again, then the same the other way",
    "plaque-tight": "Tight: four sharp bends of 95–100°, alternating, joined by straight links",
}


def plaque_turns(pose: str, n_pins: int) -> np.ndarray:
    """The joint turns (degrees, one per interior pin) of a plaque pose."""
    n = n_pins - 2
    if n < 1:
        raise ValueError(f"a plaque needs at least 3 pins, got {n_pins}")
    if pose == "plaque-s":
        # One full sine period over the joints: an S with its inflection in the middle.
        return 45.0 * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
    if pose == "plaque-tight":
        bends = (100.0, -95.0, 98.0, -100.0)
        turns = np.zeros(n)
        for k, i in enumerate(range(1, n, 3)):
            turns[i] = bends[k % len(bends)]
        return turns
    raise ValueError(f"no plaque pose {pose!r}; there are {', '.join(PLAQUE_POSES)}")


def plaque_pins(pose: str, spec: ChainSpec, n_pins: int | None = None) -> np.ndarray:
    """Pin positions (mm) of a fixed-chain plaque in this pose."""
    n = n_pins or spec.n_pins
    return pins_from_turns(spec.pitch_mm, np.radians(plaque_turns(pose, n)))


def plaque_json(pose: str, spec: ChainSpec, n_pins: int | None = None) -> dict:
    pins = plaque_pins(pose, spec, n_pins)
    return {"schema": PLAQUE_SCHEMA, "name": pose, "label": PLAQUE_POSES[pose], "units": "mm",
            "pitch_mm": spec.pitch_mm,
            "pin_points": [[round(float(x), 6), round(float(y), 6)] for x, y in pins]}


def target_names() -> list[str]:
    return [*GAUGES, *PLAQUE_POSES]


def targets_json(spec: ChainSpec) -> dict:
    """Every target, for the test page: the gauges as shapes, the plaque
    poses as their joint turns (the page has the pin count)."""
    return {
        "gauges": [g.to_json() for g in GAUGES.values()],
        "plaques": [{"name": p, "label": label, "pitch_mm": spec.pitch_mm}
                    for p, label in PLAQUE_POSES.items()],
    }


def chain_on_gauge(gauge: Gauge, spec: ChainSpec, chain_side: int, start_mm: float = 0.0,
                   n_pins: int | None = None) -> np.ndarray:
    """Pins (mm) of a chain with round contact lying against a gauge's edge.

    Each pin's sleeve (radius spec.half_width_mm) touches the edge, so the
    pins lie on the edge's offset curve, one pitch apart. chain_side: +1 if
    the chain is on the left of the edge as it runs (inside a counter-
    clockwise hollow), -1 on its right (round the outside of a disc).
    start_mm: where the first pin is, as arc length along the edge.
    """
    n = n_pins or spec.n_pins
    edge = gauge.edge(0.01)
    d = np.gradient(edge, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    path = edge + chain_side * spec.half_width_mm * np.c_[-d[:, 1], d[:, 0]]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if gauge.closed:                     # go round as often as the chain needs
        laps = int(np.ceil((start_mm + n * spec.pitch_mm) / s[-1])) + 1
        path = np.vstack([path] + [path[1:]] * laps)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    k = int(np.searchsorted(s, start_mm))
    pins = [path[k]]
    while len(pins) < n:
        p = pins[-1]
        far = np.linalg.norm(path[k:] - p, axis=1) >= spec.pitch_mm
        if not far.any():
            raise ValueError(f"{gauge.name}: the chain runs off the gauge's end after {len(pins)} pins")
        j = k + int(np.argmax(far))
        a, b = path[j - 1], path[j]      # the pitch circle about p crosses this segment
        u = b - a
        qa, qb, qc = u @ u, 2 * u @ (a - p), (a - p) @ (a - p) - spec.pitch_mm ** 2
        t = (-qb + math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
        pins.append(a + t * u)
        k = j
    return np.array(pins)
