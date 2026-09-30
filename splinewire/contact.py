"""From measured pin positions to points on the target curve.

The chain does not touch the target along its pin line. Each link is a bar
of half-width w with semicircular ends centered on the pins, so the target
touches the chain along an edge w away from the pins:

- Where the chain bends around the object (convex target, e.g. a pipe), each
  link's straight edge is tangent to the target near the link's middle, and
  the pins stand off the curve. Contact point: link midpoint offset by w.
- Where the chain bends away from the object (concave target, e.g. a cove),
  the straight edges bridge across and the rounded link ends touch the
  target. Contact point: pin offset by w along the bisector of its two links.

Contact points are exact for circular arcs and a close approximation for
curves whose radius changes slowly relative to the pitch.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

Side = Literal["inside", "outside"]

# Bends smaller than this (radians) count as straight.
_STRAIGHT_TOL = np.radians(0.5)


def object_sign(pins_mm: np.ndarray, side: Side) -> int:
    """Which side of the pin polyline the object is on: +1 left, -1 right.

    "inside" puts the object on the side the chain curls toward overall
    (a chain wrapped around something); "outside" is the opposite side
    (a chain pressed into a hollow).
    """
    net_turn = float(np.sum(_turns(pins_mm)))
    toward = 1 if net_turn >= 0 else -1
    return toward if side == "inside" else -toward


def contact_points(pins_mm: np.ndarray, half_width_mm: float, sign: int) -> np.ndarray:
    """Points where the target curve touches the chain, in chain order."""
    pins = np.asarray(pins_mm, dtype=float)
    d = np.diff(pins, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    left = np.c_[-d[:, 1], d[:, 0]]           # left normal of each link
    offset = sign * half_width_mm

    # Per pin: does the chain bend away from the object here? End pins take
    # their neighbour's value so an end link behaves like the one next to it.
    turns = _turns(pins) * sign
    away = np.r_[False, turns < -_STRAIGHT_TOL, False]
    if len(away) > 2:
        away[0], away[-1] = away[1], away[-2]

    out: list[np.ndarray] = []
    for i in range(len(pins)):
        if away[i]:
            if i == 0:
                normal = left[0]
            elif i == len(pins) - 1:
                normal = left[-1]
            else:
                normal = left[i - 1] + left[i]
                normal /= np.linalg.norm(normal)
            out.append(pins[i] + offset * normal)
        if i < len(pins) - 1 and not (away[i] and away[i + 1]):
            out.append((pins[i] + pins[i + 1]) / 2 + offset * left[i])
    return np.array(out)


def spline_samples(points_mm: np.ndarray, per_segment: int = 12) -> np.ndarray:
    """Dense samples of a cubic spline through the points (chord-length parameter).

    Not-a-knot ends: natural ends force zero curvature and pull the curve
    ~0.2 mm off a 30 mm arc near its ends.
    """
    pts = np.asarray(points_mm, dtype=float)
    if len(pts) < 3:
        return pts.copy()
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    spline = CubicSpline(s, pts, bc_type="not-a-knot")
    return spline(np.linspace(0.0, s[-1], per_segment * (len(pts) - 1) + 1))


def _turns(pins_mm: np.ndarray) -> np.ndarray:
    """Signed turn angle at each interior pin, counter-clockwise positive."""
    d = np.diff(np.asarray(pins_mm, dtype=float), axis=0)
    a, b = d[:-1], d[1:]
    return np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))
