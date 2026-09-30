"""Fiducial designs: the pattern printed around each pin.

A design is the set of white windows cut into the black top layer around a
pin, in millimetres, in a frame centred on the pin with +x along the chain.
The product uses a plain ring sized by data/chain.yaml (`ring_design`);
the others are candidates compared by experiments/fiducial_study.py, and
all are drawn through the same print model in scene.py.

Sizes respect a 0.4 mm FDM nozzle (lines ~0.42 mm wide): every black or
white band is at least 0.8 mm (two lines), and everything fits inside the
8 mm wide link with a black margin around it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from splinewire.chain import ChainSpec

_SHIFT = 4                      # cv2 drawing: 1/16 px sub-pixel precision
_ONE = 1 << _SHIFT


@dataclass(frozen=True)
class Annulus:
    r_out: float
    r_in: float


@dataclass(frozen=True)
class Disc:
    r: float


@dataclass(frozen=True)
class Quadrants:
    """Two opposite quarter discs (+x+y and -x-y): a checker corner at the pin."""
    r: float


@dataclass(frozen=True)
class Marker:
    """An ArUco marker printed inverted (white border on the black link),
    with the pin's index as its id."""
    dictionary: int
    size_mm: float
    bits: int                   # data bits per side (marker is bits + 2 cells wide)


@dataclass(frozen=True)
class Design:
    name: str
    windows: tuple
    outer_mm: float                        # overall diameter (or side of a square)
    circle_edges_mm: tuple[float, ...]     # radii of the circular edges
    note: str = ""


def ring_design(spec: ChainSpec) -> Design:
    ro, ri = spec.ring_outer_mm / 2, spec.ring_inner_mm / 2
    return Design("ring", (Annulus(ro, ri),), spec.ring_outer_mm, (ro, ri),
                  f"white ring {spec.ring_outer_mm:g}/{spec.ring_inner_mm:g} mm (current)")


RING6 = Design("ring-6", (Annulus(3.0, 1.2),), 6.0, (3.0, 1.2),
               "white ring 6.0/2.4 mm: same shape, 20% larger")
DOT = Design("dot", (Disc(2.5),), 5.0, (2.5,), "white disc 5.0 mm")
BULLSEYE = Design("bullseye", (Annulus(3.0, 2.2), Annulus(1.4, 0.6)), 6.0, (3.0, 2.2, 1.4, 0.6),
                  "two white rings, 0.8 mm bands (two lines each), 1.2 mm black dot")
RING_X = Design("ring-x", (Annulus(3.1, 2.3), Quadrants(1.5)), 6.2, (3.1, 2.3, 1.5),
                "white ring 6.2/4.6 mm, 0.8 mm black gap, 3.0 mm checker-corner centre")
ARUCO = Design("aruco", (Marker(cv2.aruco.DICT_4X4_50, 5.6, 4),), 5.6, (),
               "inverted 4x4 ArUco, 5.6 mm (0.93 mm cells), id = pin index")
ARUCO_SMALL = Design("aruco-4.8", (Marker(cv2.aruco.DICT_4X4_50, 4.8, 4),), 4.8, (),
                     "inverted 4x4 ArUco, 4.8 mm (0.8 mm cells): a black margin even at a round chain end")


def draw_windows(
    canvas: np.ndarray,
    design: Design,
    pins_mm: np.ndarray,
    to_px,
    px_per_mm: float,
) -> None:
    """Paint the design's white windows (255) around every pin into canvas.

    to_px maps (N, 2) plane mm to (N, 2) canvas pixel coordinates; the
    canvas may be flipped (row index growing as plane y shrinks), which is
    why shapes are drawn as polygons from mapped points.
    """
    tangents = pin_tangents(pins_mm)
    for k, (p, t) in enumerate(zip(pins_mm, tangents)):
        frame = np.array([t, [-t[1], t[0]]])           # rows: local +x, +y in plane coords
        for w in design.windows:
            _draw(canvas, w, k, p, frame, to_px, px_per_mm)


def pin_tangents(pins_mm: np.ndarray) -> np.ndarray:
    """Unit direction of the chain at each pin (bisector of its two links)."""
    d = np.diff(pins_mm, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    t = np.vstack([d[:1], d[:-1] + d[1:], d[-1:]])
    return t / np.linalg.norm(t, axis=1)[:, None]


def marker_cells(marker: Marker, marker_id: int) -> np.ndarray:
    """(n, n) bool: True where the printed (inverted) marker is white, row 0 at the top."""
    n = marker.bits + 2
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(marker.dictionary),
                                        marker_id, n)
    return img == 0


def _draw(canvas, w, index, p, frame, to_px, tau) -> None:
    def poly(local_mm: np.ndarray) -> np.ndarray:
        px = to_px(p + local_mm @ frame)
        return np.round(px * _ONE).astype(np.int32)

    def circle(r: float) -> np.ndarray:
        n = max(64, int(2 * math.pi * r * tau))       # segment length ~ 1 px
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return poly(r * np.c_[np.cos(a), np.sin(a)])

    if isinstance(w, Annulus):
        cv2.fillPoly(canvas, [circle(w.r_out)], 255, cv2.LINE_8, _SHIFT)
        cv2.fillPoly(canvas, [circle(w.r_in)], 0, cv2.LINE_8, _SHIFT)
    elif isinstance(w, Disc):
        cv2.fillPoly(canvas, [circle(w.r)], 255, cv2.LINE_8, _SHIFT)
    elif isinstance(w, Quadrants):
        n = max(16, int(w.r * tau))
        for a0 in (0.0, math.pi):
            a = a0 + np.linspace(0, math.pi / 2, n)
            pts = np.vstack([[0.0, 0.0], w.r * np.c_[np.cos(a), np.sin(a)]])
            cv2.fillPoly(canvas, [poly(pts)], 255, cv2.LINE_8, _SHIFT)
    elif isinstance(w, Marker):
        cells = marker_cells(w, index)
        n = cells.shape[0]
        c = w.size_mm / n
        for i, j in zip(*np.nonzero(cells)):
            x0, y0 = -w.size_mm / 2 + j * c, w.size_mm / 2 - i * c
            sq = np.array([[x0, y0], [x0 + c, y0], [x0 + c, y0 - c], [x0, y0 - c]])
            cv2.fillPoly(canvas, [poly(sq)], 255, cv2.LINE_8, _SHIFT)
    else:  # pragma: no cover
        raise TypeError(f"unknown window shape {w!r}")
