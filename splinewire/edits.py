"""Hand corrections to what the detector found: pins removed, pins added.

The detector can miss a pin (glare, a smudge) or take something else for one,
and the chain order built on that is then short or wrong. A person looking at
the photo can see which, so the app lets them point at it. Coordinates are
photo pixels, those of the upright image `load_photo` returns.

- remove: a detection to ignore, given by a point on it.
- add: a pin to use, given by a point on it. Clicks are a few pixels off, so
  each is snapped to the fiducial under it (`snap_fiducial`); where there is
  none to see (the dot is under glare) the click itself is used, and the
  result says so. An added pin replaces any detection it lands on, so
  pointing at a detection the ordering left out is how to bring it back.

Added pins skip the plausibility tests that throw out look-alikes (they are
the person's word that this is a pin), but still have to sit about one pitch
from their neighbours to join the chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from splinewire.detect import Fiducial, snap_fiducial

MAX_EDITS = 300


@dataclass(frozen=True)
class Edits:
    add_px: tuple[tuple[float, float], ...] = ()
    remove_px: tuple[tuple[float, float], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add_px or self.remove_px)

    def to_json(self) -> dict:
        return {"add": [list(p) for p in self.add_px], "remove": [list(p) for p in self.remove_px]}

    @classmethod
    def from_json(cls, doc, size: tuple[int, int] | None = None) -> "Edits":
        """Edits from the web page's JSON, checked: it comes from a browser.

        size: the photo's (width, height); points outside it are refused.
        """
        if not isinstance(doc, dict):
            raise ValueError("edits must be an object with add and remove lists")
        return cls(add_px=_points(doc.get("add", []), "add", size),
                   remove_px=_points(doc.get("remove", []), "remove", size))


@dataclass(frozen=True)
class ManualPin:
    """A pin the person added, as it ended up in the measurement."""
    fiducial: int       # index into Measurement.fiducials
    edit: int           # index into Edits.add_px
    snapped: bool       # False: no fiducial was found there, the click itself is the pin


def _points(raw, name: str, size) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) > MAX_EDITS:
        raise ValueError(f"{name} must be a list of at most {MAX_EDITS} points")
    out = []
    for p in raw:
        try:
            x, y = (float(v) for v in p)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: each point is [x, y] in pixels") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{name}: points must be finite numbers")
        if size is not None and not (-1 <= x <= size[0] + 1 and -1 <= y <= size[1] + 1):
            raise ValueError(f"{name}: a point lies outside the photo")
        out.append((round(x, 2), round(y, 2)))
    return tuple(out)


def remove_detections(fiducials: list[Fiducial], edits: Edits) -> list[Fiducial]:
    """The detections left after dropping the one under each remove point."""
    centers = np.array([f.center_px for f in fiducials]) if fiducials else np.zeros((0, 2))
    drop: set[int] = set()
    for p in edits.remove_px:
        left = [i for i in range(len(fiducials)) if i not in drop]      # one detection per point
        i = _detection_at(p, [fiducials[k] for k in left], centers[left])
        if i is not None:
            drop.add(left[i])
    return [f for i, f in enumerate(fiducials) if i not in drop]


def add_pins(
    image: np.ndarray,
    fiducials: list[Fiducial],
    spec,
    edits: Edits,
    chain: list[int],
) -> tuple[list[Fiducial], list[ManualPin]]:
    """`fiducials` with the added pins in place of the detections under them.

    chain: indices into `fiducials` of the pins that already form a chain;
    the nearest of them tell an added pin its size, foreshortening and look.
    Without any, the size comes from how far apart the added pins are.
    Where the pin is: a detection under the click, else what snap_fiducial
    finds, else the click.
    Returns the new list (detections kept first, then the added pins in the
    order of `edits.add_px`) and the added pins.
    """
    if not edits.add_px:
        return list(fiducials), []
    refs = [fiducials[i] for i in (chain or range(len(fiducials)))]
    ref_centers = np.array([f.center_px for f in refs]) if refs else np.zeros((0, 2))
    fallback = None if refs else _from_spacing(edits.add_px, spec)
    centers = np.array([f.center_px for f in fiducials]) if fiducials else np.zeros((0, 2))

    made: list[tuple[int, Fiducial, bool]] = []
    for j, p in enumerate(edits.add_px):
        like = _like(p, refs, ref_centers) if refs else fallback
        # The pin is where a detection already sits under the click, else where
        # a fiducial can be found, else the click itself. Its look is a
        # neighbour's: a person's word that it is a pin outranks how it looks.
        near = _detection_at(p, fiducials, centers)
        if near is not None:
            center, snapped = fiducials[near].center_px, True
        else:
            found = snap_fiducial(image, p, spec, like)
            center, snapped = (found.center_px, True) if found else ((float(p[0]), float(p[1])), False)
        pin = replace(like, center_px=center)
        # two clicks on one pin make one pin
        if any(math.hypot(pin.center_px[0] - q.center_px[0], pin.center_px[1] - q.center_px[1])
               < 0.5 * q.outer_axes_px[1] for _, q, _ in made):
            continue
        made.append((j, pin, snapped))

    kept = [f for f in fiducials
            if not any(math.hypot(f.center_px[0] - q.center_px[0], f.center_px[1] - q.center_px[1])
                       < 0.5 * min(f.outer_axes_px[1], q.outer_axes_px[1]) for _, q, _ in made)]
    manual = [ManualPin(fiducial=len(kept) + n, edit=j, snapped=snapped)
              for n, (j, _, snapped) in enumerate(made)]
    return kept + [q for _, q, _ in made], manual


def _detection_at(p, fiducials: list[Fiducial], centers: np.ndarray) -> int | None:
    """The detection a point is on, if any."""
    if not len(fiducials):
        return None
    dist = np.linalg.norm(centers - p, axis=1)
    i = int(np.argmin(dist))
    return i if dist[i] <= max(0.6 * fiducials[i].outer_axes_px[0], 4.0) else None


def _like(p, refs: list[Fiducial], centers: np.ndarray) -> Fiducial:
    """A fiducial standing in for one at p: the nearest chain pin's shape,
    with the median size, contrast and surround of the nearest few."""
    near = np.argsort(np.linalg.norm(centers - p, axis=1))[:3]
    group = [refs[i] for i in near]
    major = float(np.median([f.outer_axes_px[0] for f in group]))
    minor = float(np.median([f.outer_axes_px[1] for f in group]))
    first = group[0]
    return Fiducial(
        center_px=(float(p[0]), float(p[1])), outer_axes_px=(major, minor),
        inner_axes_px=first.inner_axes_px, angle_deg=first.angle_deg,
        contrast=float(np.median([f.contrast for f in group])), polarity=first.polarity,
        surround=float(np.median([f.surround for f in group])),
    )


def _from_spacing(points, spec) -> Fiducial:
    """With no detections to learn from, a fiducial as big as the spacing of the
    added pins says: neighbours are one pitch apart and the fiducial is a known
    fraction of that."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise ValueError("no fiducial was detected to compare with; "
                         "click at least two pins to say how big they are")
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    d = float(np.median(dist.min(axis=1))) * spec.fiducial_mm / spec.pitch_mm
    inner = d * spec.ring_inner_mm / spec.fiducial_mm if spec.fiducial == "ring" else 0.0
    return Fiducial(center_px=(0.0, 0.0), outer_axes_px=(d, d), inner_axes_px=(inner, inner),
                    angle_deg=0.0, contrast=40.0, polarity=1, surround=40.0)


def pin_to_pitch(
    pins_mm: np.ndarray,
    links: list[tuple[int, int]],
    free: list[int],
    pitch_mm: float,
    sigma_click_mm: float,
    sigma_link_mm: float = 0.05,
) -> np.ndarray:
    """Pins in `free` moved to where the links around them say they are.

    A pin with no fiducial to snap to is only as exact as the click, a few
    pixels, but the chain is not: its links are exactly one pitch long. Each
    free pin moves as little as it must (a click costs sigma_click_mm per mm
    of distance) to make the links to its neighbours pitch long (sigma_link_mm).
    One pin between two known ones lands exactly, on the nearer of the two
    places both links allow; several in a row keep whatever freedom the links
    leave (an elbow can flex) as clicked.
    """
    pins = np.array(pins_mm, dtype=float)
    free = list(free)
    if not free:
        return pins
    index = {k: n for n, k in enumerate(free)}
    start = pins[free].copy()
    touching = [(a, b) for a, b in links if a in index or b in index]

    def put(x: np.ndarray) -> np.ndarray:
        q = pins.copy()
        q[free] = x.reshape(-1, 2)
        return q

    def residuals(x: np.ndarray) -> np.ndarray:
        q = put(x)
        link = [(np.linalg.norm(q[a] - q[b]) - pitch_mm) / sigma_link_mm for a, b in touching]
        return np.r_[link, (x - start.ravel()) / sigma_click_mm]

    return put(least_squares(residuals, start.ravel(), method="lm", max_nfev=200).x)
