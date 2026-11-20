"""Order and rectify points found in a video frame, for the live camera overlay.

The overlay (docs/web-build.md, step 5) finds bright blobs in each video
frame with a light detector in the page (web/src/overlay/blobs.ts). Which of
them are the chain, in what order, and how the camera is tilted is this
module's job: the same `order` and `rectify` that measure a photo, run on a
few dozen points, a few times a second.

Nothing here is a measurement. The numbers coach the person (pins found,
tilt) and the still taken at the end goes through the full pipeline
(decision D7).
"""
from __future__ import annotations

import numpy as np

from splinewire.chain import ChainSpec, spec_from_dict
from splinewire.order import order_chain
from splinewire.rectify import min_links, rectify_chain

# More candidates than this in a frame is a busy background, not a chain worth
# ordering at video rate (ordering is quadratic in the number of points).
MAX_POINTS = 120


def analyse(points_px, diameters_px, size: tuple[int, int], spec: ChainSpec,
            focal_px: float | None = None) -> dict:
    """The chain among a frame's blobs.

    points_px: (N, 2) blob centres; diameters_px: (N,) their sizes;
    size: the frame's (width, height); focal_px: in this frame's pixels, or
    None to estimate it from the chain.

    Returns {"order": indices into points_px in chain order ([] if no chain),
    "gaps": positions in order that skip a missing pin,
    "tilt_deg", "link_residual_max_mm", "tilt_dir": None when there are too
    few links to take the perspective out}. tilt_dir is a unit vector in the
    frame's pixel axes (x right, y down) toward the chain's far side: the way
    to move the phone, still aimed at the chain, to look straight down on it.
    Never raises for a frame without a chain.
    """
    none = {"order": [], "gaps": [], "tilt_deg": None, "link_residual_max_mm": None, "tilt_dir": None}
    pts = np.asarray(points_px, dtype=float).reshape(-1, 2)
    d = np.asarray(diameters_px, dtype=float).reshape(-1)
    if not 3 <= len(pts) <= MAX_POINTS or len(d) != len(pts):
        return none
    try:
        order = order_chain(
            pts, axes_px=np.c_[d, d], pitch_per_diameter=spec.pitch_mm / spec.fiducial_mm,
            n_pins=spec.n_pins, max_turn_deg=spec.max_bend_deg,
        )
    except ValueError:
        return none
    out = {"order": [int(i) for i in order.indices], "gaps": [int(k) for k in order.gaps],
           "tilt_deg": None, "link_residual_max_mm": None, "tilt_dir": None}
    if len(order.links) >= min_links(focal_known=focal_px is not None):
        try:
            rect = rectify_chain(pts[order.indices], order.links, spec.pitch_mm, size, focal_px)
        except ValueError:
            return out
        out["tilt_deg"] = round(float(rect.tilt_deg), 1)
        out["link_residual_max_mm"] = round(float(rect.residual_max_mm), 3)
        out["tilt_dir"] = far_side(rect.normal)
    return out


def far_side(normal) -> list[float] | None:
    """The way across the image to the plane's far side, a unit vector rounded
    to 0.001; None when the camera looks straight down the normal.

    The pins lie on m . X = 1 with m along the normal, so a pin's inverse depth
    is m . (x, y, 1) for its normalised image point (x, y): it falls fastest,
    the plane receding, along -(n_x, n_y). Moving the camera that way, still
    aimed at the chain, turns its axis toward the normal."""
    v = -np.asarray(normal, dtype=float)[:2]
    length = float(np.hypot(*v))
    if length < 1e-9:
        return None
    return [round(float(v[0] / length), 3), round(float(v[1] / length), 3)]


def analyse_json(doc: dict) -> dict:
    """`analyse` for the worker's "analyse" command (web/src/core/protocol.ts FramePoints)."""
    return analyse(doc["points"], doc["diameters"], (int(doc["width"]), int(doc["height"])),
                   spec_from_dict(doc["chain"]), doc.get("focal_px"))
