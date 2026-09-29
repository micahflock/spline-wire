"""Find ring fiducials in a photo with sub-pixel accuracy.

A ring is a blob with exactly one hole, where both the outer and inner
edges are well fit by concentric ellipses. Both polarities are searched
(light ring on dark link, or dark ring on light background).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Ring:
    center_px: tuple[float, float]
    outer_axes_px: tuple[float, float]  # ellipse (major, minor) diameters
    inner_axes_px: tuple[float, float]
    angle_deg: float                    # outer ellipse orientation


def detect_rings(
    image: np.ndarray,
    inner_outer_ratio: float,
    min_diameter_px: float = 8.0,
    ratio_tolerance: float = 0.35,
) -> list[Ring]:
    """Detect ring fiducials.

    inner_outer_ratio is the ring's inner/outer diameter ratio from the chain
    spec; it rejects blobs-with-holes that are not our rings.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    # The local-threshold window must be wider than any uniform patch that is
    # part of a ring (e.g. the hole), or that patch gets hollowed out.
    block = max(31, (min(gray.shape) // 4) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -4)

    rings: list[Ring] = []
    for mask in (binary, 255 - binary):
        rings.extend(_rings_in(mask, inner_outer_ratio, min_diameter_px, ratio_tolerance))
    return _dedupe(rings)


def _rings_in(
    mask: np.ndarray, target_ratio: float, min_diameter_px: float, tol: float
) -> list[Ring]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    found = []
    for i, (_next, _prev, child, parent) in enumerate(hierarchy):
        if parent != -1 or child == -1 or hierarchy[child][0] != -1:
            continue  # need a top-level blob with exactly one hole
        outer, inner = contours[i], contours[child]
        if len(outer) < 12 or len(inner) < 8:
            continue
        eo, ei = cv2.fitEllipse(outer), cv2.fitEllipse(inner)
        (cxo, cyo), (ao, bo), angle = eo
        (cxi, cyi), (ai, bi), _ = ei
        if min(ao, bo) < min_diameter_px:
            continue
        if not (_ellipse_like(outer, eo) and _ellipse_like(inner, ei)):
            continue
        ratio = np.sqrt(ai * bi / (ao * bo))
        if abs(ratio - target_ratio) > tol * target_ratio:
            continue
        if np.hypot(cxo - cxi, cyo - cyi) > 0.1 * min(ao, bo):
            continue
        # Contour pixels sit half a pixel inside the blob on average, which
        # shifts both edges equally; averaging the two ellipse centers cancels
        # most of the remaining asymmetry.
        found.append(Ring(
            center_px=((cxo + cxi) / 2, (cyo + cyi) / 2),
            outer_axes_px=(max(ao, bo), min(ao, bo)),
            inner_axes_px=(max(ai, bi), min(ai, bi)),
            angle_deg=angle,
        ))
    return found


def _ellipse_like(contour: np.ndarray, ellipse) -> bool:
    """Every contour point lies close to the fitted ellipse."""
    (cx, cy), (a, b), angle = ellipse
    if min(a, b) <= 0:
        return False
    t = np.radians(angle)
    p = contour.reshape(-1, 2).astype(float) - (cx, cy)
    u = p[:, 0] * np.cos(t) + p[:, 1] * np.sin(t)
    v = -p[:, 0] * np.sin(t) + p[:, 1] * np.cos(t)
    r = np.hypot(u / (a / 2), v / (b / 2))            # 1.0 on the ellipse
    deviation_px = np.abs(r - 1.0) * min(a, b) / 2
    return float(deviation_px.max()) < max(1.5, 0.06 * min(a, b) / 2)


def _dedupe(rings: list[Ring]) -> list[Ring]:
    kept: list[Ring] = []
    for r in sorted(rings, key=lambda r: -r.outer_axes_px[1]):
        if all(np.hypot(r.center_px[0] - k.center_px[0], r.center_px[1] - k.center_px[1])
               > 0.5 * k.outer_axes_px[1] for k in kept):
            kept.append(r)
    return kept
