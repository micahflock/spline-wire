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
) -> list[Ring]:
    """Detect ring fiducials.

    inner_outer_ratio is the ring's inner/outer diameter ratio from the chain
    spec; it rejects blobs-with-holes that are not our rings. The accepted
    range is lopsided on purpose: thresholding a blurred ring erodes its
    band from both sides (a 0.4 ring measures ~0.5), and slightly small
    printed windows, recess walls seen at an angle and defocus all push the
    same way. Centers are unaffected because the erosion is symmetric.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    # The local-threshold window must be wider than any uniform patch that is
    # part of a ring (e.g. the hole), or that patch gets hollowed out.
    block = max(31, (min(gray.shape) // 4) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -4)

    rings: list[Ring] = []
    for mask in (binary, 255 - binary, _midrange_binary(gray, block)):
        rings.extend(_rings_in(mask, inner_outer_ratio, min_diameter_px))
    return _dedupe(rings)


def _midrange_binary(gray: np.ndarray, block: int) -> np.ndarray:
    """Threshold halfway between the local darkest and brightest levels.

    The local mean fails where most of the window is one shade: a black
    chain on a dark table puts the mean at the black level, so sensor noise
    speckles the links and breaks up the ring edges. The midpoint between
    local extremes stays between black and white there. Areas without real
    contrast are left empty.
    """
    step = max(1, block // 32)                      # extremes on a coarse grid: fast
    small = cv2.resize(gray, (max(1, gray.shape[1] // step), max(1, gray.shape[0] // step)),
                       interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, block // step), max(3, block // step)))
    lo = cv2.resize(cv2.erode(small, k), gray.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.int16)
    hi = cv2.resize(cv2.dilate(small, k), gray.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.int16)
    return (((gray.astype(np.int16) * 2) > lo + hi) & (hi - lo > 40)).astype(np.uint8) * 255


def _rings_in(
    mask: np.ndarray, target_ratio: float, min_diameter_px: float
) -> list[Ring]:
    ratio_lo = 0.5 * target_ratio
    ratio_hi = target_ratio + 0.75 * (1.0 - target_ratio)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    found = []
    for i, (_next, _prev, child, parent) in enumerate(hierarchy):
        if parent != -1 or child == -1:
            continue  # need a top-level blob with a hole
        outer = contours[i]
        if len(outer) < 12:
            continue
        # Exactly one real hole. Specks of noise inside the ring make extra
        # tiny holes, so holes far smaller than a ring's hole are ignored.
        min_hole = max(4.0, 0.02 * cv2.contourArea(outer))
        holes = [contours[c] for c in _children(hierarchy, child)
                 if cv2.contourArea(contours[c]) >= min_hole]
        if len(holes) != 1:
            continue
        inner = holes[0]
        if len(inner) < 8:
            continue
        eo, ei = cv2.fitEllipse(outer), cv2.fitEllipse(inner)
        (cxo, cyo), (ao, bo), angle = eo
        (cxi, cyi), (ai, bi), _ = ei
        if min(ao, bo) < min_diameter_px:
            continue
        if not (_ellipse_like(outer, eo) and _ellipse_like(inner, ei)):
            continue
        ratio = np.sqrt(ai * bi / (ao * bo))
        if not ratio_lo <= ratio <= ratio_hi:
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


def _children(hierarchy: np.ndarray, first: int) -> list[int]:
    out = []
    while first != -1:
        out.append(first)
        first = hierarchy[first][0]
    return out


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
