"""Find the fiducials on the pins, light dots or rings, with sub-pixel accuracy.

`detect_fiducials(image, spec)` picks the detector for the chain's kind of
fiducial. Rings (`detect_rings`) are told from other bright blobs by their
hole; dots (`detect_dots`), which have none, by the dark link around them:
see detect_dots. Both work in two stages, described here for rings:

1. Candidates. Binarize and look for blobs with exactly one hole whose
   outer and inner edges are roughly concentric ellipses, both polarities.
   This runs on a small image pyramid so every ring is also seen at a
   scale where it is a few dozen pixels across: there, area downsampling
   and a light blur erase fine texture that would otherwise break a ring's
   outline (the extrusion lines of a 3D print catch the light as stripes
   0.4 mm apart that can bridge a ring to its surroundings). The local
   threshold windows are about one to two rings wide at that scale: wide
   enough to span a ring's hole, narrow enough to stay on the link, so the
   table beside the link, a shadow edge or a glare hotspot on the link
   does not drag the threshold off the ring.

2. Refinement, on the full-resolution gray image. Rays are cast through
   each candidate; on each ray the inner and outer edges are located to
   sub-pixel precision where the profile crosses halfway between that
   ray's own dark and bright levels, so a shadow edge or glare gradient
   across the ring does not shift them. Ellipses are fitted to the edge
   points with outlier rejection (seam blobs, dust, scratches), and the
   centre is the mean of the two ellipse centres weighted by fit quality
   and area (print defects shift a small circle's centre the most).
   Candidates whose profiles do not look like a ring are dropped here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Fiducial:
    center_px: tuple[float, float]
    outer_axes_px: tuple[float, float]  # ellipse (major, minor) diameters
    inner_axes_px: tuple[float, float]  # a ring's hole; (0, 0) for a dot
    angle_deg: float                    # outer ellipse orientation
    residual_px: float = 0.0            # RMS distance of edge points from the fitted ellipses
    contrast: float = 0.0               # bright minus dark levels, gray levels
    polarity: int = 1                   # +1: light on dark, -1: dark on light
    surround: float = 0.0               # gray level just outside (the link, for a pin)


@dataclass(frozen=True)
class _Candidate:
    center: tuple[float, float]
    outer: tuple                        # cv2 ellipse ((cx, cy), (a, b), angle), full resolution
    inner_ratio: float                  # inner / outer size, as thresholded
    polarity: int                       # +1: light ring on dark, -1: dark ring on light


_BLOCK = 25            # local threshold window at each pyramid level, px
_MIN_LEVEL_SIDE = 360   # smallest pyramid level, px
_MAX_SCALE = 8          # coarsest level: 1/8 resolution


def detect_fiducials(image: np.ndarray, spec) -> list[Fiducial]:
    """The fiducials a ChainSpec describes: dots or rings."""
    if spec.fiducial == "dot":
        return detect_dots(image)
    return detect_rings(image, spec.ring_inner_mm / spec.fiducial_mm)


def _pyramid(gray: np.ndarray):
    """(blurred level, scale) from full size down to about 1/8."""
    level_img, scale = gray, 1.0
    while True:
        yield cv2.GaussianBlur(level_img, (0, 0), 0.8 if scale == 1 else 1.0), scale
        if min(level_img.shape) < 2 * _MIN_LEVEL_SIDE or scale >= _MAX_SCALE:
            return
        level_img = cv2.resize(level_img, (level_img.shape[1] // 2, level_img.shape[0] // 2),
                               interpolation=cv2.INTER_AREA)
        scale *= 2


def detect_rings(
    image: np.ndarray,
    inner_outer_ratio: float,
    min_diameter_px: float = 8.0,
) -> list[Fiducial]:
    """Detect ring fiducials.

    inner_outer_ratio is the ring's inner/outer diameter ratio from the chain
    spec; it rejects blobs-with-holes that are not our rings. The accepted
    range is lopsided on purpose: thresholding a blurred ring erodes its
    band from both sides (a 0.4 ring measures ~0.5), and slightly small
    printed windows, recess walls seen at an angle and defocus all push the
    same way. Centers are unaffected because the erosion is symmetric.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates: list[_Candidate] = []
    for g, scale in _pyramid(gray):
        for mask, polarity in _masks(g):
            candidates.extend(_ring_candidates(mask, polarity, inner_outer_ratio,
                                               min_diameter_px if scale == 1 else 14.0, scale))

    grayf = gray.astype(np.float32)
    rings = []
    for c in _dedupe_candidates(candidates, same_size=True):
        r = _refine(grayf, c, inner_outer_ratio)
        if r is not None:
            rings.append(r)
    return _dedupe(rings)


def _masks(gray: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Binarizations of one pyramid level, with the ring polarity each shows.

    - Above / below the local mean: the workhorse, in both polarities.
    - Well above the local mean (by 0.8 local standard deviations), where
      there is contrast at all. The mean alone fails in two places: where
      most of the window is one shade (a black chain on a dark table puts
      the mean at the black level, so sensor noise speckles the links and
      breaks the ring edges), and where sheen makes the black link lighter
      than the table beside it (the link then sits above the mean and the
      ring merges into it). The ring is still the brightest thing locally.
    """
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, _BLOCK, -4)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (_BLOCK, _BLOCK))
    var = cv2.boxFilter(g * g, -1, (_BLOCK, _BLOCK)) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    bright = ((g > mean + 0.8 * std) & (std > 6.0)).astype(np.uint8) * 255
    return [(binary, 1), (255 - binary, -1), (bright, 1)]


def _ring_candidates(
    mask: np.ndarray, polarity: int, target_ratio: float, min_diameter_px: float, scale: float,
) -> list[_Candidate]:
    ratio_lo = 0.5 * target_ratio
    ratio_hi = target_ratio + 0.75 * (1.0 - target_ratio)
    # Components first, contours only for those that could be a ring: ring-scale
    # thresholds turn fine texture (woven fabric, wood grain) into hundreds of
    # thousands of specks, and tracing every one of them took 20 s on a 12 MP
    # photo of a chain on fabric.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y, w, h, area = (stats[:, i] for i in range(5))
    lo, hi = np.minimum(w, h), np.maximum(w, h)
    maybe = ((lo >= min_diameter_px) & (hi <= max(mask.shape) / 3) & (hi <= 8 * lo)
             & (area <= 0.85 * w * h))
    maybe[0] = False                                     # label 0 is the black background
    found = []
    for k in np.flatnonzero(maybe):
        x0, y0 = max(0, x[k] - 1), max(0, y[k] - 1)
        roi = (labels[y0:y[k] + h[k] + 1, x0:x[k] + w[k] + 1] == k).astype(np.uint8)
        contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
                                               offset=(int(x0), int(y0)))
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        top = [i for i, hh in enumerate(hierarchy) if hh[3] == -1]
        if len(top) != 1 or hierarchy[top[0]][2] == -1:
            continue                                     # need one blob with a hole
        outer = contours[top[0]]
        if len(outer) < 12:
            continue
        # Exactly one real hole. Specks of noise inside the ring make extra
        # tiny holes, so holes far smaller than a ring's hole are ignored.
        min_hole = max(4.0, 0.02 * cv2.contourArea(outer))
        holes = [contours[c] for c in _children(hierarchy, hierarchy[top[0]][2])
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
        # Contour points are pixel centres; map level pixels to full resolution.
        off = (scale - 1) / 2
        found.append(_Candidate(
            center=((cxo + cxi) / 2 * scale + off, (cyo + cyi) / 2 * scale + off),
            outer=((cxo * scale + off, cyo * scale + off), (ao * scale, bo * scale), angle),
            inner_ratio=float(ratio), polarity=polarity,
        ))
    return found


def _children(hierarchy: np.ndarray, first: int) -> list[int]:
    out = []
    while first != -1:
        out.append(first)
        first = hierarchy[first][0]
    return out


def _ellipse_like(contour: np.ndarray, ellipse) -> bool:
    """The contour follows the fitted ellipse, allowing a few stray points
    (a seam blob or a speck of dust on the edge)."""
    (cx, cy), (a, b), angle = ellipse
    if min(a, b) <= 0:
        return False
    t = np.radians(angle)
    p = contour.reshape(-1, 2).astype(float) - (cx, cy)
    u = p[:, 0] * np.cos(t) + p[:, 1] * np.sin(t)
    v = -p[:, 0] * np.sin(t) + p[:, 1] * np.cos(t)
    r = np.hypot(u / (a / 2), v / (b / 2))            # 1.0 on the ellipse
    deviation_px = np.abs(r - 1.0) * min(a, b) / 2
    tol = max(1.5, 0.06 * min(a, b) / 2)
    return float(np.percentile(deviation_px, 90)) < tol and float(deviation_px.max()) < 3 * tol


def _dedupe_candidates(cands: list[_Candidate], same_size: bool = False) -> list[_Candidate]:
    """One candidate per ring: the same ring is found at several pyramid
    levels and in several binarizations. With same_size, only candidates of
    about the same size merge: a larger blob sharing a dot's centre (a halo
    of table around a dark patch) must not stand in for the dot."""
    kept: list[_Candidate] = []
    for c in sorted(cands, key=lambda c: -min(c.outer[1])):
        size = min(c.outer[1])
        if all(math.hypot(c.center[0] - k.center[0], c.center[1] - k.center[1])
               > 0.25 * min(size, min(k.outer[1])) or c.polarity != k.polarity
               or (same_size and min(k.outer[1]) > 1.3 * size) for k in kept):
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Sub-pixel refinement

_STEP_PX = 0.25          # sample spacing along each ray


def _refine(gray: np.ndarray, cand: _Candidate, target_ratio: float, passes: int = 2) -> Fiducial | None:
    ellipse, inner_ratio = cand.outer, cand.inner_ratio
    fit = None
    for _ in range(passes):
        fit = _fit_once(gray, ellipse, inner_ratio, cand.polarity)
        if fit is None:
            return None
        ellipse, inner_ratio = fit["outer"], fit["inner_ratio"]
    return _accept(fit, target_ratio, cand.polarity)


def _fit_once(gray, ellipse, inner_ratio, polarity) -> dict | None:
    (cx, cy), (A, B), ang = ellipse
    R = max(A, B) / 2
    if not np.isfinite(R) or R < 3:
        return None
    n_rays = int(np.clip(math.pi * 2 * R / 2.0, 32, 180))
    phi = np.arange(n_rays) * (2 * math.pi / n_rays)
    t = math.radians(ang)
    ca, sa = math.cos(t), math.sin(t)
    ex = (A / 2) * np.cos(phi) * ca - (B / 2) * np.sin(phi) * sa     # ray to the outer edge
    ey = (A / 2) * np.cos(phi) * sa + (B / 2) * np.sin(phi) * ca
    s_max = 1.35
    n_s = int(math.ceil(s_max * R / _STEP_PX)) + 1
    s = np.linspace(0.0, s_max, n_s, dtype=np.float32)
    ds = float(s[1] - s[0])
    mx = (cx + s[None, :] * ex[:, None]).astype(np.float32)
    my = (cy + s[None, :] * ey[:, None]).astype(np.float32)
    q = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if polarity < 0:
        q = 255.0 - q                                      # make the ring band the bright part

    si = float(np.clip(inner_ratio, 0.15, 0.85))
    band_mid = (si + 1) / 2

    def idx(v):
        return int(np.clip(round(v / ds), 0, n_s - 1))

    lv_center = np.median(q[:, :max(1, idx(0.6 * si))], axis=1)
    lv_band = np.median(q[:, idx(si + 0.25 * (1 - si)):idx(1 - 0.25 * (1 - si)) + 1], axis=1)
    lv_out = np.median(q[:, idx(1.1):idx(1.3) + 1], axis=1)
    contrast = np.minimum(lv_band - lv_center, lv_band - lv_out)

    s_in = _crossings(q, (lv_center + lv_band) / 2, idx(0.4 * si), idx(band_mid), si / ds, ds, rising=True)
    s_out = _crossings(q, (lv_band + lv_out) / 2, idx(band_mid), idx(1.3), 1.0 / ds, ds, rising=False)
    good = contrast > max(6.0, 0.25 * float(np.median(contrast)))
    pin = np.c_[cx + s_in * ex, cy + s_in * ey][good & np.isfinite(s_in)]
    pout = np.c_[cx + s_out * ex, cy + s_out * ey][good & np.isfinite(s_out)]
    if len(pin) < 0.5 * n_rays or len(pout) < 0.5 * n_rays:
        return None
    fo, fi = _robust_ellipse(pout), _robust_ellipse(pin)
    if fo is None or fi is None:
        return None
    (eo, res_o, n_o), (ei, res_i, n_i) = fo, fi
    if n_o < 0.5 * n_rays or n_i < 0.5 * n_rays:
        return None
    # Weight each ellipse's centre by its fit quality and its area: a print
    # defect (seam blob, wobble) pulls a small circle's fitted centre further
    # than a large one's, roughly in proportion to 1/radius.
    wo = n_o / max(res_o, 0.05) ** 2 * eo[1][0] * eo[1][1]
    wi = n_i / max(res_i, 0.05) ** 2 * ei[1][0] * ei[1][1]
    center =((wo * eo[0][0] + wi * ei[0][0]) / (wo + wi), (wo * eo[0][1] + wi * ei[0][1]) / (wo + wi))
    return {
        "outer": eo, "inner": ei, "center": center,
        "inner_ratio": math.sqrt(ei[1][0] * ei[1][1] / (eo[1][0] * eo[1][1])),
        "residual": math.sqrt((n_o * res_o ** 2 + n_i * res_i ** 2) / (n_o + n_i)),
        "contrast": float(np.median(contrast)),
        "surround": float(np.median(lv_out) if polarity > 0 else 255.0 - np.median(lv_out)),
        "coverage": min(n_o, n_i) / n_rays,
    }


def _crossings(q, level, k0, k1, k_target, ds, rising: bool) -> np.ndarray:
    """Per ray, where the profile crosses `level` between samples k0..k1,
    nearest to k_target; NaN where it does not."""
    n = q.shape[0]
    k1 = max(k1, k0 + 1)
    seg = q[:, k0:k1 + 1] - level[:, None]
    a, b = seg[:, :-1], seg[:, 1:]
    cross = (a < 0) & (b >= 0) if rising else (a >= 0) & (b < 0)
    k = np.arange(k0, k1)
    dist = np.where(cross, np.abs(k - k_target), np.inf)
    j = np.argmin(dist, axis=1)
    rows = np.arange(n)
    ok = np.isfinite(dist[rows, j])
    aj, bj = a[rows, j], b[rows, j]
    frac = np.where(ok, aj / np.where(aj - bj == 0, 1.0, aj - bj), 0.0)
    return np.where(ok, (k0 + j + frac) * ds, np.nan)


def _robust_ellipse(pts: np.ndarray):
    """Ellipse through edge points, dropping outliers. Returns
    (ellipse, rms residual px, inlier count) or None."""
    keep = np.ones(len(pts), bool)
    for _ in range(3):
        if keep.sum() < 6:
            return None
        e = cv2.fitEllipseDirect(pts[keep].astype(np.float32))
        d = _ellipse_distance(pts, e)
        mad = 1.4826 * float(np.median(np.abs(d[keep])))
        new_keep = np.abs(d) < max(0.3, 3.5 * mad)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if keep.sum() < 6:
        return None
    e = cv2.fitEllipseDirect(pts[keep].astype(np.float32))
    d = _ellipse_distance(pts[keep], e)
    return e, float(np.sqrt(np.mean(d ** 2))), int(keep.sum())


def _ellipse_distance(pts: np.ndarray, ellipse) -> np.ndarray:
    """Approximate signed distance (px) of points from an ellipse, outward positive."""
    (cx, cy), (a, b), angle = ellipse
    t = math.radians(angle)
    p = pts - (cx, cy)
    u = p[:, 0] * math.cos(t) + p[:, 1] * math.sin(t)
    v = -p[:, 0] * math.sin(t) + p[:, 1] * math.cos(t)
    a2, b2 = max(a / 2, 1e-6), max(b / 2, 1e-6)
    r = np.hypot(u / a2, v / b2)
    return (r - 1.0) * np.hypot(u, v) / np.maximum(r, 1e-9)


def _accept(fit: dict, target_ratio: float, polarity: int) -> Fiducial | None:
    eo, ei = fit["outer"], fit["inner"]
    ao, bo = max(eo[1]), min(eo[1])
    ai, bi = max(ei[1]), min(ei[1])
    if bo <= 0 or bi <= 0:
        return None
    ratio = fit["inner_ratio"]
    if not 0.5 * target_ratio <= ratio <= target_ratio + 0.75 * (1.0 - target_ratio):
        return None
    if math.hypot(eo[0][0] - ei[0][0], eo[0][1] - ei[0][1]) > 0.08 * bo + 0.5:
        return None                    # not concentric
    # Inner and outer should have about the same shape (one flat ring), but
    # on a print the black layer's walls, seen at a steep angle, make the
    # raised centre disc look rounder and the window flatter.
    if abs(bo / ao - bi / ai) > 0.3 + 2.0 / bi:
        return None
    if fit["residual"] > max(0.6, 0.04 * bo):
        return None
    return Fiducial(
        center_px=(float(fit["center"][0]), float(fit["center"][1])),
        outer_axes_px=(float(ao), float(bo)),
        inner_axes_px=(float(ai), float(bi)),
        angle_deg=float(eo[2]),
        residual_px=float(fit["residual"]),
        contrast=float(fit["contrast"]),
        surround=fit["surround"],
        polarity=polarity,
    )


def _dedupe(rings: list[Fiducial]) -> list[Fiducial]:
    kept: list[Fiducial] = []
    for r in sorted(rings, key=lambda r: -r.outer_axes_px[1]):
        if all(np.hypot(r.center_px[0] - k.center_px[0], r.center_px[1] - k.center_px[1])
               > 0.5 * k.outer_axes_px[1] or r.polarity != k.polarity for k in kept):
            kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# Dots

# The dark margin sampled around a dot, in dot radii: outside the dot's own
# edge blur, inside the link's edge (a 5 mm dot on an 8 mm link leaves 1.5
# mm, reaching 1.6 radii).
_DOT_MARGIN = (1.15, 1.45)


def detect_dots(image: np.ndarray, min_diameter_px: float = 8.0) -> list[Fiducial]:
    """Detect light dot fiducials on a dark link.

    A dot has no hole to tell it from any other bright blob (terrazzo chips,
    grain highlights, specks, the counters of printed letters), so each one
    must look like a dot on a link all the way round: along every ray the
    margin just outside it is darker than its inside by a steady fraction.
    The fraction rather than the difference, because a shadow edge darkens
    dot and margin alike. Candidates use their outer outline only: a dot
    wider than the threshold window comes out hollow at full size.
    Look-alikes that pass (the odd chip) are left to ordering, which knows
    the pitch and each pin's size and shape.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates: list[_Candidate] = []
    for g, scale in _pyramid(gray):
        for mask, polarity in _masks(g):
            if polarity > 0:
                candidates.extend(_dot_candidates(mask, min_diameter_px if scale == 1 else 10.0, scale))
    grayf = gray.astype(np.float32)
    dots = []
    for c in _dedupe_candidates(candidates, same_size=True):
        d = _refine_dot(grayf, c)
        if d is not None:
            dots.append(d)
    return _dedupe(dots)


def _dot_candidates(mask: np.ndarray, min_diameter_px: float, scale: float) -> list[_Candidate]:
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    x, y, w, h, _area = (stats[:, i] for i in range(5))
    lo, hi = np.minimum(w, h), np.maximum(w, h)
    maybe = (lo >= min_diameter_px) & (hi <= max(mask.shape) / 4) & (hi <= 4 * lo)
    maybe[0] = False                                     # label 0 is the dark background
    off = (scale - 1) / 2
    found = []
    for k in np.flatnonzero(maybe):
        x0, y0 = max(0, x[k] - 1), max(0, y[k] - 1)
        roi = (labels[y0:y[k] + h[k] + 1, x0:x[k] + w[k] + 1] == k).astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                                       offset=(int(x0), int(y0)))
        if len(contours) != 1 or len(contours[0]) < 12:
            continue
        e = cv2.fitEllipse(contours[0])
        (cx, cy), (a, b), angle = e
        if min(a, b) < min_diameter_px or not _ellipse_like(contours[0], e):
            continue
        found.append(_Candidate(center=(cx * scale + off, cy * scale + off),
                                outer=((cx * scale + off, cy * scale + off), (a * scale, b * scale), angle),
                                inner_ratio=0.0, polarity=1))
    return found


def _refine_dot(gray: np.ndarray, cand: _Candidate, passes: int = 2) -> Fiducial | None:
    (cx, cy), (A, B), ang = cand.outer
    for k in range(passes):
        need = 0.55 if k < passes - 1 else 0.7          # a rough first outline misses some edges
        R = max(A, B) / 2
        if not np.isfinite(R) or R < 3 or R > 0.25 * max(gray.shape):
            return None
        n_rays = int(np.clip(math.pi * R, 32, 180))
        phi = np.arange(n_rays) * (2 * math.pi / n_rays)
        t = math.radians(ang)
        ex = (A / 2) * np.cos(phi) * math.cos(t) - (B / 2) * np.sin(phi) * math.sin(t)
        ey = (A / 2) * np.cos(phi) * math.sin(t) + (B / 2) * np.sin(phi) * math.cos(t)
        n_s = int(math.ceil(1.5 * R / _STEP_PX)) + 1
        s = np.linspace(0.0, 1.5, n_s, dtype=np.float32)
        ds = float(s[1] - s[0])
        mx = (cx + s[None, :] * ex[:, None]).astype(np.float32)
        my = (cy + s[None, :] * ey[:, None]).astype(np.float32)
        q = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        def idx(v):
            return int(np.clip(round(v / ds), 0, n_s - 1))

        inside = np.median(q[:, idx(0.1):idx(0.7) + 1], axis=1)
        outside = np.median(q[:, idx(_DOT_MARGIN[0]):idx(_DOT_MARGIN[1]) + 1], axis=1)
        contrast = inside - outside
        if float(np.median(contrast)) < 15 or np.mean(contrast > 6.0) < 0.85:
            return None
        # Margin a steady fraction of the dot on most rays (a shadow band's
        # edges crossing the dot spoil a few; sheen can lift the margin to
        # ~90% of the dot and it is still one).
        ratio = (outside + 4.0) / (inside + 4.0)
        mr = float(np.median(ratio))
        if mr > 0.92 or np.mean(np.abs(ratio - mr) < 0.15) < 0.6:
            return None
        edge = _crossings(q, (inside + outside) / 2, idx(0.7), idx(1.3), 1.0 / ds, ds, rising=False)
        good = np.isfinite(edge)
        if good.sum() < need * n_rays:
            return None
        fit = _robust_ellipse(np.c_[cx + edge * ex, cy + edge * ey][good])
        if fit is None or fit[2] < need * n_rays:
            return None
        e, res, _ = fit
        (cx, cy), (A, B), ang = e
    if res > max(0.6, 0.04 * min(A, B)):
        return None
    return Fiducial(
        center_px=(float(cx), float(cy)),
        outer_axes_px=(float(max(A, B)), float(min(A, B))),
        inner_axes_px=(0.0, 0.0),
        angle_deg=float(ang),
        residual_px=float(res),
        contrast=float(np.median(contrast)),
        polarity=1,
        surround=float(np.median(outside)),
    )


# ---------------------------------------------------------------------------
# Snapping a click to a fiducial


def snap_fiducial(
    image: np.ndarray,
    point_px: tuple[float, float],
    spec,
    like: Fiducial,
) -> Fiducial | None:
    """The fiducial a person pointed at, found and refined like any detection;
    None if there is no clear one there.

    `like` is a neighbouring fiducial on the same chain, which says how big
    the one being looked for is and how it is foreshortened. The bright (for a
    ring, the ring-coloured) blob under the click is thresholded out of a
    small window and handed to the same sub-pixel refinement and acceptance
    tests the detector uses, so a click several pixels off still lands on the
    dot's real centre. Where the detector itself could see nothing (glare
    washes the dot out), neither does this, and the answer is None rather
    than a poor fit: a wrong snap looks as confident as a right one and is
    worse than the click.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ring = spec.ring_inner_mm / spec.fiducial_mm if spec.fiducial == "ring" else 0.0
    cand = _blob_under(gray, point_px, like, ring)
    if cand is None:
        return None
    grayf = gray.astype(np.float32)
    found = _refine_dot(grayf, cand) if spec.fiducial == "dot" else _refine(grayf, cand, ring)
    if found is None:
        return None
    major, minor = like.outer_axes_px
    if math.hypot(found.center_px[0] - point_px[0], found.center_px[1] - point_px[1]) > 0.5 * minor:
        return None                             # a different blob than the one pointed at
    if not 0.6 < found.outer_axes_px[0] / major < 1.6:
        return None                             # one of the wrong size
    return found


def _blob_under(gray: np.ndarray, point_px, like: Fiducial, ring_ratio: float) -> _Candidate | None:
    """Outline of the blob under (or nearest) a point, as a start for refinement."""
    major, minor = like.outer_axes_px
    half = int(round(1.7 * major))
    h, w = gray.shape
    x0, y0 = max(0, int(point_px[0]) - half), max(0, int(point_px[1]) - half)
    x1, y1 = min(w, int(point_px[0]) + half + 1), min(h, int(point_px[1]) + half + 1)
    if min(x1 - x0, y1 - y0) < 8:
        return None
    crop = cv2.GaussianBlur(np.clip(gray[y0:y1, x0:x1], 0, 255).astype(np.uint8), (0, 0), 1.0)
    flag = cv2.THRESH_BINARY if like.polarity > 0 else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(crop, 0, 255, flag + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    px, py = point_px[0] - x0, point_px[1] - y0
    label = int(labels[min(max(int(py), 0), labels.shape[0] - 1), min(max(int(px), 0), labels.shape[1] - 1)])
    if label == 0:
        # the click fell in a ring's hole or just off the blob: the nearest one
        dist = np.hypot(centroids[1:, 0] - px, centroids[1:, 1] - py)
        if not len(dist) or dist.min() > 0.7 * minor:
            return None
        label = 1 + int(np.argmin(dist))
    expected = math.pi / 4 * major * minor * (1.0 - ring_ratio ** 2)
    if not 0.35 * expected < stats[label, cv2.CC_STAT_AREA] < 2.5 * expected:
        return None
    contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or len(max(contours, key=len)) < 12:
        return None
    (cx, cy), (a, b), angle = cv2.fitEllipse(max(contours, key=len))
    c = (cx + x0, cy + y0)
    return _Candidate(center=c, outer=(c, (a, b), angle), inner_ratio=ring_ratio, polarity=like.polarity)
