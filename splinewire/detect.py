"""Find ring fiducials in a photo with sub-pixel accuracy.

Two stages:

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
class Ring:
    center_px: tuple[float, float]
    outer_axes_px: tuple[float, float]  # ellipse (major, minor) diameters
    inner_axes_px: tuple[float, float]
    angle_deg: float                    # outer ellipse orientation
    residual_px: float = 0.0            # RMS distance of edge points from the fitted ellipses
    contrast: float = 0.0               # bright band minus dark levels, gray levels
    polarity: int = 1                   # +1: light ring on dark, -1: dark ring on light


@dataclass(frozen=True)
class _Candidate:
    center: tuple[float, float]
    outer: tuple                        # cv2 ellipse ((cx, cy), (a, b), angle), full resolution
    inner_ratio: float                  # inner / outer size, as thresholded
    polarity: int                       # +1: light ring on dark, -1: dark ring on light


_BLOCK = 25            # local threshold window at each pyramid level, px
_MIN_LEVEL_SIDE = 360   # smallest pyramid level, px
_MAX_SCALE = 8          # coarsest level: 1/8 resolution


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
    candidates: list[_Candidate] = []
    level_img, scale = gray, 1.0
    while True:
        g = cv2.GaussianBlur(level_img, (0, 0), 0.8 if scale == 1 else 1.0)
        for mask, polarity in _masks(g):
            candidates.extend(_ring_candidates(mask, polarity, inner_outer_ratio,
                                               min_diameter_px if scale == 1 else 14.0, scale))
        if min(level_img.shape) < 2 * _MIN_LEVEL_SIDE or scale >= _MAX_SCALE:
            break
        level_img = cv2.resize(level_img, (level_img.shape[1] // 2, level_img.shape[0] // 2),
                               interpolation=cv2.INTER_AREA)
        scale *= 2

    grayf = gray.astype(np.float32)
    rings = []
    for c in _dedupe_candidates(candidates):
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


def _dedupe_candidates(cands: list[_Candidate]) -> list[_Candidate]:
    """One candidate per ring: the same ring is found at several pyramid
    levels and in several binarizations."""
    kept: list[_Candidate] = []
    for c in sorted(cands, key=lambda c: -min(c.outer[1])):
        size = min(c.outer[1])
        if all(math.hypot(c.center[0] - k.center[0], c.center[1] - k.center[1])
               > 0.25 * min(size, min(k.outer[1])) or c.polarity != k.polarity for k in kept):
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Sub-pixel refinement

_STEP_PX = 0.25          # sample spacing along each ray


def _refine(gray: np.ndarray, cand: _Candidate, target_ratio: float, passes: int = 2) -> Ring | None:
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


def _accept(fit: dict, target_ratio: float, polarity: int) -> Ring | None:
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
    return Ring(
        center_px=(float(fit["center"][0]), float(fit["center"][1])),
        outer_axes_px=(float(ao), float(bo)),
        inner_axes_px=(float(ai), float(bi)),
        angle_deg=float(eo[2]),
        residual_px=float(fit["residual"]),
        contrast=float(fit["contrast"]),
        polarity=polarity,
    )


def _dedupe(rings: list[Ring]) -> list[Ring]:
    kept: list[Ring] = []
    for r in sorted(rings, key=lambda r: -r.outer_axes_px[1]):
        if all(np.hypot(r.center_px[0] - k.center_px[0], r.center_px[1] - k.center_px[1])
               > 0.5 * k.outer_axes_px[1] or r.polarity != k.polarity for k in kept):
            kept.append(r)
    return kept
