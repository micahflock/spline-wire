"""Recover the chain's true shape from a photo using only the fixed pin pitch.

Every pin lies on one plane. Given the camera's focal length, each detected
pin defines a ray from the camera, and the plane {X : m . X = 1} cuts those
rays at X_i = ray_i / (m . ray_i). The three numbers in m (the plane's tilt
and distance) are chosen so that consecutive pins come out exactly one pitch
apart. That removes perspective without any reference object in the photo.

If the focal length is unknown it is added as a fourth unknown. That works
for strongly curved chains but is poorly constrained for nearly straight
ones, so prefer the EXIF value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from splinewire.camera import intrinsics


@dataclass(frozen=True)
class Rectification:
    # (N, 2) pin positions in mm, origin at the first pin. Axes follow the
    # photo: +x toward the image's right edge, +y toward its top edge.
    pins_mm: np.ndarray
    focal_px: float
    focal_estimated: bool
    tilt_deg: float          # angle between the optical axis and the plane normal
    residual_rms_mm: float   # how far link lengths deviate from the pitch
    residual_max_mm: float
    link_residuals_mm: np.ndarray | None = None   # per link: length minus pitch
    depth_mm: np.ndarray | None = None            # per pin: distance along the optical axis
    distortion_k1: float = 0.0                    # radial distortion solved for (0 unless asked)


def rectify_chain(
    points_px: np.ndarray,
    links: list[tuple[int, int]],
    pitch_mm: float,
    image_size: tuple[int, int],
    focal_px: float | None = None,
    estimate_distortion: bool = False,
) -> Rectification:
    """Place detected pins on the chain plane.

    points_px: (N, 2) pin centers in pixel coordinates.
    links: index pairs of pins known to be exactly one pitch apart.
    estimate_distortion: also solve for residual radial lens distortion k1
    (see experiments/lens_distortion.py for when it helps).
    """
    points_px = np.asarray(points_px, dtype=float)
    links_arr = np.asarray(links, dtype=int).reshape(-1, 2)
    solve_f = focal_px is None
    n_unknowns = 3 + solve_f + estimate_distortion
    if len(links_arr) < n_unknowns + 2:
        raise ValueError(
            f"need at least {n_unknowns + 2} links to rectify, got {len(links_arr)}"
        )

    w, h = image_size
    diag = math.hypot(w, h)
    focal_guesses = [focal_px] if focal_px is not None else [0.6 * diag, 0.85 * diag, 1.2 * diag]

    def unpack(p: np.ndarray, f0: float) -> tuple[float, float]:
        f = _focal(f0, p[3]) if solve_f else f0
        k1 = float(np.clip(p[-1], -30.0, 30.0)) / _K1_SCALE if estimate_distortion else 0.0
        return f, k1

    best = None
    for f0 in focal_guesses:
        px_pitch = np.median(np.linalg.norm(
            points_px[links_arr[:, 1]] - points_px[links_arr[:, 0]], axis=1))
        d0 = f0 * pitch_mm / px_pitch   # distance if the photo were straight-on

        def residuals(p: np.ndarray, f0=f0, d0=d0) -> np.ndarray:
            m = p[:3] / d0
            f, k1 = unpack(p, f0)
            X = _lift(m, _rays(points_px, f, image_size, k1))
            return np.linalg.norm(X[links_arr[:, 1]] - X[links_arr[:, 0]], axis=1) - pitch_mm

        for q0 in _plane_starts():
            p0 = np.r_[q0, [0.0] * (n_unknowns - 3)]
            try:
                sol = least_squares(residuals, p0, method="lm", max_nfev=400)
            except ValueError:
                continue
            if best is None or sol.cost < best[0].cost:
                best = (sol, f0, d0)

    sol, f0, d0 = best
    m = sol.x[:3] / d0
    f, k1 = unpack(sol.x, f0)
    X = _lift(m, _rays(points_px, f, image_size, k1))
    if np.any(X[:, 2] <= 0):   # m and -m fit equally; keep the plane in front of the camera
        m, X = -m, -X

    n = m / np.linalg.norm(m)                  # plane normal, pointing away from camera
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n  # image +x, projected into the plane
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(-n, e1)                      # image "up", projected into the plane
    pins = np.c_[X @ e1, X @ e2]
    pins -= pins[0]

    res = np.linalg.norm(pins[links_arr[:, 1]] - pins[links_arr[:, 0]], axis=1) - pitch_mm
    return Rectification(
        pins_mm=pins,
        focal_px=float(f),
        focal_estimated=focal_px is None,
        tilt_deg=float(math.degrees(math.acos(min(1.0, abs(n[2]))))),
        residual_rms_mm=float(np.sqrt(np.mean(res ** 2))),
        residual_max_mm=float(np.max(np.abs(res))),
        link_residuals_mm=res,
        depth_mm=X[:, 2].copy(),
        distortion_k1=k1,
    )


_K1_SCALE = 100.0    # solver works on 100 * k1, which is of order 1


def _focal(f0: float, log_ratio: float) -> float:
    # Keep the focal-length search within a plausible range for phone cameras.
    return f0 * math.exp(float(np.clip(log_ratio, -1.0, 1.0)))


def _rays(points_px: np.ndarray, focal_px: float, image_size: tuple[int, int], k1: float = 0.0) -> np.ndarray:
    """Viewing rays (z = 1) of pixels; k1 undoes radial distortion,
    x_ideal = x_image * (1 + k1 r^2) with r in focal lengths."""
    K = intrinsics(focal_px, image_size)
    xy = (points_px - K[:2, 2]) / focal_px
    if k1:
        xy = xy * (1.0 + k1 * np.sum(xy ** 2, axis=1, keepdims=True))
    return np.c_[xy, np.ones(len(points_px))]


def _lift(m: np.ndarray, rays: np.ndarray) -> np.ndarray:
    return rays / (rays @ m)[:, None]


def _plane_starts() -> list[np.ndarray]:
    """Initial plane orientations: straight-on plus tilts in 8 directions.

    Several starts matter: from a single straight-on start the solver can
    settle in a wrong local minimum for circular-arc chains.
    """
    starts = [np.array([0.0, 0.0, 1.0])]
    for tilt in (20.0, 40.0, 60.0):
        t = math.radians(tilt)
        for k in range(8):
            a = k * math.pi / 4
            starts.append(np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)]))
    return starts
