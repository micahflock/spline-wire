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


def rectify_chain(
    points_px: np.ndarray,
    links: list[tuple[int, int]],
    pitch_mm: float,
    image_size: tuple[int, int],
    focal_px: float | None = None,
) -> Rectification:
    """Place detected pins on the chain plane.

    points_px: (N, 2) pin centers in pixel coordinates.
    links: index pairs of pins known to be exactly one pitch apart.
    """
    points_px = np.asarray(points_px, dtype=float)
    links_arr = np.asarray(links, dtype=int).reshape(-1, 2)
    n_unknowns = 3 if focal_px is not None else 4
    if len(links_arr) < n_unknowns + 2:
        raise ValueError(
            f"need at least {n_unknowns + 2} links to rectify, got {len(links_arr)}"
        )

    w, h = image_size
    diag = math.hypot(w, h)
    focal_guesses = [focal_px] if focal_px is not None else [0.6 * diag, 0.85 * diag, 1.2 * diag]

    best = None
    for f0 in focal_guesses:
        rays0 = _rays(points_px, f0, image_size)
        px_pitch = np.median(np.linalg.norm(
            points_px[links_arr[:, 1]] - points_px[links_arr[:, 0]], axis=1))
        d0 = f0 * pitch_mm / px_pitch   # distance if the photo were straight-on

        def residuals(p: np.ndarray, f0=f0, rays0=rays0, d0=d0) -> np.ndarray:
            m = p[:3] / d0
            rays = rays0 if focal_px is not None else _rays(points_px, _focal(f0, p[3]), image_size)
            X = _lift(m, rays)
            return np.linalg.norm(X[links_arr[:, 1]] - X[links_arr[:, 0]], axis=1) - pitch_mm

        for q0 in _plane_starts():
            p0 = q0 if focal_px is not None else np.r_[q0, 0.0]
            try:
                sol = least_squares(residuals, p0, method="lm", max_nfev=400)
            except ValueError:
                continue
            if best is None or sol.cost < best[0].cost:
                best = (sol, f0, d0)

    sol, f0, d0 = best
    m = sol.x[:3] / d0
    f = f0 if focal_px is not None else _focal(f0, sol.x[3])
    X = _lift(m, _rays(points_px, f, image_size))
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
    )


def _focal(f0: float, log_ratio: float) -> float:
    # Keep the focal-length search within a plausible range for phone cameras.
    return f0 * math.exp(float(np.clip(log_ratio, -1.0, 1.0)))


def _rays(points_px: np.ndarray, focal_px: float, image_size: tuple[int, int]) -> np.ndarray:
    K = intrinsics(focal_px, image_size)
    return np.c_[(points_px - K[:2, 2]) / focal_px, np.ones(len(points_px))]


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
