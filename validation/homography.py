from __future__ import annotations

import cv2
import numpy as np


def fit_homography(pts_px: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    """Fit 3x3 homography mapping pixel coordinates to mm coordinates.

    Both inputs must be Nx2 arrays of corresponding points, N >= 4.
    """
    if pts_px.shape[0] < 4 or pts_mm.shape[0] < 4:
        raise ValueError("need at least 4 correspondences")
    if pts_px.shape != pts_mm.shape:
        raise ValueError("input shape mismatch")
    H, _mask = cv2.findHomography(pts_px, pts_mm, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None")
    return H


def apply_homography(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Apply homography to Nx2 pixel points, return Nx2 mm points."""
    if pts_px.ndim != 2 or pts_px.shape[1] != 2:
        raise ValueError("pts_px must be shape (N, 2)")
    reshaped = pts_px.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(reshaped, H)
    return out.reshape(-1, 2)
