import numpy as np
import pytest

from validation.homography import fit_homography, apply_homography


def test_identity_pixel_to_mm_with_known_scale():
    """10 px/mm uniform scale, no rotation, no translation."""
    pts_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    pts_mm = pts_px / 10.0
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[50, 50]], dtype=float))
    np.testing.assert_allclose(out, [[5.0, 5.0]], atol=1e-6)


def test_rotated_90deg_mapping():
    """90deg rotation: pixel (1,0) maps to mm (0,1)."""
    pts_px = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    pts_mm = np.array([[0, 0], [0, 1], [-1, 1], [-1, 0]], dtype=float)
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[0.5, 0.0]], dtype=float))
    np.testing.assert_allclose(out, [[0.0, 0.5]], atol=1e-6)


def test_requires_at_least_four_correspondences():
    pts_px = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    pts_mm = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    with pytest.raises(ValueError):
        fit_homography(pts_px, pts_mm)


def test_apply_handles_multiple_points():
    pts_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    pts_mm = pts_px / 10.0
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[0, 0], [50, 50], [100, 100]], dtype=float))
    np.testing.assert_allclose(
        out, [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], atol=1e-6
    )


def test_perspective_mapping_roundtrips():
    """Construct a known perspective, verify fit recovers it."""
    # quadrilateral in pixels, square in mm
    pts_px = np.array(
        [[10, 10], [200, 50], [210, 190], [20, 210]], dtype=float
    )
    pts_mm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, pts_px)
    np.testing.assert_allclose(out, pts_mm, atol=1e-6)
