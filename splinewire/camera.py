"""Pinhole camera model and focal-length recovery from photo EXIF."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Diagonal of a 36 x 24 mm frame, which defines "35 mm equivalent" focal length.
_FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

_EXIF_IFD = 0x8769
_TAG_FOCAL_LENGTH_35MM = 0xA405


def intrinsics(focal_px: float, image_size: tuple[int, int]) -> np.ndarray:
    """Camera matrix with square pixels and the principal point at the image center."""
    w, h = image_size
    return np.array([
        [focal_px, 0.0, (w - 1) / 2.0],
        [0.0, focal_px, (h - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ])


def focal_px_from_35mm(focal_35mm: float, image_size: tuple[int, int]) -> float:
    w, h = image_size
    return focal_35mm * math.hypot(w, h) / _FULL_FRAME_DIAGONAL_MM


def focal_px_from_exif(image: Image.Image) -> float | None:
    """Focal length in pixels from the 35 mm-equivalent EXIF tag, if present.

    Phones write this tag reliably. The raw FocalLength tag is not used: it
    needs the physical sensor size, which EXIF does not record.
    """
    exif = image.getexif()
    f35 = exif.get_ifd(_EXIF_IFD).get(_TAG_FOCAL_LENGTH_35MM) or exif.get(_TAG_FOCAL_LENGTH_35MM)
    if not f35:
        return None
    return focal_px_from_35mm(float(f35), image.size)


@dataclass(frozen=True)
class Camera:
    """A camera looking at the chain plane z = 0 (fiducial side facing +z)."""
    K: np.ndarray
    R: np.ndarray  # world -> camera rotation (rows are camera x, y, z axes)
    t: np.ndarray  # world -> camera translation
    image_size: tuple[int, int]

    @property
    def plane_homography(self) -> np.ndarray:
        """3x3 map from plane coordinates (mm) to image pixels."""
        return self.K @ np.c_[self.R[:, 0], self.R[:, 1], self.t]

    def project(self, pts_mm: np.ndarray) -> np.ndarray:
        return apply_homography(self.plane_homography, pts_mm)


def look_at_plane(
    focal_px: float,
    image_size: tuple[int, int],
    distance_mm: float,
    tilt_deg: float = 0.0,
    tilt_direction_deg: float = 0.0,
    roll_deg: float = 0.0,
    target_mm: tuple[float, float] = (0.0, 0.0),
) -> Camera:
    """Camera aimed at target_mm on the plane from distance_mm away.

    tilt_deg is the angle between the optical axis and the plane normal;
    tilt_direction_deg is the in-plane direction the camera leans from.
    At zero tilt and roll the image x axis is plane +x and image "up" is plane +y.
    """
    th, ph = math.radians(tilt_deg), math.radians(tilt_direction_deg)
    toward_camera = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    center = np.array([target_mm[0], target_mm[1], 0.0]) + distance_mm * toward_camera

    z_c = -toward_camera
    x_c = np.cross([0.0, -1.0, 0.0], z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    r = math.radians(roll_deg)
    x_c, y_c = math.cos(r) * x_c + math.sin(r) * y_c, -math.sin(r) * x_c + math.cos(r) * y_c

    R = np.vstack([x_c, y_c, z_c])
    return Camera(K=intrinsics(focal_px, image_size), R=R, t=-R @ center, image_size=image_size)


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    v = np.c_[pts, np.ones(len(pts))] @ H.T
    return v[:, :2] / v[:, 2:]
