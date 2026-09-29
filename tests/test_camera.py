import numpy as np
import pytest
from PIL import Image

from splinewire.camera import focal_px_from_35mm, focal_px_from_exif, look_at_plane


def test_straight_on_camera_keeps_plane_orientation():
    cam = look_at_plane(1000.0, (1001, 801), distance_mm=100.0)
    center, right, up = cam.project(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(center, [500.0, 400.0])
    np.testing.assert_allclose(right - center, [10.0, 0.0])   # 1000 px / 100 mm
    np.testing.assert_allclose(up - center, [0.0, -10.0])     # plane +y is image up


def test_tilted_camera_foreshortens_the_far_side():
    cam = look_at_plane(1000.0, (1000, 1000), distance_mm=100.0, tilt_deg=30.0, tilt_direction_deg=-90.0)
    near, far = cam.project(np.array([[0.0, -10.0], [0.0, 10.0]]))
    center = cam.project(np.array([[0.0, 0.0]]))[0]
    assert np.linalg.norm(far - center) < np.linalg.norm(near - center)


def test_focal_from_35mm_uses_diagonal():
    # 43.27 mm is the full-frame diagonal, so f35 = 43.27 maps to one image diagonal
    assert focal_px_from_35mm(np.hypot(36, 24), (3000, 4000)) == pytest.approx(5000.0)


def test_focal_from_exif(tmp_path):
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    path = tmp_path / "p.jpg"
    Image.new("L", (400, 300)).save(path, exif=exif)
    with Image.open(path) as im:
        assert focal_px_from_exif(im) == pytest.approx(focal_px_from_35mm(26, (400, 300)))


def test_focal_from_exif_missing(tmp_path):
    path = tmp_path / "p.jpg"
    Image.new("L", (40, 30)).save(path)
    with Image.open(path) as im:
        assert focal_px_from_exif(im) is None
