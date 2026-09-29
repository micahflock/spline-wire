import cv2
import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.detect import detect_rings
from splinewire.synthetic import s_curve_pins, render_photo

SIZE = (2000, 1500)


def _photo(spec, tilt):
    pins = s_curve_pins(spec)
    cam = look_at_plane(focal_px_from_35mm(26, SIZE), SIZE, 180.0, tilt_deg=tilt,
                        tilt_direction_deg=60.0, roll_deg=20.0, target_mm=tuple(pins.mean(axis=0)))
    return render_photo(pins, spec, cam), cam.project(pins)


@pytest.mark.parametrize("tilt", [0.0, 25.0, 40.0])
def test_finds_every_ring_with_subpixel_accuracy(spec, tilt):
    img, truth = _photo(spec, tilt)
    rings = detect_rings(img, spec.ring_inner_mm / spec.ring_outer_mm)
    assert len(rings) == spec.n_pins
    centers = np.array([r.center_px for r in rings])
    err = np.linalg.norm(centers[:, None] - truth[None], axis=2).min(axis=0)
    assert err.max() < 0.3


def test_dark_ring_on_light_background_is_found(spec):
    img = np.full((400, 400), 220, np.uint8)
    cv2.circle(img, (200, 200), 40, 30, -1, cv2.LINE_AA)
    cv2.circle(img, (200, 200), 20, 220, -1, cv2.LINE_AA)
    rings = detect_rings(img, 0.5)
    assert len(rings) == 1
    np.testing.assert_allclose(rings[0].center_px, (200, 200), atol=0.2)


def test_shapes_that_are_not_our_ring_are_ignored(spec):
    img = np.full((400, 600), 220, np.uint8)
    cv2.circle(img, (100, 200), 40, 30, -1, cv2.LINE_AA)            # solid dot
    cv2.circle(img, (300, 200), 40, 30, -1, cv2.LINE_AA)            # thin ring: wrong ratio
    cv2.circle(img, (300, 200), 36, 220, -1, cv2.LINE_AA)
    cv2.rectangle(img, (440, 160), (520, 240), 30, -1)               # square with square hole
    cv2.rectangle(img, (460, 180), (500, 220), 220, -1)
    assert detect_rings(img, 0.5) == []
