import cv2
import numpy as np
import pytest

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.detect import detect_dots, detect_fiducials, detect_rings
from splinewire.synthetic import s_curve_pins, render_photo

SIZE = (2000, 1500)


def _photo(spec, tilt):
    pins = s_curve_pins(spec)
    cam = look_at_plane(focal_px_from_35mm(26, SIZE), SIZE, 180.0, tilt_deg=tilt,
                        tilt_direction_deg=60.0, roll_deg=20.0, target_mm=tuple(pins.mean(axis=0)))
    return render_photo(pins, spec, cam), cam.project(pins)


@pytest.mark.parametrize("tilt", [0.0, 25.0, 40.0])
def test_finds_every_fiducial_with_subpixel_accuracy(spec, tilt):
    img, truth = _photo(spec, tilt)
    rings = detect_fiducials(img, spec)
    assert len(rings) == spec.n_pins
    centers = np.array([r.center_px for r in rings])
    err = np.linalg.norm(centers[:, None] - truth[None], axis=2).min(axis=0)
    assert err.max() < 0.3


def test_dark_ring_on_light_background_is_found():
    img = np.full((400, 400), 220, np.uint8)
    cv2.circle(img, (200, 200), 40, 30, -1, cv2.LINE_AA)
    cv2.circle(img, (200, 200), 20, 220, -1, cv2.LINE_AA)
    rings = detect_rings(img, 0.5)
    assert len(rings) == 1
    np.testing.assert_allclose(rings[0].center_px, (200, 200), atol=0.2)


def test_shapes_that_are_not_our_ring_are_ignored():
    img = np.full((400, 600), 220, np.uint8)
    cv2.circle(img, (100, 200), 40, 30, -1, cv2.LINE_AA)            # solid dot
    cv2.circle(img, (300, 200), 40, 30, -1, cv2.LINE_AA)            # thin ring: wrong ratio
    cv2.circle(img, (300, 200), 38, 220, -1, cv2.LINE_AA)
    cv2.rectangle(img, (440, 160), (520, 240), 30, -1)               # square with square hole
    cv2.rectangle(img, (460, 180), (500, 220), 220, -1)
    assert detect_rings(img, 0.5) == []


def test_noise_specks_inside_the_ring_do_not_hide_it():
    # Real photos threshold into rings with a few tiny dark specks, which
    # appear as extra holes. They must not count as the ring's hole.
    img = np.full((400, 400), 220, np.uint8)
    cv2.circle(img, (200, 200), 40, 30, -1, cv2.LINE_AA)
    cv2.circle(img, (200, 200), 20, 220, -1, cv2.LINE_AA)   # light ring on a dark disc...
    img = 255 - img                                          # ...so dark specks land in a light ring
    for x, y in [(200, 170), (228, 205), (185, 228)]:
        img[y:y + 3, x:x + 3] = 0
    rings = detect_rings(img, 0.5)
    assert len(rings) == 1
    np.testing.assert_allclose(rings[0].center_px, (200, 200), atol=0.3)


def test_dot_on_a_dark_link_is_found_to_subpixel():
    img = np.full((400, 400), 200, np.uint8)
    cv2.circle(img, (200 * 16 + 5, 200 * 16 - 3), 64 * 16, 40, -1, cv2.LINE_AA, 4)   # the link
    cv2.circle(img, (200 * 16 + 5, 200 * 16 - 3), 40 * 16, 230, -1, cv2.LINE_AA, 4)  # the dot
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    dots = detect_dots(img)
    assert len(dots) == 1
    np.testing.assert_allclose(dots[0].center_px, (200 + 5 / 16, 200 - 3 / 16), atol=0.05)
    assert dots[0].outer_axes_px[0] == pytest.approx(80, abs=1.5)


def test_bright_blobs_without_a_steady_dark_margin_are_not_dots():
    """A dot has no hole, so it is told from other bright blobs by a dark
    margin all round it. A chip half on a dark patch and a ring (dark
    middle) are rejected here. (A white disc on a plain grey table does pass:
    locally it is a dot; ordering and the chain's geometry reject those.)"""
    img = np.full((400, 900), 150, np.uint8)
    cv2.rectangle(img, (250, 100), (330, 300), 40, -1)                # dark patch...
    cv2.circle(img, (330, 200), 40, 235, -1, cv2.LINE_AA)            # ...a chip half on it
    cv2.circle(img, (600, 200), 70, 40, -1, cv2.LINE_AA)             # a ring on a dark disc
    cv2.circle(img, (600, 200), 40, 235, -1, cv2.LINE_AA)
    cv2.circle(img, (600, 200), 20, 40, -1, cv2.LINE_AA)
    assert detect_dots(img) == []


def test_dot_crossed_by_a_shadow_edge_is_found():
    """A cast shadow over part of a dot, with a ~1.5 mm penumbra (this dot is
    5 mm = 80 px). A knife-edge shadow is found too but can pull the centre
    by ~2 px; real shadows at phone distance are this soft or softer."""
    img = np.full((400, 400), 200, np.uint8)
    cv2.circle(img, (200, 200), 64, 40, -1, cv2.LINE_AA)
    cv2.circle(img, (200, 200), 40, 230, -1, cv2.LINE_AA)
    ramp = np.clip((np.arange(400) - 210) / 25.0 + 0.5, 0, 1)
    shade = np.tile(1 - 0.55 * ramp, (400, 1)).astype(np.float32)
    img = np.round(cv2.GaussianBlur(img * shade, (0, 0), 1.0)).astype(np.uint8)
    dots = detect_dots(img)
    assert len(dots) == 1
    np.testing.assert_allclose(dots[0].center_px, (200, 200), atol=0.6)
