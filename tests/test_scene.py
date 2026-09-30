"""The realistic photo simulator: its truth must be right, or every
robustness number built on it is wrong."""
from dataclasses import replace

import numpy as np
import pytest

from splinewire.detect import detect_rings
from splinewire.fiducials import ARUCO, BULLSEYE, RING_X, marker_cells, ring_design
from splinewire.scene import (
    BASE, PERFECT_PRINT, PRESETS, PrintQuality, printed_fiducial, random_environment, render_scene,
)
from splinewire.synthetic import s_curve_pins

# 200 mm from the chain, like a real phone photo. Much closer, the perspective
# bias of ellipse centres (~f (r/Z)^2) grows past the tolerances used here.
SMALL = dict(image_size=(1600, 1200), px_per_mm=6.0)


def _centres_error(scene, spec):
    rings = detect_rings(scene.image, spec.ring_inner_mm / spec.ring_outer_mm)
    c = np.array([r.center_px for r in rings]).reshape(-1, 2)
    d = np.linalg.norm(scene.pins_px[:, None] - c[None], axis=2).min(axis=1)
    return d


def test_perfect_print_rings_sit_on_the_true_pins(spec):
    env = replace(PRESETS["ideal"], **SMALL, tilt_deg=30.0)
    scene = render_scene(s_curve_pins(spec), spec, env, seed=1)
    assert _centres_error(scene, spec).max() < 0.15


def test_relief_makes_rings_appear_at_half_its_height(spec):
    """Walls hide the same share of every window edge: the pattern looks as if
    it lay at half the relief height, which is where pins_px is projected."""
    pq = replace(PERFECT_PRINT, relief_mm=0.8)
    env = replace(PRESETS["ideal"], **SMALL, tilt_deg=35.0, print_quality=pq)
    scene = render_scene(s_curve_pins(spec), spec, env, seed=2)
    assert _centres_error(scene, spec).max() < 0.2


def test_lens_distortion_moves_the_truth_with_the_image(spec):
    env = replace(PRESETS["ideal"], **SMALL, distortion_k1=0.05, offset_frac=(0.4, 0.3))
    scene = render_scene(s_curve_pins(spec), spec, env, seed=3)
    assert _centres_error(scene, spec).max() < 0.2


def test_every_preset_and_random_environment_renders(spec):
    rng = np.random.default_rng(0)
    envs = list(PRESETS.values()) + [random_environment(rng) for _ in range(3)]
    for env in envs:
        small = replace(env, image_size=(640, 480), px_per_mm=min(env.px_per_mm, 3.5))
        img = render_scene(s_curve_pins(spec), spec, small, seed=0).image
        assert img.shape == (480, 640) and img.dtype == np.uint8
        assert 5 < img.mean() < 250, env.name


def test_print_model_rounds_black_corners_and_keeps_window_corners(spec):
    """A 0.4 mm nozzle rounds convex black corners (the checker's black
    quadrants meeting at the pin) but the white windows keep sharp corners."""
    cover, tau = printed_fiducial(spec, RING_X, PrintQuality(corner_radius_mm=0.2, gap_close_mm=0.0,
                                                             edge_offset_mm=0.0, wobble_mm=0.0, seam_mm=0.0))
    c = cover.shape[0] // 2
    # 0.1 mm out along the black quadrants' diagonal: black in the design, white once rounded
    k = int(round(0.1 / np.sqrt(2) * tau))
    assert cover[c + k, c + k] < 0.5 and cover[c - k, c - k] < 0.5
    # well inside a black quadrant it is black
    k = int(round(0.8 / np.sqrt(2) * tau))
    assert cover[c + k, c + k] > 0.5


def test_print_model_closes_gaps_narrower_than_a_line(spec):
    thin = replace(BULLSEYE, windows=(type(BULLSEYE.windows[0])(3.0, 2.85),))   # a 0.15 mm white ring
    cover, tau = printed_fiducial(spec, thin, PrintQuality(gap_close_mm=0.1, edge_offset_mm=0.0,
                                                           wobble_mm=0.0, seam_mm=0.0))
    c = cover.shape[0] // 2
    assert cover[c, c + int(round(2.93 * tau))] > 0.5          # filled in: black


def test_bullseye_bands_survive_a_normal_print(spec):
    cover, tau = printed_fiducial(spec, BULLSEYE, PrintQuality(wobble_mm=0.0, seam_mm=0.0))
    c = cover.shape[0] // 2
    radial = cover[c, c:c + int(3.5 * tau)]
    # white 3.0-2.2, black 2.2-1.4, white 1.4-0.6, black centre: four transitions
    assert np.count_nonzero(np.diff(radial > 0.5)) == 4


def test_aruco_marker_is_printed_inverted_and_unmirrored(spec):
    """The printed marker's white cells are the standard marker's black ones,
    oriented to read correctly from above."""
    cells = marker_cells(ARUCO.windows[0], 5)
    assert cells[0].all() and cells[-1].all() and cells[:, 0].all() and cells[:, -1].all()  # white border
    cover, tau = printed_fiducial(spec, ARUCO, PERFECT_PRINT)
    n = cells.shape[0]
    cell = ARUCO.windows[0].size_mm / n
    c = cover.shape[0] // 2
    # the middle pin of printed_fiducial is pin 1
    expect = marker_cells(ARUCO.windows[0], 1)
    for i in range(n):
        for j in range(n):
            row = c + int(round((-ARUCO.windows[0].size_mm / 2 + (i + 0.5) * cell) * tau))
            col = c + int(round((-ARUCO.windows[0].size_mm / 2 + (j + 0.5) * cell) * tau))
            assert (cover[row, col] < 0.5) == expect[i, j]


@pytest.mark.parametrize("design", [ring_design, lambda s: BULLSEYE])
def test_designs_fit_inside_the_link(spec, design):
    d = design(spec)
    assert d.outer_mm / 2 < spec.half_width_mm - 0.5


def test_base_environment_is_a_plain_good_photo():
    assert BASE.background == "paper" and BASE.shadow is None and BASE.clutter == 0
