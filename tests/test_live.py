"""The live overlay's frame analysis: order and rectify blob centres."""
import time
from dataclasses import asdict

import numpy as np

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.live import analyse, analyse_json
from splinewire.synthetic import s_curve_pins


def frame(spec, tilt=25.0, size=(1920, 1080)):
    pins = s_curve_pins(spec)
    f = focal_px_from_35mm(26, size)
    cam = look_at_plane(f, size, 260.0, tilt_deg=tilt, tilt_direction_deg=40,
                        target_mm=tuple(pins.mean(axis=0)))
    pts = cam.project(pins)
    d = np.full(len(pts), spec.fiducial_mm * f / 260.0)
    return pts, d, f, size


def test_finds_the_chain_and_its_tilt_among_clutter(dot_spec):
    pts, d, f, size = frame(dot_spec)
    rng = np.random.default_rng(1)
    clutter = rng.uniform([0, 0], size, (15, 2))
    clutter = clutter[np.linalg.norm(clutter[:, None] - pts[None], axis=2).min(axis=1) > 3 * d[0]]
    allpts = np.vstack([clutter, pts])
    alld = np.r_[rng.uniform(4, 40, len(clutter)), d]
    a = analyse(allpts, alld, size, dot_spec, f)
    found = [i - len(clutter) for i in a["order"]]
    assert found in (list(range(dot_spec.n_pins)), list(range(dot_spec.n_pins))[::-1])
    assert abs(a["tilt_deg"] - 25.0) < 1.0
    assert a["link_residual_max_mm"] < 0.1
    assert a == analyse_json({"points": allpts.tolist(), "diameters": alld.tolist(), "width": size[0],
                              "height": size[1], "chain": asdict(dot_spec), "focal_px": f})


def test_a_frame_without_a_chain_is_not_an_error(dot_spec):
    empty = {"order": [], "gaps": [], "tilt_deg": None, "link_residual_max_mm": None}
    assert analyse([], [], (1920, 1080), dot_spec) == empty
    assert analyse([[1, 1], [5, 5]], [3, 3], (1920, 1080), dot_spec) == empty
    assert analyse(np.zeros((500, 2)), np.ones(500), (1920, 1080), dot_spec) == empty
    part = analyse([[100, 100], [200, 100], [300, 100]], [50, 50, 50], (1920, 1080), dot_spec)
    assert len(part["order"]) == 3 and part["tilt_deg"] is None     # too few links to rectify


def test_is_quick_enough_to_run_a_few_times_a_second(dot_spec):
    pts, d, f, size = frame(dot_spec)
    analyse(pts, d, size, dot_spec, f)
    start = time.perf_counter()
    for _ in range(5):
        analyse(pts, d, size, dot_spec, f)
    known = (time.perf_counter() - start) / 5
    start = time.perf_counter()
    analyse(pts, d, size, dot_spec, None)
    print(f"analyse: {known * 1000:.0f} ms with the focal length, "
          f"{(time.perf_counter() - start) * 1000:.0f} ms estimating it")
    assert known < 0.25
