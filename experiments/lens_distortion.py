"""How much does residual lens distortion cost, and can the deskew absorb it?

Phones correct their main camera's distortion in-camera, but not perfectly,
and the correction is weaker toward the frame's corners. Residual radial
distortion (x_ideal = x_image * (1 + k1 r^2), r in focal lengths) bends a
straight chain and changes link lengths across the frame.

This projects chains through a camera with distortion k1, adds 0.1 px of
centre noise, and deskews them two ways: ignoring distortion (the default),
and with k1 as one more unknown in the pitch solve. Prints the median over
trials of the worst pin error in mm.

    uv run python experiments/lens_distortion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import pins_from_turns  # noqa: E402
from splinewire.pipeline import rigid_fit_errors  # noqa: E402
from splinewire.rectify import rectify_chain  # noqa: E402

PITCH = 10.0
SIZE = (4032, 3024)
FOCAL = focal_px_from_35mm(26, SIZE)
TRIALS = 12
SHAPES = {
    "straight": np.zeros(11),
    "arc R=60": np.full(11, PITCH / 60),
    "S-curve": np.radians(np.r_[np.full(5, 14.0), np.full(6, -14.0)]),
}


def distort(px: np.ndarray, k1: float) -> np.ndarray:
    """Ideal pixels -> where a lens with residual k1 puts them."""
    c = np.array([(SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2])
    xu = (px - c) / FOCAL
    xd = xu.copy()
    for _ in range(30):
        xd = xu / (1 + k1 * np.sum(xd ** 2, axis=1, keepdims=True))
    return xd * FOCAL + c


def main() -> None:
    rng = np.random.default_rng(0)
    links = [(i, i + 1) for i in range(12)]
    print(f"{'shape':10s} {'where':>10s} {'k1':>6s} | {'ignore k1':>9s} {'solve k1':>9s}   (worst pin error, mm)")
    for name, turns in SHAPES.items():
        pins = pins_from_turns(PITCH, turns)
        for where, offset in (("centre", (0.0, 0.0)), ("corner", (0.35, 0.25))):
            for k1 in (0.0, 0.005, 0.01, 0.02, 0.04):
                errs = {"ignore": [], "solve": []}
                for t in range(TRIALS):
                    cam = look_at_plane(FOCAL, SIZE, 300.0, tilt_deg=20.0, tilt_direction_deg=30.0 + 40 * t,
                                        roll_deg=10.0, target_mm=tuple(pins.mean(axis=0) - np.array(offset) * 180))
                    px = distort(cam.project(pins), k1) + rng.normal(0, 0.1, (len(pins), 2))
                    for key, est in (("ignore", False), ("solve", True)):
                        r = rectify_chain(px, links, PITCH, SIZE, FOCAL, estimate_distortion=est)
                        errs[key].append(rigid_fit_errors(r.pins_mm, pins).max())
                print(f"{name:10s} {where:>10s} {k1:6.3f} | {np.median(errs['ignore']):9.3f} "
                      f"{np.median(errs['solve']):9.3f}")


if __name__ == "__main__":
    main()
