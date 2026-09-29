"""How accurately does the pin pitch alone deskew a photo?

Poses a 12-link chain in several shapes, photographs it with a tilted
simulated phone camera, adds Gaussian noise to the pin centers, and compares
three ways of recovering the shape:

  top-down   assume the photo is straight-on; scale from the median pitch
  known f    rectify_chain with the true focal length (as from EXIF)
  unknown f  rectify_chain estimating the focal length as well

Prints the median over trials of the worst pin error in mm.

    uv run python experiments/self_rectification.py
"""
from __future__ import annotations

import numpy as np

from splinewire.camera import look_at_plane
from splinewire.chain import pins_from_turns
from splinewire.pipeline import rigid_fit_errors
from splinewire.rectify import rectify_chain

PITCH = 10.0
SIZE = (2000, 1500)
FOCAL = 1500.0          # ~26 mm equivalent for this image size
DISTANCE = 180.0        # mm; about 8 px per mm on the chain
TRIALS = 20            # takes ~5 min, mostly the unknown-f solves

SHAPES = {
    "straight": np.zeros(11),
    "gentle arc R=300": np.full(11, PITCH / 300),
    "arc R=60": np.full(11, PITCH / 60),
    "pipe R=25": np.full(11, PITCH / 25),
    "S-curve": np.radians(np.r_[np.full(5, 14.0), np.full(6, -14.0)]),
}


def top_down(px: np.ndarray) -> np.ndarray:
    scale = PITCH / np.median(np.linalg.norm(np.diff(px, axis=0), axis=1))
    return (px - px[0]) * [scale, -scale]


def main() -> None:
    rng = np.random.default_rng(0)
    links = [(i, i + 1) for i in range(12)]
    print(f"{'shape':18s} {'tilt':>4s} {'noise':>6s} | {'top-down':>8s} {'known f':>8s} {'unknown f':>9s}")
    for name, turns in SHAPES.items():
        pins = pins_from_turns(PITCH, turns)
        for tilt in (10.0, 25.0):
            cam = look_at_plane(FOCAL, SIZE, DISTANCE, tilt_deg=tilt, tilt_direction_deg=30.0,
                                roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
            px_true = cam.project(pins)
            for noise in (0.5, 1.0, 3.0):
                errs = {"top": [], "known": [], "unknown": []}
                for _ in range(TRIALS):
                    px = px_true + rng.normal(0, noise, px_true.shape)
                    errs["top"].append(rigid_fit_errors(top_down(px), pins).max())
                    for key, f in (("known", FOCAL), ("unknown", None)):
                        r = rectify_chain(px, links, PITCH, SIZE, f)
                        errs[key].append(rigid_fit_errors(r.pins_mm, pins).max())
                med = {k: np.median(v) for k, v in errs.items()}
                print(f"{name:18s} {tilt:4.0f} {noise:5.1f}px | {med['top']:8.2f} "
                      f"{med['known']:8.2f} {med['unknown']:9.2f}")


if __name__ == "__main__":
    main()
