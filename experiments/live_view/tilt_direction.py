"""Can the live view tell which way the phone should move to look straight down?

The live analysis (splinewire/live.py, web/src/pipeline/live.ts) fits the
chain's plane to its pins every few hundred ms and reports the tilt angle;
`tilt_dir` is the way across the frame to the plane's far side, where the
camera should move (still aimed at the chain) to look straight down on it.
Here the snap-fit chain's pins are projected exactly through a phone camera
(3024x4032, 24 mm equivalent, the chain spanning SPAN of the short side) at
known tilts, in 12 directions with a roll each, plus Gaussian noise on each
pin's centre, and `tilt_dir` is compared with the truth: the camera's offset
from straight overhead, seen in the frame.

Noise is given in pitches. A live rim is drawn about 12 work px across and a
pitch about 22 (region.ts REGIONS.target), so 0.3 work px of centre noise is
0.014 pitch; 0.03 is twice that and more.

Part 1, one reading: per tilt and noise, the median and 90th-percentile angle
between the reading's direction and the truth, and the share of readings more
than 45 degrees off (the wrong way, to the eye) or with none. Focal length
known, 5% off, and estimated from the chain (slow: ~15 min in all).

Part 2, the arrow the coach shows (web/src/overlay/coach.ts nextArrow, its
limits copied here): runs of READINGS readings at each pose, fresh noise each,
focal known. Per tilt and noise: the share of readings showing an arrow, and
of those, the share more than 45 degrees off and the median angle off.

    uv run python experiments/live_view/tilt_direction.py          # both parts
    uv run python experiments/live_view/tilt_direction.py arrow    # part 2 only (~10 min)
"""
from __future__ import annotations

import math
import sys

import numpy as np

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.live import analyse
from splinewire.synthetic import circle_wrap_pins, s_curve_pins

W, H = 3024, 4032
SPAN = 0.30
TILTS = [3, 5, 8, 12, 16, 20, 25, 30, 40]
DIRECTIONS = range(0, 360, 30)
NOISE_PITCH = [0.0, 0.014, 0.03]
FOCALS = {"known": 1.0, "5% off": 1.05, "estimated": None}


def shapes(spec):
    return {
        "s-curve": s_curve_pins(spec),
        "arc r40": circle_wrap_pins(spec, 40.0),
        "gentle": s_curve_pins(spec, bend_deg=4.0),
    }


def truth_dir(cam, target, distance) -> np.ndarray:
    """The way the camera must move, in the frame's axes (x right, y down): from
    where it is to the spot the same distance straight above the target."""
    centre = -cam.R.T @ cam.t
    overhead = np.array([target[0], target[1], distance])
    v = cam.R @ (overhead - centre)
    return v[:2] / np.hypot(*v[:2])


# coach.ts COACH.arrowDeg, arrowAgree, arrowWeight
ARROW_DEG = (12.0, 9.0)
ARROW_AGREE = (0.6, 0.4)
ARROW_WEIGHT = 0.4
READINGS = 8


def next_arrow(mean, shown, tilt_deg, tilt_dir):
    """coach.ts nextArrow, for a chain in view: (mean, shown direction or None)."""
    if tilt_deg is None:
        return np.zeros(2), None
    if tilt_dir is None:
        return mean, None
    mean = (1 - ARROW_WEIGHT) * mean + ARROW_WEIGHT * np.asarray(tilt_dir)
    agree = float(np.hypot(*mean))
    k = 1 if shown is not None else 0
    show = tilt_deg > ARROW_DEG[k] and agree > ARROW_AGREE[k]
    return mean, (mean / agree if show else None)


def poses(spec, focal):
    for shape, pins in shapes(spec).items():
        extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
        yield shape, pins, focal * extent / (SPAN * min(W, H)), tuple(pins.mean(axis=0))


def reading(spec, pins, cam, noise, rng, focal_px):
    px = cam.project(pins)
    pitch_px = float(np.median(np.linalg.norm(np.diff(px, axis=0), axis=1)))
    px = px + rng.normal(0, noise * pitch_px, px.shape)
    d = np.full(len(px), pitch_px * spec.fiducial_mm / spec.pitch_mm)
    return analyse(px, d, (W, H), spec, focal_px)


def off_deg(v, t) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, t))))))


def arrow_part(spec, focal) -> None:
    rng = np.random.default_rng(2)
    print(f"\nthe arrow, {READINGS} readings a pose, focal known: shown / of those >45 deg off / median off")
    for shape, pins, distance, target in poses(spec, focal):
        print(f"\n{shape}")
        print("tilt  " + "   ".join(f"noise {n:.3f} pitch       " for n in NOISE_PITCH[1:]))
        for tilt in TILTS:
            row = []
            for noise in NOISE_PITCH[1:]:
                shown = wrong = 0
                offs = []
                for k, direction in enumerate(DIRECTIONS):
                    cam = look_at_plane(focal, (W, H), distance, tilt_deg=tilt, tilt_direction_deg=direction,
                                        roll_deg=37.0 * k, target_mm=target)
                    t = truth_dir(cam, target, distance)
                    mean, arrow = np.zeros(2), None
                    for _ in range(READINGS):
                        out = reading(spec, pins, cam, noise, rng, focal)
                        mean, arrow = next_arrow(mean, arrow, out["tilt_deg"], out["tilt_dir"])
                        if arrow is not None:
                            shown += 1
                            offs.append(off_deg(arrow, t))
                            wrong += offs[-1] > 45
                n = len(DIRECTIONS) * READINGS
                row.append(f"{shown / n:4.0%} / {wrong / max(shown, 1):4.0%} / {np.median(offs) if offs else float('nan'):4.0f}")
            print(f"{tilt:4d}  " + "     ".join(row))


def main() -> None:
    spec = load_chain_spec(default_chain_path())
    focal = focal_px_from_35mm(24.0, (W, H))
    if sys.argv[1:] == ["arrow"]:
        arrow_part(spec, focal)
        return
    single_part(spec, focal)
    arrow_part(spec, focal)


def single_part(spec, focal) -> None:
    rng = np.random.default_rng(1)
    print(f"snap-fit chain, {spec.n_pins} pins, span {SPAN}; one reading: angle off (median / p90), share >45 deg off or none")
    for shape, pins, distance, target in poses(spec, focal):
        for fname, fk in FOCALS.items():
            print(f"\n{shape}, focal {fname}")
            print("tilt  " + "   ".join(f"noise {n:.3f} pitch     " for n in NOISE_PITCH))
            for tilt in TILTS:
                row = []
                for noise in NOISE_PITCH:
                    errs = []
                    for k, direction in enumerate(DIRECTIONS):
                        cam = look_at_plane(focal, (W, H), distance, tilt_deg=tilt, tilt_direction_deg=direction,
                                            roll_deg=37.0 * k, target_mm=target)
                        out = reading(spec, pins, cam, noise, rng, None if fk is None else focal * fk)
                        errs.append(180.0 if out["tilt_dir"] is None else off_deg(out["tilt_dir"], truth_dir(cam, target, distance)))
                    bad = sum(e > 45 for e in errs)
                    row.append(f"{np.median(errs):5.0f} / {np.percentile(errs, 90):4.0f}  {bad / len(errs):4.0%}")
                print(f"{tilt:4d}  " + "     ".join(row))


if __name__ == "__main__":
    main()
