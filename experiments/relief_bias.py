"""Does the printed plaque's relief bias the measurement?

On the printed plaque the white rings are windows in a black layer, so
they sit at the bottom of a shallow recess. At a tilt, a recess wall hides
part of each ring. This ray-casts tilted photos of the plaque with its real
3D relief (black top at height h, white floor at 0, black walls between),
runs the normal pipeline, and compares the error with a flat print (h = 0).

Most of the shift is the same for every ring (all rings are seen from
about the same direction), so it moves the whole chain rather than
distorting it; only the variation across the photo matters.

    uv run python experiments/relief_bias.py
"""
from __future__ import annotations

import cv2
import numpy as np
import shapely

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import load_chain_spec, default_chain_path
from splinewire.pipeline import compare_to_truth, measure
from splinewire.plaque import plaque_geometry
from splinewire.synthetic import s_curve_pins

SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)
TAU = 40.0               # mask resolution, px per mm
BLACK, WHITE = 40.0, 225.0


def rasterize(geom, bounds, tau):
    x0, y0, x1, y1 = bounds
    xs = x0 + np.arange(int((x1 - x0) * tau) + 1) / tau
    ys = y0 + np.arange(int((y1 - y0) * tau) + 1) / tau
    gx, gy = np.meshgrid(xs, ys)
    return shapely.contains_xy(geom, gx, gy)


def render(mask, bounds, cam, height_mm, rng, ss=2, wall_samples=6):
    x0, y0, _, _ = bounds
    w, h = cam.image_size
    K_inv = np.linalg.inv(cam.K)
    center = -cam.R.T @ cam.t
    out = np.empty((h * ss, w * ss))

    def black_at(p):
        ix = np.clip(np.round((p[..., 0] - x0) * TAU).astype(int), 0, mask.shape[1] - 1)
        iy = np.clip(np.round((p[..., 1] - y0) * TAU).astype(int), 0, mask.shape[0] - 1)
        return mask[iy, ix]

    us = (np.arange(w * ss) + 0.5) / ss - 0.5
    for r0 in range(0, h * ss, 256):
        vs = (np.arange(r0, min(r0 + 256, h * ss)) + 0.5) / ss - 0.5
        gu, gv = np.meshgrid(us, vs)
        rays = np.stack([gu, gv, np.ones_like(gu)], axis=-1) @ K_inv.T @ cam.R  # world directions
        t_top = (height_mm - center[2]) / rays[..., 2]
        t_floor = -center[2] / rays[..., 2]
        top = center[:2] + t_top[..., None] * rays[..., :2]
        floor = center[:2] + t_floor[..., None] * rays[..., :2]
        dark = black_at(top)
        if height_mm > 0:
            for k in range(1, wall_samples + 1):   # does the ray hit a wall on its way down?
                f = k / (wall_samples + 1)
                dark |= black_at(top + f * (floor - top))
            dark |= black_at(floor)
        out[r0 - 0:r0 - 0 + len(vs)] = np.where(dark, BLACK, WHITE)
    img = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 3.0, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def main() -> None:
    spec = load_chain_spec(default_chain_path())
    pins = s_curve_pins(spec)
    plaque = plaque_geometry(pins, spec)
    bounds = plaque.plate.bounds
    mask = rasterize(plaque.pattern, bounds, TAU)
    rng = np.random.default_rng(0)

    print(f"{'tilt':>4s} {'lean from':>9s} | " + " | ".join(
        f"h={h:.1f} mm: max err / scaled" for h in (0.0, 0.4, 0.8)))
    for tilt in (0.0, 25.0, 40.0):
        for direction in ((0.0,) if tilt == 0 else (-90.0, 30.0)):
            cam = look_at_plane(FOCAL, SIZE, 180.0, tilt_deg=tilt, tilt_direction_deg=direction,
                                roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
            cells = []
            for height in (0.0, 0.4, 0.8):
                img = render(mask, bounds, cam, height, rng)
                found = measure(img, spec, FOCAL).pins_mm
                if len(found) != len(pins):
                    cells.append(f"found {len(found)}/{len(pins)} pins")
                    continue
                t = compare_to_truth(found, pins)
                cells.append(f"{t['max_error_mm']:.3f} / {t['max_error_scaled_mm']:.3f} mm")
            print(f"{tilt:4.0f} {direction:8.0f}° | " + " | ".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
