"""A chain for Chromium's fake camera: the camera view's end-to-end test
(e2e/camera.spec.ts) streams it as if it were a phone's rear camera.

    uv run python web/e2e/camera_scene.py OUT_DIR [WIDTHxHEIGHT] [snap]
        -> OUT_DIR/camera.y4m   two identical frames (Chromium loops the file)
           OUT_DIR/camera.json  pins in frame pixels, the focal length, the chain
           OUT_DIR/camera.jpg   the same frame as a phone's JPEG with its EXIF focal
                                length, for the upload path

Portrait 3024x4032 by default: the full 4:3 sensor an iPhone streams (S3).
The chain spans about 0.55 of the short side and is seen 12 degrees off
straight-on, in the "daylight" scene of splinewire.scene.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from splinewire.camera import focal_35mm_from_px, focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec, oring_chain_path
from splinewire.scene import PRESETS, render_scene
from splinewire.synthetic import render_snap_photo, s_curve_pins, save_photo

FOCAL_35MM = 24.0
SPAN = 0.55


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "3024x4032").split("x"))
    if len(sys.argv) > 3 and sys.argv[3] == "snap":
        return snap(out, w, h)
    spec = load_chain_spec(oring_chain_path())
    pins = s_curve_pins(spec)
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    env = replace(PRESETS["daylight"], image_size=(w, h), focal_35mm=FOCAL_35MM, px_per_mm=SPAN * min(w, h) / extent,
                  tilt_deg=12.0, roll_deg=70.0, offset_frac=(0.0, 0.0))
    sc = render_scene(pins, spec, env, seed=3)
    write(out, sc.image.astype(np.uint8), sc.pins_px, sc.focal_px, spec)


def snap(out: Path, w: int, h: int) -> None:
    """The snap-fit chain (data/chain-snap.yaml: thin white rims on round
    sleeves, 13 mm pitch), drawn with its relief by synthetic.render_snap_photo,
    framed and tilted as the dot chain's scene is."""
    spec = load_chain_spec(default_chain_path())
    pins = s_curve_pins(spec)
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    focal = focal_px_from_35mm(FOCAL_35MM, (w, h))
    distance = focal * extent / (SPAN * min(w, h))
    cam = look_at_plane(focal, (w, h), distance, tilt_deg=12.0, tilt_direction_deg=30.0, roll_deg=70.0,
                        target_mm=tuple(pins.mean(axis=0)))
    image = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(3))
    write(out, np.clip(image, 0, 255).astype(np.uint8), cam.project(pins), focal, spec)


def write(out: Path, y: np.ndarray, pins_px: np.ndarray, focal_px: float, spec) -> None:
    h, w = y.shape
    chroma = np.full((h // 2) * (w // 2), 128, np.uint8).tobytes()
    with open(out / "camera.y4m", "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for _ in range(2):
            f.write(b"FRAME\n")
            f.write(y.tobytes())
            f.write(chroma)
            f.write(chroma)
    save_photo(out / "camera.jpg", y, FOCAL_35MM)
    (out / "camera.json").write_text(json.dumps({
        "size": [w, h], "pins_px": np.asarray(pins_px).tolist(), "focal_px": focal_px,
        "focal_35mm": focal_35mm_from_px(focal_px, (w, h)), "chain": asdict(spec),
    }))


if __name__ == "__main__":
    main()
