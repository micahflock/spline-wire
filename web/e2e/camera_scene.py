"""A chain for Chromium's fake camera: the camera view's end-to-end test
(e2e/camera.spec.ts) streams it as if it were a phone's rear camera.

    uv run python web/e2e/camera_scene.py OUT_DIR [WIDTHxHEIGHT] [snap [SPAN [dark] [tilt=DEG] [orange]]]
        -> OUT_DIR/camera.y4m   two identical frames (Chromium loops the file)
           OUT_DIR/camera.json  pins in frame pixels, the focal length, the chain,
                                and move_dir: the way (frame axes, y down) the
                                camera must move to look straight down (snap)
           OUT_DIR/camera.jpg   the same frame as a phone's JPEG with its EXIF focal
                                length, for the upload path

Portrait 3024x4032 by default: the full 4:3 sensor an iPhone streams (S3).
The chain spans about 0.55 of the short side (the snap-fit chain: SPAN, its
pins' greatest distance plus a link's width over the short side) and is seen
12 degrees off straight-on (the snap chain: tilt=DEG for another tilt), in
the "daylight" scene of splinewire.scene. With `orange`, the snap chain in
colour: orange rims on off-white links on a dark table (dark rims, so its
chain says fiducial_shade "dark"), the case of ChainSpec gray_channel, with
the video's chroma 4:2:0 as a camera's is.
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
from splinewire.synthetic import SNAP_RGB_NOISE, render_snap_photo, s_curve_pins, save_photo

FOCAL_35MM = 24.0
SPAN = 0.55


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "3024x4032").split("x"))
    if len(sys.argv) > 3 and sys.argv[3] == "snap":
        rest = sys.argv[4:]
        tilt = next((float(a[5:]) for a in rest if a.startswith("tilt=")), 12.0)
        orange = "orange" in rest
        rest = [a for a in rest if not a.startswith("tilt=") and a != "orange"]
        return snap(out, w, h, float(rest[0]) if rest else SPAN, orange or (len(rest) > 1 and rest[1] == "dark"), tilt, orange)
    spec = load_chain_spec(oring_chain_path())
    pins = s_curve_pins(spec)
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    env = replace(PRESETS["daylight"], image_size=(w, h), focal_35mm=FOCAL_35MM, px_per_mm=SPAN * min(w, h) / extent,
                  tilt_deg=12.0, roll_deg=70.0, offset_frac=(0.0, 0.0))
    sc = render_scene(pins, spec, env, seed=3)
    write(out, sc.image.astype(np.uint8), sc.pins_px, sc.focal_px, spec)


def snap(out: Path, w: int, h: int, span: float = SPAN, dark: bool = False, tilt: float = 12.0, orange: bool = False) -> None:
    """The snap-fit chain (data/chain-snap.yaml: thin white rims on round
    sleeves, 13 mm pitch), drawn with its relief by synthetic.render_snap_photo,
    framed and tilted as the dot chain's scene is. With `dark`, printed the other
    way round: dark rims on light links (fiducial_shade "dark")."""
    spec = load_chain_spec(default_chain_path())
    if dark:
        spec = replace(spec, fiducial_shade="dark")
    pins = s_curve_pins(spec)
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    focal = focal_px_from_35mm(FOCAL_35MM, (w, h))
    distance = focal * extent / (span * min(w, h))
    target = tuple(pins.mean(axis=0))
    cam = look_at_plane(focal, (w, h), distance, tilt_deg=tilt, tilt_direction_deg=30.0, roll_deg=70.0, target_mm=target)
    colours = dict(table_gray=(60, 60, 60), link_gray=(235, 230, 214), rim_gray=(240, 122, 35),
                   noise_gray=SNAP_RGB_NOISE) if orange else {}
    image = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(3), **colours)
    # From where the camera is to the spot as far straight above the chain, in the frame's axes.
    move = cam.R @ (np.array([target[0], target[1], distance]) + cam.R.T @ cam.t)
    write(out, np.clip(image, 0, 255).astype(np.uint8), cam.project(pins), focal, spec,
          move_dir=(move[:2] / np.hypot(*move[:2])).tolist())


def write(out: Path, img: np.ndarray, pins_px: np.ndarray, focal_px: float, spec, move_dir=None) -> None:
    """img: gray (the luma, written as it is, with neutral chroma) or RGB
    (written as BT.601 video, its chroma averaged over 2x2 pixels)."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        y = img
        cb = cr = np.full((h // 2) * (w // 2), 128, np.uint8).tobytes()
    else:
        r, g, b = (img[..., k].astype(np.float64) for k in range(3))
        y = np.clip(np.round(16 + 219 / 255 * (0.299 * r + 0.587 * g + 0.114 * b)), 0, 255).astype(np.uint8)
        half = lambda c: c[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        cb, cr = (np.clip(np.round(128 + 224 / 255 * half(c)), 0, 255).astype(np.uint8).tobytes() for c in (
            -0.168736 * r - 0.331264 * g + 0.5 * b, 0.5 * r - 0.418688 * g - 0.081312 * b))
    with open(out / "camera.y4m", "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for _ in range(2):
            f.write(b"FRAME\n")
            f.write(y.tobytes())
            f.write(cb)
            f.write(cr)
    save_photo(out / "camera.jpg", img, FOCAL_35MM)
    (out / "camera.json").write_text(json.dumps({
        "size": [w, h], "pins_px": np.asarray(pins_px).tolist(), "focal_px": focal_px,
        "focal_35mm": focal_35mm_from_px(focal_px, (w, h)), "chain": asdict(spec), "move_dir": move_dir,
    }))


if __name__ == "__main__":
    main()
