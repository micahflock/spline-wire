"""The native side of e2e/decode.spec.ts: JPEGs as a phone writes them, and
what Python reads from each (the gray pixels Pillow decodes, the focal length
from EXIF) and measures in it, for the browser's decoder to be compared with.

    uv run python web/e2e/decode_scene.py OUT_DIR    -> OUT_DIR/decode.json, *.jpg, *.gray

Photos of the default chain, the snap-fit one (4000x3000): a gray JPEG; a
colour one (chroma subsampled 4:2:0, Pillow's default, as phones write them);
the colour one stored on its side with EXIF orientation 6, as a phone held
upright stores it; and the chain on a black table at 40 degrees' tilt.
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from splinewire.camera import EXIF_IFD, TAG_FOCAL_LENGTH_35MM, focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.detect import detect_fiducials
from splinewire.pipeline import load_photo_bytes, measure
from splinewire.synthetic import circle_wrap_pins, render_snap_photo, s_curve_pins

ORIENTATION = 0x0112


def exif(focal_35mm: int, orientation: int | None = None) -> Image.Exif:
    e = Image.Exif()
    e.get_ifd(EXIF_IFD)[TAG_FOCAL_LENGTH_35MM] = focal_35mm
    if orientation:
        e[ORIENTATION] = orientation
    return e


def tinted(gray: np.ndarray, seed: int) -> np.ndarray:
    """A colour photo whose channels differ as a real one's do: warm light, some colour noise."""
    rng = np.random.default_rng(seed)
    g = gray.astype(float)
    rgb = np.stack([g * 1.02 + 6, g * 0.97 + 2, g * 0.88 - 3], axis=2) + rng.normal(0, 2.0, (*gray.shape, 3))
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def jpeg(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, **kw)
    return buf.getvalue()


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = load_chain_spec(default_chain_path())
    size = (4000, 3000)
    pins = circle_wrap_pins(spec, 40.0)
    cam = look_at_plane(focal_px_from_35mm(26, size), size, 200.0, tilt_deg=25.0, tilt_direction_deg=35.0,
                        roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
    gray = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(1))
    color = tinted(gray, 2)
    s_pins = s_curve_pins(spec)
    steep = look_at_plane(focal_px_from_35mm(26, size), size, 260.0, tilt_deg=40.0, tilt_direction_deg=120.0,
                          roll_deg=-20.0, target_mm=tuple(s_pins.mean(axis=0)))
    black = render_snap_photo(s_pins, spec, steep, rng=np.random.default_rng(3), table_gray=40)
    photos = {
        "gray": jpeg(Image.fromarray(gray), exif=exif(26)),
        "color": jpeg(Image.fromarray(color), exif=exif(26)),
        "rotated": jpeg(Image.fromarray(color).transpose(Image.Transpose.ROTATE_90), exif=exif(26, 6)),
        "black-table": jpeg(Image.fromarray(tinted(black, 4)), exif=exif(26)),
    }
    doc = {"chain": asdict(spec), "photos": {}}
    for name, data in photos.items():
        (out / f"{name}.jpg").write_bytes(data)
        img, focal = load_photo_bytes(data, f"{name}.jpg")
        (out / f"{name}.gray").write_bytes(np.ascontiguousarray(img).tobytes())
        fids = detect_fiducials(img, spec)
        m = measure(img, spec, focal, detected=fids)
        doc["photos"][name] = {
            "width": img.shape[1], "height": img.shape[0], "focal_px": focal,
            "fiducials": [[*f.center_px, *f.outer_axes_px] for f in fids],
            "pins_px": m.pins_px.tolist(), "pins_mm": m.pins_mm.tolist(),
        }
        print(f"{name}: {img.shape[1]}x{img.shape[0]}, {len(fids)} fiducials, {len(m.order.indices)} pins, focal {focal:.1f} px")
    (out / "decode.json").write_text(json.dumps(doc), encoding="utf-8")


if __name__ == "__main__":
    main()
