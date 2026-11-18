"""The decode check's test photo and what Python makes of it
(spikes/decode-check/): the default chain (the snap-fit one) in a 2000x1500
colour JPEG with 4:2:0 chroma, stored on its side with EXIF orientation 6 as a
phone held upright stores it; the gray image Pillow decodes from it (as a PNG,
lossless); and Python's detections and pins. The page measures a person's own
photos with the same chain.

    uv run python spikes/decode-check/make.py
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "e2e"))

from decode_scene import exif, jpeg, tinted  # noqa: E402

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import default_chain_path, load_chain_spec  # noqa: E402
from splinewire.detect import detect_fiducials  # noqa: E402
from splinewire.pipeline import load_photo_bytes, measure  # noqa: E402
from splinewire.synthetic import render_snap_photo, s_curve_pins  # noqa: E402


def main() -> None:
    spec = load_chain_spec(default_chain_path())
    size = (2000, 1500)
    pins = s_curve_pins(spec)
    cam = look_at_plane(focal_px_from_35mm(26, size), size, 230.0, tilt_deg=25.0, tilt_direction_deg=35.0,
                        roll_deg=10.0, target_mm=tuple(pins.mean(axis=0)))
    color = tinted(render_snap_photo(pins, spec, cam, rng=np.random.default_rng(5)), 6)
    data = jpeg(Image.fromarray(color).transpose(Image.Transpose.ROTATE_90), exif=exif(26, 6))
    (HERE / "photo.jpg").write_bytes(data)
    img, focal = load_photo_bytes(data, "photo.jpg")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    (HERE / "pillow-gray.png").write_bytes(buf.getvalue())
    fids = detect_fiducials(img, spec)
    m = measure(img, spec, focal, detected=fids)
    (HERE / "python.json").write_text(json.dumps({
        "chain": asdict(spec), "width": img.shape[1], "height": img.shape[0], "focal_px": focal,
        "fiducials": len(fids), "pins_px": m.pins_px.tolist(), "pins_mm": m.pins_mm.tolist(),
    }, indent=1) + "\n", encoding="utf-8")
    print(f"photo.jpg {len(data) // 1024} KiB, pillow-gray.png {len(buf.getvalue()) // 1024} KiB, "
          f"{len(fids)} fiducials, {len(m.order.indices)} pins")


if __name__ == "__main__":
    main()
