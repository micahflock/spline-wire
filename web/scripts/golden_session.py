"""PhotoSession documents and output files from Python, for the TypeScript
session's tests (web/src/pipeline/session.test.ts): the committed test photos
(golden/photo-*.png, 8-bit gray, no EXIF) measured with a range of options,
including ones that fail. One is in colour (photo-snap-orange.png, written
here: the snap-fit chain with orange rims on off-white links on a dark table,
the case of ChainSpec gray_channel), measured in gray, where it is not found,
and through its blue channel; golden/channels.json has the SHA-256 of
Pillow's gray of it for every channel (pipeline.gray_from_image).

    uv run python web/scripts/golden_session.py     # after golden_detect.py and golden_edits.py
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import GRAY_CHANNELS, chain_presets, load_chain_spec  # noqa: E402
from splinewire.pipeline import gray_from_image  # noqa: E402
from splinewire.session import PhotoSession  # noqa: E402
from splinewire.synthetic import render_snap_photo, s_curve_pins  # noqa: E402

DIR = REPO / "web" / "src" / "pipeline" / "golden"


ORANGE_PHOTO = "photo-snap-orange.png"


def orange_photo() -> None:
    """The snap-fit chain with orange rims on off-white links on a dark table
    (experiments/colour_channel/), cropped to the chain in multiples of 16 px:
    an RGB PNG, with Pillow's gray of each channel's digest beside it."""
    spec = load_chain_spec(chain_presets()["snap"])
    pins = s_curve_pins(spec)
    size = (1600, 1200)
    cam = look_at_plane(focal_px_from_35mm(26, size), size, 300.0, tilt_deg=12.0, tilt_direction_deg=30.0,
                        roll_deg=5.0, target_mm=tuple(pins.mean(axis=0)))
    rgb = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(5), table_gray=(60, 60, 60),
                            link_gray=(235, 230, 214), rim_gray=(240, 122, 35), noise_gray=(2.0, 1.5, 3.0))
    px = cam.project(pins)
    x0, y0 = (np.maximum(px.min(axis=0) - 70, 0) // 16 * 16).astype(int)
    x1, y1 = np.minimum(px.max(axis=0) + 70, size).astype(int)
    x1, y1 = x0 + (x1 - x0) // 16 * 16, y0 + (y1 - y0) // 16 * 16
    Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1])).save(DIR / ORANGE_PHOTO, optimize=True)
    with Image.open(DIR / ORANGE_PHOTO) as im:
        digests = {ch: hashlib.sha256(gray_from_image(im, ch).tobytes()).hexdigest() for ch in GRAY_CHANNELS}
    (DIR / "channels.json").write_text(json.dumps({ORANGE_PHOTO: digests}, indent=1) + "\n")
    print(f"{ORANGE_PHOTO}: {x1 - x0}x{y1 - y0}, {(DIR / ORANGE_PHOTO).stat().st_size // 1024} KiB")


def main() -> None:
    orange_photo()
    dot = asdict(load_chain_spec(chain_presets()["oring"]))
    snap = asdict(load_chain_spec(chain_presets()["snap"]))
    truth = {"name": "s-curve.json", "pin_points": s_curve_pins(load_chain_spec(chain_presets()["oring"])).tolist()}
    base = {"chain": dot, "focal_override_35mm": 26}
    cases = {
        "photo-edits.png": [
            base,
            {"chain": dot},                                          # estimated
            {"chain": dot, "saved_focal_35mm": 26, "side": "outside"},
            {**base, "side": [700, 200]},
            {**base, "edits": {"remove": [[372.6, 345.3]], "add": [[100.5, 100.5]]}},
            {**base, "truth": truth},
            {**base, "truth": {"name": "short", "pin_points": truth["pin_points"][:5]}},
            {**base, "truth": {"name": "wrong", "pin_points": [[x, 2 * y] for x, y in truth["pin_points"]]},
             "compare_truth": True},
            {**base, "truth": {"name": "wrong", "pin_points": [[x, 2 * y] for x, y in truth["pin_points"]]}},
            {**base, "truth": truth, "compare_truth": False},
            {"chain": {**dot, "n_pins": 11}, "focal_override_35mm": 26},
            {"chain": {k: v for k, v in dot.items() if k != "pitch_mm"}},
            {"chain": dot, "side": "up"},
            {"chain": dot, "focal_override_35mm": 600},
            {"chain": {**dot, "pitch_mm": 3.0}},
        ],
        "photo-clutter-1.png": [base, {"chain": dot}],
        "photo-snap-07.png": [{"chain": snap, "focal_override_35mm": 26}, {"chain": snap}],
        "photo-snap-09.png": [{"chain": snap, "focal_override_35mm": 26, "side": "outside"}],
        # In gray the orange rims are not found, in either Colours; through the blue
        # channel they are as dark as black: found with Colours dark (the session reads
        # the photo again for each channel), then gray once more.
        ORANGE_PHOTO: [{"chain": snap, "focal_override_35mm": 26},
                       {"chain": {**snap, "fiducial_shade": "dark"}, "focal_override_35mm": 26},
                       {"chain": {**snap, "gray_channel": "b"}, "focal_override_35mm": 26},
                       {"chain": {**snap, "fiducial_shade": "dark", "gray_channel": "b"}, "focal_override_35mm": 26},
                       {"chain": {**snap, "fiducial_shade": "dark", "gray_channel": "b"}, "focal_override_35mm": 26,
                        "side": "outside"},
                       {"chain": {**snap, "fiducial_shade": "dark", "gray_channel": "g"}, "focal_override_35mm": 26},
                       {"chain": {**snap, "fiducial_shade": "dark"}, "focal_override_35mm": 26},
                       {"chain": {**snap, "gray_channel": "x"}}],
    }
    out = []
    for photo, options in cases.items():
        data = (DIR / photo).read_bytes()
        s = PhotoSession.open(data, photo)
        for opts in options:
            doc = s.measure(opts)
            out.append({"photo": photo, "options": opts, "document": doc,
                        "outputs": s.outputs() if doc["ok"] else None})
            print(f"{photo:22s} ok={doc['ok']!s:5s} {doc['error'] or (doc['result']['warnings'] + doc['result']['notes'])}")
    with gzip.GzipFile(DIR / "session.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(out).encode("utf-8"))
    print(f"{len(out)} documents -> {DIR / 'session.json.gz'} ({(DIR / 'session.json.gz').stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
