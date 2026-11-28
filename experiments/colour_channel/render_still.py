"""Orange rims on off-white links, read through each colour channel: the still.

Work queue item 13 in docs/web-build.md: detection reads one channel of the
photo (ChainSpec gray_channel: luma, r, g or b) instead of always the luma.
This renders the snap-fit chain (data/chain-snap.yaml, with its relief:
synthetic.render_snap_photo) in colour, its rims orange and its links
off-white, on six tables, as a phone's camera would save it, and measures
each photo through every channel, with the chain's Colours light and dark.
It also writes the live view's frames for live.ts.

    uv run python experiments/colour_channel/render_still.py OUT_DIR [dim]   (~3 min)
    -> a table on stdout; OUT_DIR/still.json; OUT_DIR/live/ (frames, meta.json)

With `dim`, a dim room: every channel's noise three times as large (sigma
12, 9 and 18), the gray renders' "dim" condition for blue's sake, as blue is
the noisiest channel.

The scene:
- Colours (sRGB): orange PLA (240, 122, 35), gray 147 by Pillow's luma, and
  off-white PLA (235, 230, 214), gray 230: the friend's screenshot's ~148
  and ~230 (docs/future-work.md, Fiducial colours picked from a photo).
- Tables: the four neutral ones of that earlier model and of blobs.test.ts
  (gray 60, 120, 200 and white paper 235), then two coloured ones: a wooden
  table (150, 100, 60) and a green cutting mat (40, 90, 60).
- Camera: 24 mm, 12 degrees off straight-on; the still a 3024 x 4032 photo at
  span 0.40, the live view's frames a 1440 x 1920 stream (the phone's
  default) at span 0.30, where the earlier model found 0 of 13 on a dark
  table. Span: the pins' extent plus a link's width over the short side.
- What a phone does to colour, per channel: its own noise, larger in blue
  and red than in green (white balance under indoor light multiplies blue
  by about 2 and red by about 1.4: sigma 4, 3 and 6 gray levels for R, G,
  B, against the gray renders' 3), and 4:2:0 chroma, half resolution across
  and down (JPEG quality 90 for the still; the video's chroma is 4:2:0 too,
  modelled by JPEG at 95). A channel is then a mix of full-resolution luma
  and half-resolution chroma: blue reads blurred.

Still: PhotoSession over each channel's gray (Pillow's convert("L") for
luma, getchannel for the others), the photo's true focal length, the truth's
pin positions to score against.
"""
from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.session import PhotoSession
from splinewire.synthetic import render_snap_photo, s_curve_pins

ORANGE = (240, 122, 35)
OFF_WHITE = (235, 230, 214)
TABLES = {
    "dark 60": (60, 60, 60),
    "mid-gray 120": (120, 120, 120),
    "light gray 200": (200, 200, 200),
    "white paper 235": (235, 235, 235),
    "wood": (150, 100, 60),
    "green mat": (40, 90, 60),
}
NOISE = (4.0, 3.0, 6.0)
CHANNELS = ("luma", "r", "g", "b")
SHADES = ("light", "dark")
FOCAL_35MM = 24.0


def render(spec, pins, size, span, table, seed):
    w, h = size
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    focal = focal_px_from_35mm(FOCAL_35MM, size)
    distance = focal * extent / (span * min(w, h))
    cam = look_at_plane(focal, size, distance, tilt_deg=12.0, tilt_direction_deg=30.0, roll_deg=70.0,
                        target_mm=tuple(pins.mean(axis=0)))
    rgb = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(seed), table_gray=table,
                            link_gray=OFF_WHITE, rim_gray=ORANGE, noise_gray=NOISE)
    return rgb, cam, focal


def jpeg(rgb: np.ndarray, quality: int) -> Image.Image:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, subsampling=2)     # 4:2:0
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def channel(im: Image.Image, ch: str) -> np.ndarray:
    return np.asarray(im.convert("L") if ch == "luma" else im.getchannel("RGB".index(ch.upper())))


def main() -> None:
    out = Path(sys.argv[1])
    global NOISE
    if "dim" in sys.argv[2:]:
        NOISE = tuple(3 * n for n in NOISE)
    (out / "live").mkdir(parents=True, exist_ok=True)
    spec = load_chain_spec(default_chain_path())
    pins = s_curve_pins(spec)
    rows, meta = [], {"chain": asdict(spec), "frames": []}
    t0 = time.time()
    for k, (tname, table) in enumerate(TABLES.items()):
        # The still.
        rgb, cam, focal = render(spec, pins, (3024, 4032), 0.40, table, 20 + k)
        im = jpeg(rgb, 90)
        truth_px = cam.project(pins)
        levels = {ch: {part: float(np.median(channel(im, ch)[mask])) for part, mask in parts(truth_px, cam, spec, im.size).items()}
                  for ch in CHANNELS}
        for ch in CHANNELS:
            gray = channel(im, ch)
            for shade in SHADES:
                s = PhotoSession(f"{tname}-{ch}", gray, focal, {"width": gray.shape[1], "height": gray.shape[0]})
                sp = replace(spec, fiducial_shade=shade)
                doc = s.measure({"chain": asdict(sp), "truth": {"name": "truth", "pin_points": pins.tolist()},
                                 "compare_truth": True})
                near = sum(1 for p in truth_px if any(np.hypot(d["x"] - p[0], d["y"] - p[1]) < 2.0
                                                      for d in doc["view"]["pins"] + doc["view"]["unused"]))
                r = doc["result"]
                rows.append({"table": tname, "channel": ch, "shade": shade, "ok": doc["ok"],
                             "pins": r["pins"] if r else 0, "rims_found": near,
                             "worst_mm": (r["truth"] or {}).get("max_error_mm") if r else None,
                             "error": None if doc["ok"] else doc["error"].split(".")[0][:70]})
        rows[-1]["levels"] = levels
        # The live view's frames: one channel each, as the camera view's grab
        # gives it (its own luma, toGray, is computed by live.ts from RGB).
        frgb, fcam, ffocal = render(spec, pins, (1440, 1920), 0.30, table, 40 + k)
        fim = np.asarray(jpeg(frgb, 95))
        name = tname.replace(" ", "_")
        fim.tofile(out / "live" / f"{name}.rgb")
        meta["frames"].append({"table": tname, "file": f"{name}.rgb", "size": [1440, 1920], "focal_px": ffocal,
                               "pins": fcam.project(pins).tolist()})
        print(f"[{time.time() - t0:5.0f} s] {tname}", file=sys.stderr)
    (out / "live" / "meta.json").write_text(json.dumps(meta))
    (out / "still.json").write_text(json.dumps(rows, indent=1))
    print(f"{'table':16} {'channel':7} {'Colours':7} {'pins':>5} {'rims':>5} {'worst mm':>9}  error")
    for r in rows:
        worst = f"{r['worst_mm']:.3f}" if r["worst_mm"] is not None else "-"
        print(f"{r['table']:16} {r['channel']:7} {r['shade']:7} {r['pins']:>5} {r['rims_found']:>5} {worst:>9}  {r['error'] or ''}")
    print("\nMedian levels in the photo (rim, link head inside the rim, table), per channel:")
    for r in rows:
        if "levels" in r:
            print(f"  {r['table']:16} " + "  ".join(f"{ch}: {v['rim']:.0f}/{v['hole']:.0f}/{v['table']:.0f}"
                                                   for ch, v in r["levels"].items()))


def parts(pins_px, cam, spec, size):
    """Masks of the rims, the heads inside them and the table near the chain,
    from the true pin centres and the rim's size in pixels (straight-on
    approximation, trimmed by a pixel each side)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    centre = np.asarray(pins_px).mean(axis=0)
    scale = np.linalg.norm(np.diff(pins_px, axis=0), axis=1).mean() / spec.pitch_mm      # px per mm
    ro, ri = spec.fiducial_mm / 2 * scale, spec.ring_inner_mm / 2 * scale
    d = np.min([np.hypot(xx - x, yy - y) for x, y in pins_px], axis=0)
    rim = (d > ri + 1) & (d < ro - 1)
    hole = d < ri - 2
    far = np.hypot(xx - centre[0], yy - centre[1])
    table = (d > (spec.half_width_mm + 25) * scale) & (far < 1.2 * far[rim].max())
    return {"rim": rim, "hole": hole, "table": table}


if __name__ == "__main__":
    main()
