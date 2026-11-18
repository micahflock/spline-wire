"""Hand edits measured by Python, for the TypeScript port's tests
(web/src/pipeline/edits.test.ts): the scenes of tests/test_edits.py and a
snap-fit render with a rim the detector misses, each with several edits,
and what pipeline.measure made of them (or its ValueError).

    uv run python web/scripts/golden_edits.py

Writes golden/edits.json.gz and the photos it needs, as 8-bit gray PNGs,
cropped to the chain (multiples of 16 px) to keep them small: the crops are
the scenes measured, with the full photo's focal length.
"""
from __future__ import annotations

import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.detect import detect_fiducials  # noqa: E402
from splinewire.edits import Edits  # noqa: E402
from splinewire.pipeline import measure  # noqa: E402
from splinewire.synthetic import LINK_GRAY, circle_wrap_pins, render_photo, render_snap_photo, s_curve_pins  # noqa: E402

DIR = REPO / "web" / "src" / "pipeline" / "golden"
SIZE = (2000, 1500)
FOCAL = focal_px_from_35mm(26, SIZE)


def fid(f) -> list:
    return [*f.center_px, *f.outer_axes_px, *f.inner_axes_px, f.angle_deg, f.residual_px, f.contrast,
            f.polarity, f.surround]


def crop_box(points: np.ndarray, size, margin: int = 110) -> tuple[int, int, int, int]:
    x0, y0 = (max(0, int(v) - margin) // 16 * 16 for v in points.min(axis=0))
    x1, y1 = (min(s, int(v) + margin) for v, s in zip(points.max(axis=0), size))
    return x0, y0, x0 + (x1 - x0) // 16 * 16, y0 + (y1 - y0) // 16 * 16


def run(img, spec, detected, edits: Edits, name: str) -> dict:
    out = {"name": name, "edits": edits.to_json()}
    try:
        m = measure(img, spec, FOCAL, edits=edits, detected=detected)
    except ValueError as err:
        out["error"] = str(err)
        return out
    out["result"] = {
        "fiducials": [fid(f) for f in m.fiducials], "indices": [int(i) for i in m.order.indices],
        "gaps": [int(g) for g in m.order.gaps], "manual": [[mp.fiducial, mp.edit, mp.snapped] for mp in m.manual],
        "pins_mm": m.pins_mm.tolist(), "contacts_mm": m.contacts_mm.tolist(),
        "warnings": m.warnings, "notes": m.notes,
    }
    return out


def clicks(cam, pins, which, jitter, seed):
    rng = np.random.default_rng(seed)
    pts = cam.project(pins[which]) + rng.uniform(-jitter, jitter, (len(which), 2))
    return tuple((float(x), float(y)) for x, y in pts)


def main() -> None:
    dot = load_chain_spec(chain_presets()["oring"])
    snap = load_chain_spec(chain_presets()["snap"])
    pins = s_curve_pins(dot)
    cam = look_at_plane(FOCAL, SIZE, 180.0, tilt_deg=30.0, tilt_direction_deg=35.0, roll_deg=-15.0,
                        target_mm=tuple(pins.mean(axis=0)))
    full = render_photo(pins, dot, cam)
    x0, y0, x1, y1 = crop_box(cam.project(pins), SIZE)
    img = np.ascontiguousarray(full[y0:y1, x0:x1])
    shift = np.array([x0, y0], float)

    class Cropped:                      # the camera, in the crop's pixels
        def __init__(self, inner):
            self.inner = inner

        def project(self, p):
            return self.inner.project(p) - shift

    cam = Cropped(cam)
    glare = img.copy()
    for k in (5, 6):
        cv2.circle(glare, tuple(int(v) for v in cam.project(pins[[k]])[0]), 34, LINK_GRAY, -1)
    det = detect_fiducials(img, dot)
    det_glare = detect_fiducials(glare, dot)
    nearest = lambda k: int(np.argmin([np.hypot(*(np.array(f.center_px) - cam.project(pins[[k]])[0])) for f in det]))
    without = [f for i, f in enumerate(det) if i not in {nearest(5), nearest(6)}]
    exact = tuple((float(x), float(y)) for x, y in cam.project(pins[[5, 6]]))
    on_pin3 = tuple(float(v) + 1.5 for v in det[nearest(3)].center_px)
    cases = [
        ("photo-edits.png", img, dot, without, [
            ("add-snapped", Edits(add_px=clicks(cam, pins, [5, 6], 3.0, 1))),
            ("add-far-off", Edits(add_px=clicks(cam, pins, [5, 6], 6.0, 7))),
            ("add-twice", Edits(add_px=clicks(cam, pins, [5, 5, 6], 2.0, 3))),
            ("add-nowhere", Edits(add_px=((40.0, 40.0),) + clicks(cam, pins, [5, 6], 2.0, 4))),
        ]),
        ("photo-edits.png", img, dot, det, [
            ("remove", Edits(remove_px=(on_pin3,))),
            ("remove-then-add", Edits(remove_px=(on_pin3,), add_px=(on_pin3,))),
            ("add-on-detection", Edits(add_px=clicks(cam, pins, [8], 2.0, 5))),
            ("remove-many", Edits(remove_px=tuple(det[nearest(k)].center_px for k in range(0, 12)))),
        ]),
        ("photo-edits.png", img, dot, [], [
            ("no-detections", Edits(add_px=clicks(cam, pins, list(range(13)), 2.0, 6))),
            ("one-click", Edits(add_px=clicks(cam, pins, [3], 2.0, 6))),
        ]),
        ("photo-edits-glare.png", glare, dot, det_glare, [
            ("glare-clicked", Edits(add_px=exact)),
            ("glare-clicked-off", Edits(add_px=clicks(cam, pins, [5, 6], 4.0, 2))),
        ]),
    ]
    # The snap-fit chain: a render where Python misses a rim (golden.py's
    # snap-09 scene, a pipe at a 40 degree tilt on white, drawn softer: blurred
    # 1.6 px instead of 0.7, it loses the rim of pin 11, which a click finds).
    spins = circle_wrap_pins(snap, 40.0)
    scam = look_at_plane(FOCAL, SIZE, 300.0, tilt_deg=40.0, tilt_direction_deg=37.0 * 9, roll_deg=10.0,
                         target_mm=tuple(spins.mean(axis=0)))
    sfull = render_snap_photo(spins, snap, scam, rng=np.random.default_rng(9), table_gray=250, blur_px=1.6)
    sx0, sy0, sx1, sy1 = crop_box(scam.project(spins), SIZE, 90)
    simg = np.ascontiguousarray(sfull[sy0:sy1, sx0:sx1])
    sdet = detect_fiducials(simg, snap)
    truth = scam.project(spins) - np.array([sx0, sy0], float)
    missing = [k for k, p in enumerate(truth)
               if min(np.hypot(*(np.array(f.center_px) - p)) for f in sdet) > 5.0]
    assert missing, "the snap-fit render no longer misses a rim: pick a harder one"
    adds = tuple((float(x) + 2.0, float(y) - 1.0) for x, y in truth[missing])
    cases.append(("photo-snap-09.png", simg, snap, sdet, [("snap-add-missed", Edits(add_px=adds))]))

    out, written = [], set()
    for photo, image, spec, detected, runs in cases:
        if photo not in written:
            cv2.imwrite(str(DIR / photo), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            written.add(photo)
        out.append({"photo": photo, "chain": asdict(spec), "focal_px": FOCAL, "size": [image.shape[1], image.shape[0]],
                    "detected": [fid(f) for f in detected],
                    "detected_is_full": detected is det or detected is det_glare or photo == "photo-snap-09.png",
                    "runs": [run(image, spec, detected, e, name) for name, e in runs]})
    with gzip.GzipFile(DIR / "edits.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(out).encode("utf-8"))
    for c in out:
        for r in c["runs"]:
            res = r.get("result")
            print(f"{c['photo']:22s} {r['name']:18s}",
                  r.get("error", "")[:70] if not res else f"{len(res['indices'])} pins, manual {res['manual']}, {res['warnings']}")
    for p in sorted(written):
        print(p, (DIR / p).stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
