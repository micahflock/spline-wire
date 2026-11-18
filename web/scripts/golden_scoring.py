"""Scores by Python's splinewire.scoring, for the TypeScript port's tests
(web/src/pipeline/scoring.test.ts), and the targets the test page offers.

    uv run python web/scripts/golden_scoring.py

Writes:
- web/src/pipeline/targets.json: every target (targets.targets_json), which
  the test page (spikes/accuracy/) and the port read. tests/test_scoring.py
  checks it is up to date.
- web/src/pipeline/golden/scoring.json.gz: plaque pins for several link
  counts, and chains on every target with noise, moved and turned, some cut
  short or on the wrong side, each with Python's score; and Python's
  measurement and score of the photo below (with its EXIF focal length, and
  with the focal length estimated).
- web/src/pipeline/golden/photo-plaque-s.png: a 1600x1200 gray render of the
  S plaque at a 25 degree tilt, for the test page's logic (bench.test.ts).
- web/spikes/accuracy/test-photo.jpg: the same render as a JPEG with its focal
  length in EXIF, as a phone saves it, for the page's end-to-end test.
"""
from __future__ import annotations

import gzip
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from splinewire.camera import focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.pipeline import measure  # noqa: E402
from splinewire.synthetic import render_snap_photo, save_photo  # noqa: E402
from splinewire.scoring import score  # noqa: E402
from splinewire.targets import GAUGES, PLAQUE_POSES, chain_on_gauge, plaque_pins, targets_json  # noqa: E402

TARGETS = REPO / "web" / "src" / "pipeline" / "targets.json"
OUT = REPO / "web" / "src" / "pipeline" / "golden" / "scoring.json.gz"
PHOTO = REPO / "web" / "src" / "pipeline" / "golden" / "photo-plaque-s.png"
PHOTO_SIZE, PHOTO_FOCAL_35MM = (1600, 1200), 26.0


def moved(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = pts @ rot.T + rng.uniform(-80, 80, 2)
    return out[::-1] if rng.random() < 0.5 else out


def main() -> None:
    spec = load_chain_spec(chain_presets()["snap"])
    TARGETS.write_text(json.dumps(targets_json(spec), indent=1) + "\n", encoding="utf-8")
    rng = np.random.default_rng(7)
    plaques = [{"pose": p, "n": n, "pins": plaque_pins(p, spec, n).tolist()}
               for p in PLAQUE_POSES for n in (6, 9, 13, 20)]
    cases = []
    for name, g in GAUGES.items():
        side = (1 if g.concave else -1) if g.concave is not None else -1
        for k, noise in enumerate((0.0, 0.05, 0.2)):
            pins = chain_on_gauge(g, spec, side, start_mm=3.0 + 4 * k)
            pins = moved(pins + rng.normal(0, noise, pins.shape), rng)
            sign = None if g.concave is None else (None if k == 2 else int(rng.choice([1, -1])))
            gaps = [5] if k == 1 else []
            cases.append({"target": name, "pins": pins.tolist(), "sign": sign, "gaps": gaps,
                          "spec": asdict(spec), "score": score(name, pins, spec, sign=sign, gaps=gaps)})
    for pose in PLAQUE_POSES:
        truth = plaque_pins(pose, spec)
        for cut in (0, 2):
            pins = moved(truth[cut:] + rng.normal(0, 0.05, truth[cut:].shape), rng)
            cases.append({"target": pose, "pins": pins.tolist(), "sign": None, "gaps": [],
                          "spec": asdict(spec), "score": score(pose, pins, spec)})
    dark = replace(spec, n_pins=9)
    pins = moved(plaque_pins("plaque-s", dark), rng)
    cases.append({"target": "plaque-s", "pins": pins.tolist(), "sign": None, "gaps": [],
                  "spec": asdict(dark), "score": score("plaque-s", pins, dark)})
    pins = plaque_pins("plaque-s", spec)
    focal = focal_px_from_35mm(PHOTO_FOCAL_35MM, PHOTO_SIZE)
    cam = look_at_plane(focal, PHOTO_SIZE, 260.0, tilt_deg=25.0, tilt_direction_deg=40.0, roll_deg=12.0,
                        target_mm=tuple(pins.mean(axis=0)))
    img = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(3), noise_gray=0.0)   # noise-free: a small PNG
    cv2.imwrite(str(PHOTO), img)
    save_photo(REPO / "web" / "spikes" / "accuracy" / "test-photo.jpg", img, PHOTO_FOCAL_35MM, quality=90)
    gray = cv2.imread(str(PHOTO), cv2.IMREAD_GRAYSCALE)
    photo = {"focal_35mm": PHOTO_FOCAL_35MM, "runs": {}}
    for key, f in (("exif", focal), ("estimated", None)):
        m = measure(gray, spec, f)
        photo["runs"][key] = {"pins_found": len(m.order.indices), "tilt_deg": m.rectification.tilt_deg,
                              "score": score("plaque-s", m.pins_mm, spec)}
    OUT.write_bytes(gzip.compress(json.dumps({"plaques": plaques, "cases": cases, "photo": photo}).encode(), 9, mtime=0))
    print(f"wrote {TARGETS.relative_to(REPO)} and {OUT.relative_to(REPO)} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
