"""What the contact gauges should measure, before any are printed
(docs/real-world-tests.md, build step 2).

For each gauge, the snap-fit chain is placed as its sleeves would sit on the
gauge's edge (targets.chain_on_gauge) and scored two ways:

- model only: the exact pins, no photo. What is left is the contact model's
  own error: the contact point put on each pin's bisector, and the spline
  between the points.
- simulated photos: the chain rendered with its relief (synthetic.render_snap_photo)
  at 0, 20 and 35 degrees of tilt, two placements along the edge, measured
  with the EXIF focal length and with it estimated from the chain.

Run:  uv run python experiments/gauge_simulation.py [--json out.json]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splinewire.camera import EXIF_IFD, TAG_FOCAL_LENGTH_35MM, focal_px_from_35mm, look_at_plane  # noqa: E402
from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.scoring import score, score_photo  # noqa: E402
from splinewire.synthetic import render_snap_photo  # noqa: E402
from splinewire.targets import GAUGES, chain_on_gauge  # noqa: E402

SIZE = (4032, 3024)
FOCAL_35MM = 24.0
DISTANCE_MM = 230.0
TILTS = (0.0, 20.0, 35.0)
STARTS = (4.0, 14.0)            # where the first pin sits along the edge, mm


def cases():
    for name, g in GAUGES.items():
        sides = ((1 if g.concave else -1),) if g.concave is not None else (-1, 1)
        for side in sides:
            for start in STARTS:
                for k, tilt in enumerate(TILTS):
                    yield name, side, start, tilt, 60.0 * k + 25.0 + start


def run_case(case) -> dict:
    name, side, start, tilt, direction = case
    spec = load_chain_spec(chain_presets()["snap"])
    pins = chain_on_gauge(GAUGES[name], spec, side, start)
    cam = look_at_plane(focal_px_from_35mm(FOCAL_35MM, SIZE), SIZE, DISTANCE_MM, tilt_deg=tilt,
                        tilt_direction_deg=direction, roll_deg=8.0, target_mm=tuple(pins.mean(axis=0)))
    img = render_snap_photo(pins, spec, cam, rng=np.random.default_rng(int(start * 10 + tilt)))
    exif = Image.Exif()
    exif.get_ifd(EXIF_IFD)[TAG_FOCAL_LENGTH_35MM] = int(FOCAL_35MM)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=92, exif=exif)
    r = score_photo(buf.getvalue(), f"{name}.jpg", name, spec)
    return {"gauge": name, "side": side, "start": start, "tilt": tilt, **r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    spec = load_chain_spec(chain_presets()["snap"])

    model = {}
    for name, g in GAUGES.items():
        sides = ((1 if g.concave else -1),) if g.concave is not None else (-1, 1)
        worst = {"max": 0.0, "curve": 0.0}
        for side in sides:
            for start in np.arange(0.0, 13.0, 1.0):
                s = score(name, chain_on_gauge(g, spec, side, float(start)), spec)["inner"]
                worst["max"] = max(worst["max"], s["max_mm"])
                worst["curve"] = max(worst["curve"], s["curve_max_mm"])
        model[name] = worst

    with ProcessPoolExecutor() as pool:
        photos = list(pool.map(run_case, list(cases())))

    print(f"{'gauge':13} {'model pts':>9} {'model curve':>11} | photos: {'found':>6} "
          f"{'worst (EXIF)':>12} {'|dR| (EXIF)':>11} {'worst (est.)':>12} {'|dR| (est.)':>11}")
    rows = []
    for name in GAUGES:
        ps = [p for p in photos if p["gauge"] == name]
        ok = [p for p in ps if "error" not in p and p["pins_found"] == spec.n_pins]

        def worst(key, field):
            vals = [p["focal"][key]["score"]["inner"][field] for p in ok
                    if key in p["focal"] and "score" in p["focal"][key] and field in p["focal"][key]["score"]["inner"]]
            return max((abs(v) for v in vals), default=float("nan"))

        row = {"gauge": name, "model_max_mm": model[name]["max"], "model_curve_mm": model[name]["curve"],
               "photos": len(ps), "whole": len(ok),
               "exif_max_mm": worst("exif", "max_mm"), "exif_radius_mm": worst("exif", "radius_error_mm"),
               "est_max_mm": worst("estimated", "max_mm"), "est_radius_mm": worst("estimated", "radius_error_mm")}
        rows.append(row)
        print(f"{name:13} {row['model_max_mm']:9.3f} {row['model_curve_mm']:11.3f} | "
              f"{row['whole']:>3}/{row['photos']:<2} {row['exif_max_mm']:12.3f} {row['exif_radius_mm']:11.3f} "
              f"{row['est_max_mm']:12.3f} {row['est_radius_mm']:11.3f}")
    for p in photos:
        if "error" in p or p["pins_found"] != spec.n_pins:
            print(f"  not whole: {p['gauge']} side {p['side']:+d} start {p['start']} tilt {p['tilt']}: "
                  f"{p.get('error') or str(p['pins_found']) + ' pins; ' + '; '.join(p['warnings'])}")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "photos": photos}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
