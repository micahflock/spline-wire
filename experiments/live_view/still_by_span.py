"""The still's accuracy by span, for the coach's minSpan (web/src/overlay/coach.ts).

The snap-fit chain (S-curve, 13 pins) rendered at 12 MP (3024x4032, 24 mm
equivalent, 12 degrees off straight-on) at several spans and poses, measured
by the full pipeline with the focal length known, and compared with the truth
(pipeline.compare_to_truth: worst pin after a rigid fit).

    uv run python experiments/live_view/still_by_span.py
"""
from __future__ import annotations

import numpy as np

from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.pipeline import compare_to_truth, measure
from splinewire.synthetic import render_snap_photo, s_curve_pins

SPANS = [0.40, 0.30, 0.26, 0.22, 0.18]
POSES = [(12.0, 30.0, 70.0), (12.0, 120.0, 10.0), (20.0, 200.0, 40.0), (5.0, 300.0, 100.0)]  # tilt, direction, roll


def main() -> None:
    spec = load_chain_spec(default_chain_path())
    pins = s_curve_pins(spec)
    extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
    w, h = 3024, 4032
    focal = focal_px_from_35mm(24.0, (w, h))
    print("span  px/mm  worst pin (mm) per pose                 median  max")
    for span in SPANS:
        distance = focal * extent / (span * min(w, h))
        worst = []
        for k, (tilt, direction, roll) in enumerate(POSES):
            cam = look_at_plane(focal, (w, h), distance, tilt_deg=tilt, tilt_direction_deg=direction, roll_deg=roll,
                                target_mm=tuple(pins.mean(axis=0)))
            img = np.clip(render_snap_photo(pins, spec, cam, rng=np.random.default_rng(10 + k)), 0, 255).astype(np.uint8)
            try:
                m = measure(img, spec, focal)
                worst.append(compare_to_truth(m.pins_mm, pins)["max_error_mm"])
            except ValueError as e:
                print(f"  {span} pose {k}: {e}")
                worst.append(float("nan"))
        print(f"{span:.2f}  {span * min(w, h) / extent:5.2f}  " + "  ".join(f"{e:.3f}" for e in worst)
              + f"   {np.nanmedian(worst):.3f}  {np.nanmax(worst):.3f}")


if __name__ == "__main__":
    main()
