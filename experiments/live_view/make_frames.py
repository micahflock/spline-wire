import json, sys
from pathlib import Path
import numpy as np, cv2
from dataclasses import asdict
from splinewire.camera import focal_px_from_35mm, look_at_plane
from splinewire.chain import default_chain_path, load_chain_spec
from splinewire.synthetic import render_snap_photo, s_curve_pins
out = Path(sys.argv[1])
spec = load_chain_spec(default_chain_path())
pins = s_curve_pins(spec)
extent = float(np.linalg.norm(pins[:, None] - pins[None], axis=2).max()) + 2 * spec.half_width_mm
w, h = 3024, 4032
focal = focal_px_from_35mm(24.0, (w, h))
meta = {"chain": asdict(spec), "focal_px": focal, "size": [w, h], "frames": []}
for span in [0.55, 0.40, 0.30, 0.22, 0.16]:
    distance = focal * extent / (span * min(w, h))
    cam = look_at_plane(focal, (w, h), distance, tilt_deg=12.0, tilt_direction_deg=30.0, roll_deg=70.0, target_mm=tuple(pins.mean(axis=0)))
    img = np.clip(render_snap_photo(pins, spec, cam, rng=np.random.default_rng(3)), 0, 255).astype(np.uint8)
    np.save(out / f"full-{span}.npy", img) if span in (0.55, 0.22) else None
    img.tofile(out / f"full-{span}.gray")
    for L in [480, 640, 800, 1008, 1200, 1600, 2016]:
        s = 4032 / L
        ww, hh = round(w / s), round(h / s)
        small = cv2.resize(img, (ww, hh), interpolation=cv2.INTER_AREA)
        small.tofile(out / f"s{span}-L{L}.gray")
        meta["frames"].append({"span": span, "L": L, "w": ww, "h": hh, "file": f"s{span}-L{L}.gray"})
    meta.setdefault("pins", {})[str(span)] = np.asarray(cam.project(pins)).tolist()
(out / "meta.json").write_text(json.dumps(meta))
