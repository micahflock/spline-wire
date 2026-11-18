"""Small test photos and Python's detections on them, for the TypeScript
detector's CI test (web/src/pipeline/detect.test.ts): crops of a few
cv_benchmark renders and snap-fit renders around their chains, cut to
multiples of 16 px, so the pyramid halves them evenly. The test over all
191 full renders runs where they are cached (out/), not in CI.

    uv run python web/scripts/golden_detect.py     # after web/scripts/golden.py
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

from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.detect import detect_fiducials  # noqa: E402

DIR = REPO / "web" / "src" / "pipeline" / "golden"
PHOTOS = [  # (source, crop margin around the chain in px)
    (REPO / "out" / "cv-benchmark" / "dot" / "clutter-1.png", 120),
    (REPO / "out" / "golden-snap" / "snap-07.png", 80),
]


def chain_box(path: Path, img: np.ndarray, spec) -> tuple[int, int, int, int]:
    found = detect_fiducials(img, spec)
    pts = np.array([f.center_px for f in found])
    return (*pts.min(axis=0), *pts.max(axis=0))


def main() -> None:
    out = []
    for path, margin in PHOTOS:
        spec = load_chain_spec(chain_presets()["snap" if "snap" in path.name else "oring"])
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = chain_box(path, img, spec)
        h, w = img.shape
        x0, y0 = max(0, int(x0) - margin) // 16 * 16, max(0, int(y0) - margin) // 16 * 16
        x1, y1 = min(w, int(x1) + margin), min(h, int(y1) + margin)
        x1, y1 = x0 + (x1 - x0) // 16 * 16, y0 + (y1 - y0) // 16 * 16
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        name = f"photo-{path.stem}.png"
        cv2.imwrite(str(DIR / name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        fids = detect_fiducials(crop, spec)
        out.append({"photo": name, "chain": asdict(spec), "fiducials": [
            {"center_px": list(f.center_px), "outer_axes_px": list(f.outer_axes_px),
             "inner_axes_px": list(f.inner_axes_px), "angle_deg": f.angle_deg, "residual_px": f.residual_px,
             "contrast": f.contrast, "polarity": f.polarity, "surround": f.surround} for f in fids]})
        print(f"{name}: {crop.shape[1]}x{crop.shape[0]}, {len(fids)} fiducials, "
              f"{(DIR / name).stat().st_size // 1024} KiB")
    with gzip.GzipFile(DIR / "detect.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(out).encode("utf-8"))


if __name__ == "__main__":
    main()
