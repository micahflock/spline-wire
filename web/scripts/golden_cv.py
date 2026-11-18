"""cv2's own results for the OpenCV operations web/src/pipeline/cv.ts
rebuilds, written to web/src/pipeline/golden/cv.json.gz for cv.test.ts.

Inputs: a 320x240 crop of a cv_benchmark render (terrazzo: a chain among
many dot-like chips, so plenty of components) and a 64x48 noise image.
Outputs: GaussianBlur at the pyramid's two sigmas, the half-size INTER_AREA
resize, the three masks of detect._masks, connectedComponentsWithStats of
the adaptive mask, the outer contour and cv2.fitEllipse of each component
detect._dot_candidates would trace, fitEllipseDirect on noisy ellipse
points, and remap at random points. Arrays are base64 of their raw bytes.

    uv run python web/scripts/golden_cv.py     # needs cv_benchmark's renders (out/cv-benchmark/)
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from splinewire.detect import _masks  # noqa: E402

OUT = REPO / "web" / "src" / "pipeline" / "golden" / "cv.json.gz"
RENDER = REPO / "out" / "cv-benchmark" / "dot" / "terrazzo-0.png"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def image_record(img: np.ndarray) -> dict:
    h, w = img.shape
    masks = _masks(img)
    rec = {
        "width": w, "height": h, "image": b64(img),
        "blur08": b64(cv2.GaussianBlur(img, (0, 0), 0.8)),
        "blur10": b64(cv2.GaussianBlur(img, (0, 0), 1.0)),
        "half": b64(cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)),
        "adaptive": b64(masks[0][0]), "bright": b64(masks[2][0]),
    }
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masks[0][0], connectivity=8)
    rec["labels"] = b64(labels.astype(np.int32))
    rec["stats"] = stats.astype(int).tolist()
    contours = []
    for k in range(1, n):
        x, y, cw, ch, _ = stats[k]
        if min(cw, ch) < 4 or len(contours) >= 80:
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        roi = (labels[y0:y + ch + 1, x0:x + cw + 1] == k).astype(np.uint8)
        cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(int(x0), int(y0)))
        pts = cs[0].reshape(-1, 2)
        rec_c = {"label": k, "points": pts.astype(int).ravel().tolist()}
        if len(pts) >= 5:
            (cx, cy), (a, b), ang = cv2.fitEllipse(cs[0])
            rec_c["ellipse"] = [cx, cy, a, b, ang]
        contours.append(rec_c)
    rec["contours"] = contours
    return rec


def main() -> None:
    rng = np.random.default_rng(5)
    full = cv2.imread(str(RENDER), cv2.IMREAD_GRAYSCALE)
    if full is None:
        raise SystemExit(f"{RENDER} is missing: run experiments/cv_benchmark.py once to render it")
    meta = json.loads(RENDER.with_suffix(".json").read_text(encoding="utf-8"))
    px = np.array(meta["pins_px"])
    cx, cy = (int(v) for v in px[len(px) // 2])
    crop = np.ascontiguousarray(full[cy - 120:cy + 120, cx - 160:cx + 160])
    noise = rng.integers(0, 256, (48, 64)).astype(np.uint8)
    odd = np.ascontiguousarray(full[cy - 121:cy + 120, cx - 160:cx + 161])     # 321x241: OpenCV's general area path
    images = [image_record(crop), image_record(noise), image_record(odd), image_record(noise[:47, :63].copy())]

    direct = []
    for _ in range(200):
        n = int(rng.integers(12, 160))
        a, b = rng.uniform(4, 40), rng.uniform(4, 40)
        t = rng.uniform(0, 2 * np.pi, n)
        ang = rng.uniform(0, np.pi)
        x = a * np.cos(t) * np.cos(ang) - b * np.sin(t) * np.sin(ang) + rng.uniform(20, 3000)
        y = a * np.cos(t) * np.sin(ang) + b * np.sin(t) * np.cos(ang) + rng.uniform(20, 3000)
        pts = (np.c_[x, y] + rng.normal(0, rng.uniform(0.01, 0.5), (n, 2))).astype(np.float32)
        (ex, ey), (ea, eb), eang = cv2.fitEllipseDirect(pts)
        direct.append({"points": b64(pts), "ellipse": [ex, ey, ea, eb, eang]})

    g = crop.astype(np.float32)
    mx = rng.uniform(-3, crop.shape[1] + 3, 4000).astype(np.float32)
    my = rng.uniform(-3, crop.shape[0] + 3, 4000).astype(np.float32)
    remap = cv2.remap(g, mx[None], my[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {"opencv": cv2.__version__, "images": images, "direct": direct,
            "remap": {"x": b64(mx), "y": b64(my), "value": b64(remap.astype(np.float32))}}
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(json.dumps(data, separators=(",", ":")).encode("utf-8"))
    print(f"{OUT} ({OUT.stat().st_size // 1024} KiB), {sum(len(i['contours']) for i in images)} contours")


if __name__ == "__main__":
    main()
