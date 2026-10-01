"""Contact sheet of simulated photos from the benchmark cache (docs figure).

    uv run python experiments/cv_benchmark.py            # renders the cache first
    uv run python experiments/contact_sheet.py docs/img/environments.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "out" / "cv-benchmark"
SCENES = ["daylight-0", "shadow-1", "glare-0", "glare-matte-0", "black-table-1", "wood-0",
          "cutting-mat-0", "text-1", "clutter-0", "fabric-0", "terrazzo-0", "worst-0"]


def main(out: Path, tile=(360, 270), cols=4) -> None:
    tw, th = tile
    rows = (len(SCENES) + cols - 1) // cols
    sheet = np.full((rows * (th + 26), cols * tw), 255, np.uint8)
    for k, sid in enumerate(SCENES):
        meta = json.loads((CACHE / f"{sid}.json").read_text(encoding="utf-8"))
        img = cv2.imread(str(CACHE / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        p = np.array(meta["pins_px"])
        c = p.mean(axis=0)
        half = max(np.ptp(p[:, 0]) / 2 / tw * th, np.ptp(p[:, 1]) / 2) + 14 * meta["px_per_mm"]
        x0, y0 = int(c[0] - half * tw / th), int(c[1] - half)
        x1, y1 = int(c[0] + half * tw / th), int(c[1] + half)
        pad = cv2.copyMakeBorder(img, th * 4, th * 4, tw * 4, tw * 4, cv2.BORDER_REPLICATE)
        crop = pad[y0 + th * 4:y1 + th * 4, x0 + tw * 4:x1 + tw * 4]
        r, q = divmod(k, cols)
        y, x = r * (th + 26), q * tw
        sheet[y:y + th, x + 2:x + tw - 2] = cv2.resize(crop, (tw - 4, th), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, meta["environment"], (x + 6, y + th + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1,
                    cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
