"""Render the sample photos the screenshots are taken from.

    uv run python "for UI review/tools/make_sample_photos.py" OUT_DIR

All simulated (splinewire.scene), so the geometry is known; none is a real photo.
Real photos of the printed chain are only in docs/img/ (hero.jpg, real-chain-app.jpg).
The "no chain" photo is plain noise, used to show the app's failure state.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# (shape, simulated environment, output name, extra synth args). The photos are chosen to
# show different looks: tilted plaque on clutter, glare, wood table, concave cove, bare chain.
PHOTOS = [
    ("pipe", "steep", "pipe-steep.jpg"),
    ("s-curve", "glare", "s-curve-glare.jpg"),
    ("cove", "wood", "cove-wood.jpg"),
    ("pipe", "clutter", "pipe-clutter.jpg"),
    ("s-curve", "bare-chain", "s-curve-bare-chain.jpg"),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for shape, env, name in PHOTOS:
        tmp = out / f"_{shape}-{env}"
        subprocess.run(["uv", "run", "splinewire", "synth", "--shape", shape, "--env", env,
                        "--out", str(tmp), "--seed", "3"], check=True, capture_output=True)
        (tmp / f"{shape}-{env}.jpg").replace(out / name)
        for left in tmp.iterdir():
            left.unlink()
        tmp.rmdir()
        print("wrote", out / name)
    rng = np.random.default_rng(1)
    img = (120 + rng.normal(0, 6, (1200, 1600, 3))).clip(0, 255).astype("uint8")
    cv2.putText(img, "no chain here", (500, 620), cv2.FONT_HERSHEY_SIMPLEX, 2, (60, 60, 60), 4)
    cv2.imwrite(str(out / "kitchen-table-no-chain.jpg"), img)
    print("wrote", out / "kitchen-table-no-chain.jpg")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "sample-photos"))
