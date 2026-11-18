"""Python's DXF (splinewire/dxf.py) for a few curves, for the TypeScript
writer's test (web/src/pipeline/dxf.test.ts): the same text, byte for byte.

    uv run python web/scripts/golden_dxf.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from splinewire.dxf import dxf_text

GOLDEN = Path(__file__).resolve().parents[2] / "web" / "src" / "pipeline" / "golden" / "dxf.json.gz"


def main() -> None:
    rng = np.random.default_rng(11)
    curves = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[0.0, 0.0], [10.0, 2.0], [20.0, 0.0]]),
        np.array([[i * 10.0, (i % 3) * 2.0] for i in range(6)]),
        np.cumsum(rng.normal(0, 5, (13, 2)), axis=0),
        np.cumsum(rng.normal(0, 5, (30, 2)), axis=0),
        np.array([[-1e-12, 0.0], [10.0, -1e-9], [20.0, 0.0]]),          # no "-0.000000"
        np.array([[5.0, 5.0]]),                                         # one point: no spline
    ]
    golden = [{"points": c.tolist(), "dxf": dxf_text(c)} for c in curves]
    with gzip.GzipFile(GOLDEN, "wb", mtime=0) as f:
        f.write(json.dumps({"curves": golden}).encode("utf-8"))
    print(f"{len(golden)} files -> {GOLDEN}")


if __name__ == "__main__":
    main()
