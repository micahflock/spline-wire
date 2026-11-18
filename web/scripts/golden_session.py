"""PhotoSession documents and output files from Python, for the TypeScript
session's tests (web/src/pipeline/session.test.ts): the committed test photos
(golden/photo-*.png, 8-bit gray, no EXIF) measured with a range of options,
including ones that fail.

    uv run python web/scripts/golden_session.py     # after golden_detect.py and golden_edits.py
"""
from __future__ import annotations

import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.session import PhotoSession  # noqa: E402
from splinewire.synthetic import s_curve_pins  # noqa: E402

DIR = REPO / "web" / "src" / "pipeline" / "golden"


def main() -> None:
    dot = asdict(load_chain_spec(chain_presets()["oring"]))
    snap = asdict(load_chain_spec(chain_presets()["snap"]))
    truth = {"name": "s-curve.json", "pin_points": s_curve_pins(load_chain_spec(chain_presets()["oring"])).tolist()}
    base = {"chain": dot, "focal_override_35mm": 26}
    cases = {
        "photo-edits.png": [
            base,
            {"chain": dot},                                          # estimated
            {"chain": dot, "saved_focal_35mm": 26, "side": "outside"},
            {**base, "side": [700, 200]},
            {**base, "edits": {"remove": [[372.6, 345.3]], "add": [[100.5, 100.5]]}},
            {**base, "truth": truth},
            {**base, "truth": {"name": "short", "pin_points": truth["pin_points"][:5]}},
            {**base, "truth": {"name": "wrong", "pin_points": [[x, 2 * y] for x, y in truth["pin_points"]]},
             "compare_truth": True},
            {**base, "truth": {"name": "wrong", "pin_points": [[x, 2 * y] for x, y in truth["pin_points"]]}},
            {**base, "truth": truth, "compare_truth": False},
            {"chain": {**dot, "n_pins": 11}, "focal_override_35mm": 26},
            {"chain": {k: v for k, v in dot.items() if k != "pitch_mm"}},
            {"chain": dot, "side": "up"},
            {"chain": dot, "focal_override_35mm": 600},
            {"chain": {**dot, "pitch_mm": 3.0}},
        ],
        "photo-clutter-1.png": [base, {"chain": dot}],
        "photo-snap-07.png": [{"chain": snap, "focal_override_35mm": 26}, {"chain": snap}],
        "photo-snap-09.png": [{"chain": snap, "focal_override_35mm": 26, "side": "outside"}],
    }
    out = []
    for photo, options in cases.items():
        data = (DIR / photo).read_bytes()
        s = PhotoSession.open(data, photo)
        for opts in options:
            doc = s.measure(opts)
            out.append({"photo": photo, "options": opts, "document": doc,
                        "outputs": s.outputs() if doc["ok"] else None})
            print(f"{photo:22s} ok={doc['ok']!s:5s} {doc['error'] or (doc['result']['warnings'] + doc['result']['notes'])}")
    with gzip.GzipFile(DIR / "session.json.gz", "wb", mtime=0) as f:
        f.write(json.dumps(out).encode("utf-8"))
    print(f"{len(out)} documents -> {DIR / 'session.json.gz'} ({(DIR / 'session.json.gz').stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
