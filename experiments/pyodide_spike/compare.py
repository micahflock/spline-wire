"""Compare the browser's measurement of a photo with the native one, pin by pin.

    uv run python experiments/pyodide_spike/compare.py NATIVE.json BROWSER_RESULTS.json [--session cold]

NATIVE.json is measure_dump.py's --json output for the same photo;
BROWSER_RESULTS.json is run.mjs's output. Prints the largest differences in
detected pin centres (px), all detected fiducials (px), pin positions (mm)
and curve points (mm).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def max_diff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def matched_fiducials(a, b) -> float:
    """Largest distance from each native fiducial to the nearest browser one."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return float(d.min(axis=1).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("browser")
    ap.add_argument("--session", default=None, help="cold, warm, ... (default: all)")
    args = ap.parse_args()
    nat = json.load(open(args.native))
    res = json.load(open(args.browser))
    for s in res["sessions"]:
        if args.session and s["label"] != args.session:
            continue
        for m in s.get("measurements", []):
            if m["photo"] != nat["photo"]:
                continue
            out = {
                "session": s["label"],
                "photo": m["photo"],
                "pins": (len(nat["pins_px"]), len(m["pins_px"])),
                "fiducials": (nat["n_fiducials"], m["n_fiducials"]),
                "pins_px_max_diff": max_diff(nat["pins_px"], m["pins_px"]),
                "fiducials_px_max_diff": matched_fiducials(nat["fiducials_px"], m["fiducials_px"]),
                "pins_mm_max_diff": max_diff(nat["pins_mm"], m["pins_mm"]),
                "contacts_mm_max_diff": max_diff(nat["contacts_mm"], m["contacts_mm"]),
                "tilt_deg": (nat["tilt_deg"], m["tilt_deg"]),
                "focal_px": (nat["focal_px"], m["focal_px"]),
                "truth_max_error_mm": ((nat["truth_comparison"] or {}).get("max_error_mm"),
                                       (m["truth_comparison"] or {}).get("max_error_mm")),
                "versions_native": {k: v for k, v in nat["versions"].items() if k != "platform"},
            }
            print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
