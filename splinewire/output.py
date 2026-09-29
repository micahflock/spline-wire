"""Write measurement results: JSON for tools, SVG for CAD import, PNG preview."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from splinewire.contact import spline_samples
from splinewire.pipeline import Measurement

SCHEMA = "spline-wire/points@1"


def write_json(path: Path, m: Measurement, extra: dict | None = None) -> None:
    r = m.rectification
    doc = {
        "schema": SCHEMA,
        "units": "mm",
        # Points on the target curve, in order. Fit a spline through these in CAD.
        "curve_points": _round(m.contacts_mm),
        # Measured pin centers (the chain's own pin line).
        "pin_points": _round(m.pins_mm),
        "object_side": m.object_side,
        "diagnostics": {
            "pins_found": len(m.order.indices),
            "missing_pins": len(m.order.gaps),
            "rejected_detections": len(m.order.rejected),
            "focal_px": round(r.focal_px, 1),
            "focal_estimated": r.focal_estimated,
            "tilt_deg": round(r.tilt_deg, 2),
            "link_residual_rms_mm": round(r.residual_rms_mm, 4),
            "link_residual_max_mm": round(r.residual_max_mm, 4),
        },
        "warnings": m.warnings,
    }
    if extra:
        doc.update(extra)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, points_mm: np.ndarray) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm"])
        writer.writerows(_round(points_mm))


def write_svg(path: Path, m: Measurement, margin_mm: float = 5.0) -> None:
    """1:1 scale drawing in mm: the fitted curve plus measured points.

    The curve is the only path in the file's "curve" group, so it can be
    brought into a CAD sketch via SVG import.
    """
    curve = spline_samples(m.contacts_mm)
    everything = np.vstack([curve, m.pins_mm])
    lo = everything.min(axis=0) - margin_mm
    hi = everything.max(axis=0) + margin_mm
    w, h = hi - lo

    def xy(p: np.ndarray) -> tuple[float, float]:
        return float(p[0] - lo[0]), float(hi[1] - p[1])  # SVG y points down

    path_d = "M " + " L ".join(f"{x:.4f} {y:.4f}" for x, y in map(xy, curve))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
        f'viewBox="0 0 {w:.3f} {h:.3f}">',
        f'<g id="curve"><path d="{path_d}" fill="none" stroke="#000" stroke-width="0.2"/></g>',
        '<g id="pins" fill="none" stroke="#888" stroke-width="0.1">',
        *(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="0.6"/>' for x, y in map(xy, m.pins_mm)),
        "</g>",
        '<g id="curve-points" fill="#c00">',
        *(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="0.3"/>' for x, y in map(xy, m.contacts_mm)),
        "</g>",
        "</svg>",
    ]
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_preview(path: Path, image: np.ndarray, m: Measurement) -> None:
    """The photo with detections drawn on: green = chain pins (numbered in
    chain order), red = rejected detections, yellow line = chain order."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    scale = max(1, int(round(max(vis.shape[:2]) / 1500)))
    pts = m.pins_px
    for k in range(len(pts) - 1):
        color = (0, 200, 255) if k not in m.order.gaps else (0, 0, 255)
        cv2.line(vis, _ip(pts[k]), _ip(pts[k + 1]), color, scale, cv2.LINE_AA)
    for k, i in enumerate(m.order.indices):
        ring = m.rings[i]
        c = _ip(ring.center_px)
        cv2.circle(vis, c, int(ring.outer_axes_px[0] / 2) + 2 * scale, (0, 200, 0), scale, cv2.LINE_AA)
        cv2.putText(vis, str(k), (c[0] + int(ring.outer_axes_px[0] / 2) + 3 * scale, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 160, 0), scale, cv2.LINE_AA)
    for i in m.order.rejected:
        ring = m.rings[i]
        cv2.circle(vis, _ip(ring.center_px), int(ring.outer_axes_px[0] / 2) + 2 * scale,
                   (0, 0, 255), scale, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def _ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _round(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]
