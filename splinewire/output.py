"""Write measurement results: JSON for tools, DXF/CSV/SVG for CAD, JPEG preview."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np

from splinewire.contact import spline_samples
from splinewire.pipeline import Measurement

SCHEMA = "spline-wire/points@1"


def json_text(m: Measurement, extra: dict | None = None) -> str:
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
        "notes": m.notes,
    }
    if extra:
        doc.update(extra)
    return json.dumps(doc, indent=2) + "\n"


def csv_text(points_mm: np.ndarray) -> str:
    f = io.StringIO(newline="")
    writer = csv.writer(f)          # its default line ending is \r\n, as the files have always had
    writer.writerow(["x_mm", "y_mm"])
    writer.writerows(_round(points_mm))
    return f.getvalue()


def dxf_text(curve_points_mm: np.ndarray) -> str:
    """DXF in mm for CAD import (Fusion: Insert > Insert DXF).

    Layer CURVE holds one spline through the curve points, built the way
    AutoCAD builds a spline from fit points, so it passes exactly through
    them. Layer CURVE_POINTS holds the points themselves. $INSUNITS = mm so
    importers don't have to guess the scale.
    """
    import ezdxf

    doc = ezdxf.new("R2010", units=4)   # 4 = millimeters
    doc.layers.add("CURVE", color=7)
    doc.layers.add("CURVE_POINTS", color=1)
    msp = doc.modelspace()
    pts3 = [(float(x), float(y), 0.0) for x, y in curve_points_mm]
    msp.add_cad_spline_control_frame(pts3, dxfattribs={"layer": "CURVE"})
    for p in pts3:
        msp.add_point(p, dxfattribs={"layer": "CURVE_POINTS"})
    f = io.StringIO()
    doc.write(f)
    return f.getvalue()


def fusion_csv_text(curve_points_mm: np.ndarray) -> str:
    """CSV for Fusion's built-in ImportSplineCSV script: no header, x,y,z per
    line, in centimeters (Fusion's internal unit; the script doesn't convert)."""
    cm = np.asarray(curve_points_mm, dtype=float) / 10.0
    lines = [f"{x:.5f},{y:.5f},0" for x, y in cm]
    return "\n".join(lines) + "\n"


def points_tsv(points_mm: np.ndarray) -> str:
    """Curve points as a tab-separated table in mm, for the clipboard.

    Pastes into a spreadsheet as two columns, and the Spline Wire Fusion
    add-in's "Paste points" command reads it straight into a sketch.
    """
    rows = ["x_mm\ty_mm"] + [f"{x:.4f}\t{y:.4f}" for x, y in np.asarray(points_mm, dtype=float)]
    return "\n".join(rows) + "\n"


def svg_text(m: Measurement, margin_mm: float = 5.0) -> str:
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
    return "\n".join(parts) + "\n"


# The same documents as files. The *_text functions are the single source, so
# a file written here and a download made in the browser cannot differ.

def _write(path: Path, text: str) -> None:
    # newline="": the text already carries its line endings (CSV's are \r\n)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def write_json(path: Path, m: Measurement, extra: dict | None = None) -> None:
    _write(path, json_text(m, extra))


def write_csv(path: Path, points_mm: np.ndarray) -> None:
    _write(path, csv_text(points_mm))


def write_dxf(path: Path, curve_points_mm: np.ndarray) -> None:
    _write(path, dxf_text(curve_points_mm))


def write_fusion_csv(path: Path, curve_points_mm: np.ndarray) -> None:
    _write(path, fusion_csv_text(curve_points_mm))


def write_svg(path: Path, m: Measurement, margin_mm: float = 5.0) -> None:
    _write(path, svg_text(m, margin_mm))


def write_image(path: Path, bgr: np.ndarray) -> None:
    # imencode + write_bytes rather than cv2.imwrite, which cannot open
    # non-ASCII paths on Windows.
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", bgr)
    if not ok:
        raise OSError(f"could not encode image for {path}")
    Path(path).write_bytes(buf.tobytes())


def render_preview(image: np.ndarray, m: Measurement) -> np.ndarray:
    """BGR copy of the photo with detections drawn on: green = chain pins
    (numbered in chain order), blue = pins added by hand, red = rejected
    detections, yellow line = chain order (red across a missing pin)."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    scale = max(1, int(round(max(vis.shape[:2]) / 1500)))
    pts = m.pins_px
    for k in range(len(pts) - 1):
        color = (0, 200, 255) if k not in m.order.gaps else (0, 0, 255)
        cv2.line(vis, _ip(pts[k]), _ip(pts[k + 1]), color, scale, cv2.LINE_AA)
    by_hand = {mp.fiducial for mp in m.manual}
    for k, i in enumerate(m.order.indices):
        fid = m.fiducials[i]
        c = _ip(fid.center_px)
        cv2.circle(vis, c, int(fid.outer_axes_px[0] / 2) + 2 * scale,
                   (255, 120, 0) if i in by_hand else (0, 200, 0), scale, cv2.LINE_AA)
        cv2.putText(vis, str(k), (c[0] + int(fid.outer_axes_px[0] / 2) + 3 * scale, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 160, 0), scale, cv2.LINE_AA)
    for i in m.order.rejected:
        fid = m.fiducials[i]
        cv2.circle(vis, _ip(fid.center_px), int(fid.outer_axes_px[0] / 2) + 2 * scale,
                   (0, 0, 255), scale, cv2.LINE_AA)
    return vis


def _ip(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _round(pts: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]


def crop_to_chain(preview: np.ndarray, m: Measurement, margin: float = 0.25) -> np.ndarray:
    """The preview cropped to the detected fiducials plus a margin, so the chain
    fills the view even when it is small in the photo."""
    if not m.fiducials:
        return preview
    pts = np.array([r.center_px for r in m.fiducials])
    size = max(r.outer_axes_px[0] for r in m.fiducials)
    lo, hi = pts.min(axis=0) - size, pts.max(axis=0) + size
    pad = margin * (hi - lo).max()
    h, w = preview.shape[:2]
    x0, y0 = int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad))
    x1, y1 = int(min(w, hi[0] + pad)), int(min(h, hi[1] + pad))
    return preview[y0:y1, x0:x1]
