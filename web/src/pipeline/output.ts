// splinewire/output.py: the measurement's files as text, byte for byte as
// Python writes them (golden tests in session.test.ts). The DXF is in dxf.ts.

import { splineSamples } from "../desktop/spline";
import type { Measurement } from "./measure";
import { fixed, type Vec2 } from "./numeric";
import { pyDumps, pyFloat } from "./pyjson";

export const POINTS_SCHEMA = "spline-wire/points@1";

/** Python's round(x, digits) as a float. */
export function pyRound(x: number, digits: number): number {
  return Number(fixed(x, digits));
}

const round4 = (pts: readonly Vec2[]) => pts.map(([x, y]) => [pyFloat(pyRound(x, 4)), pyFloat(pyRound(y, 4))]);

/** output.json_text; `extra` is already in pyDumps form (floats marked). */
export function jsonText(m: Measurement, extra: Record<string, unknown> = {}): string {
  const r = m.rectification;
  const doc: Record<string, unknown> = {
    schema: POINTS_SCHEMA,
    units: "mm",
    curve_points: round4(m.contacts_mm),
    pin_points: round4(r.pins_mm),
    object_side: m.object_side,
    diagnostics: {
      pins_found: m.order.indices.length,
      missing_pins: m.order.gaps.length,
      rejected_detections: m.order.rejected.length,
      focal_px: pyFloat(pyRound(r.focal_px, 1)),
      focal_estimated: r.focal_estimated,
      tilt_deg: pyFloat(pyRound(r.tilt_deg, 2)),
      link_residual_rms_mm: pyFloat(pyRound(r.residual_rms_mm, 4)),
      link_residual_max_mm: pyFloat(pyRound(r.residual_max_mm, 4)),
    },
    warnings: m.warnings,
    notes: m.notes,
    ...extra,
  };
  return pyDumps(doc, 2) + "\n";
}

/** output.csv_text: csv.writer's \r\n line endings, floats as repr. */
export function csvText(points: readonly Vec2[]): string {
  const rows = ["x_mm,y_mm", ...round4(points).map(([x, y]) => `${pyDumps(x)},${pyDumps(y)}`)];
  return rows.map((r) => r + "\r\n").join("");
}

/** output.fusion_csv_text: x,y,z in centimetres, no header. */
export function fusionCsvText(points: readonly Vec2[]): string {
  return points.map(([x, y]) => `${fixed(x / 10, 5)},${fixed(y / 10, 5)},0`).join("\n") + "\n";
}

/** output.points_tsv: the clipboard table the Fusion add-in's Paste points reads. */
export function pointsTsv(points: readonly Vec2[]): string {
  return ["x_mm\ty_mm", ...points.map(([x, y]) => `${fixed(x, 4)}\t${fixed(y, 4)}`)].join("\n") + "\n";
}

/** output.svg_text: a 1:1 drawing in mm, the curve the only path in its group. */
export function svgText(m: Measurement, marginMm = 5.0): string {
  const curve = splineSamples(m.contacts_mm as [number, number][]) as Vec2[];
  const all = [...curve, ...m.rectification.pins_mm];
  const lo = [Math.min(...all.map((p) => p[0])) - marginMm, Math.min(...all.map((p) => p[1])) - marginMm];
  const hi = [Math.max(...all.map((p) => p[0])) + marginMm, Math.max(...all.map((p) => p[1])) + marginMm];
  const w = hi[0]! - lo[0]!, h = hi[1]! - lo[1]!;
  const xy = (p: Vec2): Vec2 => [p[0] - lo[0]!, hi[1]! - p[1]];
  const d = "M " + curve.map(xy).map(([x, y]) => `${fixed(x, 4)} ${fixed(y, 4)}`).join(" L ");
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${fixed(w, 3)}mm" height="${fixed(h, 3)}mm" `
      + `viewBox="0 0 ${fixed(w, 3)} ${fixed(h, 3)}">`,
    `<g id="curve"><path d="${d}" fill="none" stroke="#000" stroke-width="0.2"/></g>`,
    '<g id="pins" fill="none" stroke="#888" stroke-width="0.1">',
    ...m.rectification.pins_mm.map(xy).map(([x, y]) => `<circle cx="${fixed(x, 4)}" cy="${fixed(y, 4)}" r="0.6"/>`),
    "</g>",
    '<g id="curve-points" fill="#c00">',
    ...m.contacts_mm.map(xy).map(([x, y]) => `<circle cx="${fixed(x, 4)}" cy="${fixed(y, 4)}" r="0.3"/>`),
    "</g>",
    "</svg>",
  ];
  return parts.join("\n") + "\n";
}
