// The markers over the photo, as SVG in photo pixels (the editor's SVG has
// the photo's own coordinates as its viewBox, viewport.ts). The look is the
// retired Windows app's desktop editor's (drawOverlay): pins
// green, pins added by hand blue (dashed where no dot was under the tap),
// stray dots red and dashed, removed pins as crossed ghosts, gaps dashed red,
// the edge that touched the object magenta.

import type { Marker, PhotoView, Point } from "../core/types";

export interface Extra {
  /** Where the person just tapped, until the new measurement arrives. */
  pending?: { x: number; y: number } | null;
}

const f = (n: number) => (Math.round(n * 100) / 100).toString();

export function overlaySvg(v: PhotoView, extra: Extra = {}): string {
  const out: string[] = [];
  const rr = (m: { r?: number }) => (m.r ?? v.r) * 1.3;
  const circle = (cls: string, m: { x: number; y: number; r?: number }, attrs = "") =>
    `<circle class="${cls}" cx="${f(m.x)}" cy="${f(m.y)}" r="${f(rr(m))}"${attrs}/>`;
  if (v.curve.length > 1) out.push(`<polyline class="ov-edge" points="${pts(v.curve)}"/>`);
  if (v.side_px) {
    out.push(`<circle class="ov-side" cx="${f(v.side_px[0])}" cy="${f(v.side_px[1])}" r="${f(v.r * 0.6)}"/>`);
  }
  v.pins.forEach((m, i) => {
    if (!i) return;
    const a = v.pins[i - 1]!;
    out.push(`<line class="${v.gaps.includes(i - 1) ? "ov-gap" : "ov-link"}" x1="${f(a.x)}" y1="${f(a.y)}" x2="${f(m.x)}" y2="${f(m.y)}"/>`);
  });
  for (const m of v.removed) {
    const r = rr({}), c = r * 0.55;
    out.push(`<g class="ov-removed" data-edit="${m.edit}">${circle("", m)}` +
      `<path d="M${f(m.x - c)} ${f(m.y - c)}L${f(m.x + c)} ${f(m.y + c)}M${f(m.x - c)} ${f(m.y + c)}L${f(m.x + c)} ${f(m.y - c)}"/></g>`);
  }
  for (const m of v.unused) out.push(circle(m.edit === undefined ? "ov-unused" : "ov-hand", m, data(m)));
  for (const m of v.pins) {
    const mine = m.edit !== undefined;
    out.push(circle(!mine ? "ov-pin" : m.snapped ? "ov-mine" : "ov-hand", m, data(m)),
      `<text class="ov-label${mine ? " mine" : ""}" x="${f(m.x + rr(m) * 0.8)}" y="${f(m.y - rr(m) * 0.8)}" ` +
      `font-size="${f(v.r * 1.1)}">${(m.k ?? 0) + 1}</text>`);
  }
  if (extra.pending) out.push(circle("ov-pending", { ...extra.pending, r: v.r * 1.1 }));
  return out.join("");
}

function data(m: Marker): string {
  return (m.k !== undefined ? ` data-k="${m.k}"` : "") + (m.edit !== undefined ? ` data-edit="${m.edit}"` : "");
}

function pts(points: Point[]): string {
  return points.map(([x, y]) => `${f(x)},${f(y)}`).join(" ");
}
