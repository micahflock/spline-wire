// The markers over the photo, as SVG in photo pixels (the editor's SVG has
// the photo's own coordinates as its viewBox, viewport.ts). Each fiducial is
// drawn as the outline detection fitted to it, so the marker lies on the dot's
// edge: pins green, pins added by hand blue (dashed where no dot was under the
// tap), stray dots red and dashed, removed pins as crossed ghosts, gaps dashed
// red, the edge that touched the object magenta.

import type { Marker, PhotoView, Point } from "../core/types";

export interface Extra {
  /** Where the person just tapped, until the new measurement arrives. */
  pending?: { x: number; y: number } | null;
}

const f = (n: number) => (Math.round(n * 100) / 100).toString();

export function overlaySvg(v: PhotoView, extra: Extra = {}): string {
  const out: string[] = [];
  // The detected outline: an ellipse where detection gave one, else a circle of radius r.
  const outline = (cls: string, m: { x: number; y: number; r?: number; r2?: number; angle?: number }, attrs = "") => {
    const r = m.r ?? v.r;
    if (m.r2 === undefined || m.r2 === r) return `<circle class="${cls}" cx="${f(m.x)}" cy="${f(m.y)}" r="${f(r)}"${attrs}/>`;
    const turn = m.angle ? ` transform="rotate(${f(m.angle)} ${f(m.x)} ${f(m.y)})"` : "";
    return `<ellipse class="${cls}" cx="${f(m.x)}" cy="${f(m.y)}" rx="${f(r)}" ry="${f(m.r2)}"${turn}${attrs}/>`;
  };
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
    const c = v.r * 0.55;
    out.push(`<g class="ov-removed" data-edit="${m.edit}">${outline("", m)}` +
      `<path d="M${f(m.x - c)} ${f(m.y - c)}L${f(m.x + c)} ${f(m.y + c)}M${f(m.x - c)} ${f(m.y + c)}L${f(m.x + c)} ${f(m.y - c)}"/></g>`);
  }
  for (const m of v.unused) out.push(outline(m.edit === undefined ? "ov-unused" : "ov-hand", m, data(m)));
  for (const m of v.pins) {
    out.push(outline(m.edit === undefined ? "ov-pin" : m.snapped ? "ov-mine" : "ov-hand", m, data(m)));
  }
  if (extra.pending) out.push(outline("ov-pending", { ...extra.pending, r: v.r }));
  return out.join("");
}

function data(m: Marker): string {
  return (m.k !== undefined ? ` data-k="${m.k}"` : "") + (m.edit !== undefined ? ` data-edit="${m.edit}"` : "");
}

function pts(points: Point[]): string {
  return points.map(([x, y]) => `${f(x)},${f(y)}`).join(" ");
}
