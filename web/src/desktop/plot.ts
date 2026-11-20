// The curve drawn in millimetres, as the current app's curveSvg draws it: a
// millimetre graticule (10 mm major lines, 2 mm minor; 50 and 10 mm on big
// curves), the chain's pins faintly, the spline through the points, the points,
// and a scale bar. One SVG unit is one millimetre in both directions, so the
// drawing keeps the curve's true proportions at any size.
//
// plotGeometry is the arithmetic, tested; plotSvg turns it into markup. The
// markup holds numbers, and the label escaped.

import type { Point } from "../relay/messages";
import { esc } from "./wording";

export interface PlotGeometry {
  /** The drawing's extent in mm, on whole major lines. y is up, as in the files. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  major: number;
  minor: number;
  /** Grid lines in mm; `major` marks the heavier ones. */
  xs: { at: number; major: boolean }[];
  ys: { at: number; major: boolean }[];
  /** A size for dots, in mm, so they look the same on a 30 mm and a 300 mm curve. */
  r: number;
  /** The scale bar: one major division long, in the bottom-right corner. */
  bar: { x1: number; x2: number; y: number; label: string };
}

const PAD_MM = 8;
/** Beyond this extent in mm the grid coarsens, else a big curve is all grid. */
const BIG_MM = 160;

export function plotGeometry(points: readonly Point[], pins: readonly Point[], spline: readonly Point[]): PlotGeometry {
  const all = [...spline, ...points, ...pins];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  let x0 = Math.min(...xs) - PAD_MM;
  let x1 = Math.max(...xs) + PAD_MM;
  let y0 = Math.min(...ys) - PAD_MM;
  let y1 = Math.max(...ys) + PAD_MM;
  const big = Math.max(x1 - x0, y1 - y0) > BIG_MM;
  const major = big ? 50 : 10;
  const minor = big ? 10 : 2;
  x0 = Math.floor(x0 / major) * major;
  y0 = Math.floor(y0 / major) * major;
  x1 = Math.ceil(x1 / major) * major;
  y1 = Math.ceil(y1 / major) * major;
  const lines = (from: number, to: number) => {
    const out: { at: number; major: boolean }[] = [];
    // Counted in whole steps, so 0.1 mm of float drift never drops the last line.
    const steps = Math.round((to - from) / minor);
    for (let i = 0; i <= steps; i++) {
      const at = from + i * minor;
      out.push({ at, major: Math.abs(at / major - Math.round(at / major)) < 1e-6 });
    }
    return out;
  };
  const r = Math.max(x1 - x0, y1 - y0) / 160;
  return {
    x0, y0, x1, y1, major, minor,
    xs: lines(x0, x1),
    ys: lines(y0, y1),
    r,
    bar: { x1: x1 - major - 3 * r, x2: x1 - 3 * r, y: y0 + 4 * r, label: `${major} mm` },
  };
}

/** What the drawing adds for sharp corners (corners.ts): the curve as pieces,
 * the corners, and the curve's own points as targets to click. */
export interface CornerLayer {
  /** One spline per piece, in place of `spline`. */
  pieces: readonly (readonly Point[])[];
  /** Where each corner is, and the curve's own point it was marked on. */
  corners: readonly { at: Point; source: number }[];
  /** The curve's own points: each is a target (data-act="corner", data-i). */
  targets: readonly Point[];
  /** Targets a corner replaced, drawn faint. */
  replaced: readonly number[];
}

/** The SVG for a curve. Colours come from the page's CSS variables. */
export function plotSvg(points: readonly Point[], pins: readonly Point[], spline: readonly Point[], label: string,
  layer?: CornerLayer): string {
  const g = plotGeometry(points, pins, layer ? [...spline, ...layer.pieces.flat(), ...layer.targets] : spline);
  const w = g.x1 - g.x0;
  const h = g.y1 - g.y0;
  const n = (v: number) => +v.toFixed(3);
  const path = (pts: readonly Point[]) => pts.map((p, i) => `${i ? "L" : "M"}${n(p[0])} ${n(p[1])}`).join(" ");
  const line = (x1: number, y1: number, x2: number, y2: number, major: boolean) =>
    `<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" class="${major ? "grat" : "grat-2"}"/>`;
  const grid = [
    ...g.xs.map((l) => line(l.at, g.y0, l.at, g.y1, l.major)),
    ...g.ys.map((l) => line(g.x0, l.at, g.x1, l.at, l.major)),
  ].join("");
  // y is flipped so that +y is up, as in the DXF and in Fusion; the label is
  // drawn outside the flip so its text is upright.
  return `<svg viewBox="${n(g.x0)} ${n(-g.y1)} ${n(w)} ${n(h)}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(label)}">
<g transform="scale(1,-1)">
<g class="grid">${grid}</g>
<path class="chain" d="${path(pins)}"/>
${pins.map((p) => `<circle class="pin" cx="${n(p[0])}" cy="${n(p[1])}" r="${n(g.r * 1.6)}"/>`).join("")}
${layer ? layer.pieces.map((pc) => `<path class="trace" d="${path(pc)}"/>`).join("") : `<path class="trace" d="${path(spline)}"/>`}
${layer ? layer.replaced.map((i) => layer.targets[i]!).map((p) => `<circle class="pt-was" cx="${n(p[0])}" cy="${n(p[1])}" r="${n(g.r * 1.1)}"/>`).join("") : ""}
${points.filter((p) => !layer?.corners.some((c) => c.at[0] === p[0] && c.at[1] === p[1]))
    .map((p) => `<circle class="pt" cx="${n(p[0])}" cy="${n(p[1])}" r="${n(g.r * 1.1)}"/>`).join("")}
${layer ? layer.corners.map((c) => `<rect class="corner" x="${n(c.at[0] - g.r * 1.6)}" y="${n(c.at[1] - g.r * 1.6)}" width="${n(g.r * 3.2)}" height="${n(g.r * 3.2)}" transform="rotate(45 ${n(c.at[0])} ${n(c.at[1])})"/>`).join("") : ""}
${layer ? layer.targets.map((p, i) => (i === 0 || i === layer.targets.length - 1 ? "" :
    `<circle class="hit" data-act="corner" data-i="${i}" cx="${n(p[0])}" cy="${n(p[1])}" r="${n(g.r * 3)}"><title>${layer.corners.some((c) => c.source === i) || layer.replaced.includes(i) ? "Sharp corner: click to make it smooth" : "Click to make a sharp corner here"}</title></circle>`)).join("") : ""}
${layer ? layer.corners.map((c) => `<circle class="hit" data-act="corner" data-i="${c.source}" cx="${n(c.at[0])}" cy="${n(c.at[1])}" r="${n(g.r * 3)}"><title>Sharp corner: click to make it smooth</title></circle>`).join("") : ""}
<line class="bar" x1="${n(g.bar.x1)}" y1="${n(g.bar.y)}" x2="${n(g.bar.x2)}" y2="${n(g.bar.y)}"/>
</g>
<text class="bar-label" x="${n((g.bar.x1 + g.bar.x2) / 2)}" y="${n(-(g.bar.y + 2 * g.r))}" text-anchor="middle" font-size="${n(g.r * 3.4)}">${g.bar.label}</text>
</svg>`;
}
