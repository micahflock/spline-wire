// The phone page's drawings (for UI review/phone/BRIEF.md): the chain strip
// under the header (one pin per pin of the chain, rows wrapping serpentine),
// the history's tiles (the chain and the curve in mm, no photo) and the working
// curve card's crop of the photo (the chain's box, at the card's aspect).
// Pure: SVG strings and numbers, so they are tested without a page.

import type { Point } from "../core/types";
import { chainArt, reachMm, type ChainLook } from "../drawing/chain-art";

const f = (n: number) => (Math.round(n * 10) / 10).toString();

/** Where the strip's pins go: rows of `per` pins `pitch` apart, every other row
 * running back right to left, so the chain reads as one line folded. */
export function stripLayout(n: number, width: number, pitch = 22, row = 20, margin = 6): { pins: Point[]; per: number; height: number } {
  const per = Math.max(2, Math.floor((width - 2 * margin) / pitch) + 1);
  const pins: Point[] = [];
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / per), k = i % per;
    pins.push([margin + (r % 2 ? per - 1 - k : k) * pitch, margin + 1 + r * row]);
  }
  const rows = Math.max(1, Math.ceil(n / per));
  return { pins, per, height: 2 * (margin + 1) + (rows - 1) * row };
}

/** The strip's links, px apart; each chain is drawn to its own proportions at that length. */
export const STRIP_PITCH = 22;

/** The chain strip: the chosen chain (drawing/chain-art.ts), one pin per pin of the chain,
 * rows wrapping serpentine. `grew`: the pin just added, drawn popping in. With no chain
 * yet (its presets still loading), an empty strip of the default chain's height. */
export function stripSvg(look: ChainLook | null, n: number, width: number, grew = -1): string {
  if (!look) return `<svg class="strip" viewBox="0 0 ${f(width)} 20" height="20" aria-hidden="true"></svg>`;
  const s = STRIP_PITCH / look.pitch_mm, reach = reachMm(look) * s;
  const margin = Math.ceil(reach) + 1, row = Math.max(STRIP_PITCH, Math.ceil(2 * reach + 4));
  const { pins, height } = stripLayout(n, width, STRIP_PITCH, row, margin);
  return `<svg class="strip" viewBox="0 0 ${f(width)} ${height}" height="${height}" role="img" aria-label="A chain of ${n} pins">`
    + chainArt(look, pins, s, { grew }) + "</svg>";
}

export type Box = [x: number, y: number, w: number, h: number];

export function bbox(points: readonly Point[]): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of points) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  return points.length ? [x0, y0, x1 - x0, y1 - y0] : [0, 0, 1, 1];
}

/** `b` grown by `pad` on every side, then widened or heightened about its centre to `aspect` (w / h). */
export function fitBox(b: Box, aspect: number, pad: number): Box {
  let [x, y, w, h] = [b[0] - pad, b[1] - pad, Math.max(b[2] + 2 * pad, 1e-6), Math.max(b[3] + 2 * pad, 1e-6)];
  if (w / h < aspect) {
    const nw = h * aspect;
    x -= (nw - w) / 2;
    w = nw;
  } else {
    const nh = w / aspect;
    y -= (nh - h) / 2;
    h = nh;
  }
  return [x, y, w, h];
}

/** A crop kept inside a W × H photo at the same aspect: shrunk if larger, then slid in. */
export function clampBox([x, y, w, h]: Box, W: number, H: number): Box {
  const a = w / h;
  if (w > W) { w = W; h = w / a; }
  if (h > H) { h = H; w = h * a; }
  return [Math.min(Math.max(x, 0), W - w), Math.min(Math.max(y, 0), H - h), w, h];
}

/** The card's crop of the photo: the chain's pins with a margin of about two
 * pin diameters, at the card's aspect, inside the photo. */
export function cardCrop(pins: readonly Point[], r: number, photo: { width: number; height: number }, aspect: number): Box {
  const b = bbox(pins);
  const pad = Math.max(4 * r, 0.06 * Math.max(b[2], b[3]));
  return clampBox(fitBox(b, aspect, pad), photo.width, photo.height);
}

/** A smooth path through the points (Catmull-Rom as cubic Béziers), as the spline reads. */
export function smoothPath(pts: readonly Point[]): string {
  if (pts.length < 3) return pts.map(([x, y], i) => `${i ? "L" : "M"}${f(x)} ${f(y)}`).join("");
  let d = `M${f(pts[0]![0])} ${f(pts[0]![1])}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)]!, p1 = pts[i]!, p2 = pts[i + 1]!, p3 = pts[Math.min(pts.length - 1, i + 2)]!;
    d += `C${f(p1[0] + (p2[0] - p0[0]) / 6)} ${f(p1[1] + (p2[1] - p0[1]) / 6)} ${f(p2[0] - (p3[0] - p1[0]) / 6)} ${f(p2[1] - (p3[1] - p1[1]) / 6)} ${f(p2[0])} ${f(p2[1])}`;
  }
  return d;
}

const line = (pts: readonly Point[]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${f(x)} ${f(y)}`).join("");

/** A curve as a drawing, from its mm (y up, as in CAD; drawn y down): the chain
 * it was measured with, faded (drawing/chain-art.ts; with no `look`, a faint thick
 * line with its pins), the curve over it in the curve's colour. For the history's
 * tiles (aspect 1), the card of a curve whose photo was deleted, and the desktop
 * page's list. */
export function curveDrawing(c: { pins: readonly Point[]; spline: readonly Point[] }, aspect: number,
  opts: { pad?: number; big?: boolean; look?: ChainLook | null } = {}): string {
  const flip = (p: readonly Point[]) => p.map(([x, y]): Point => [x, -y]);
  const pins = flip(c.pins), spline = flip(c.spline);
  const b = fitBox(bbox([...pins, ...spline]), aspect, Math.max(opts.pad ?? 12, opts.look ? reachMm(opts.look) + 1 : 0));
  const chain = opts.look
    ? `<g class="chain-art">${chainArt(opts.look, pins, 1)}</g>`
    : `<path class="ch" d="${line(pins)}" stroke-width="9.5"/>`
      + pins.map(([x, y]) => `<circle class="pd" cx="${f(x)}" cy="${f(y)}" r="${opts.big ? 1.6 : 1.9}"/>`).join("");
  return `<svg viewBox="${b.map(f).join(" ")}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">`
    + chain + `<path class="cv${opts.big ? " big" : ""}" d="${line(spline)}" vector-effect="non-scaling-stroke"/></svg>`;
}

/** The working curve card's picture: the photo (a preview stretched over the
 * original's pixels, as the editor draws it), cropped to the chain, and the
 * contact edge in blue over a white halo. No pin markers. */
export function cardSvg(o: { href: string | null; photo: { width: number; height: number }; crop: Box; curve: readonly Point[] | null }): string {
  const { width: W, height: H } = o.photo;
  const img = o.href ? `<image href="${o.href}" x="-0.5" y="-0.5" width="${W}" height="${H}" preserveAspectRatio="none"/>` : "";
  const d = o.curve && o.curve.length > 1 ? smoothPath(o.curve) : "";
  const curve = d ? `<path class="ov-halo" d="${d}"/><path class="ov-curve" d="${d}"/>` : "";
  return `<svg class="card-svg" viewBox="${o.crop.map(f).join(" ")}" preserveAspectRatio="xMidYMid slice" role="img" aria-label="The photo of the chain with the measured curve">${img}${curve}</svg>`;
}
