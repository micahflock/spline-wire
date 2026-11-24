// splinewire/detect.py: find the fiducials, light dots (detect_dots) or the
// snap-fit chain's thin rims (detect_rims), with sub-pixel accuracy. Thick
// rings (detect_rings) are not ported yet: no chain preset uses them.
//
// The same two stages as Python, with the OpenCV operations from cv.ts:
//   1. Candidates: on a small pyramid, threshold against the local mean (and
//      a "well above" mask for dots), take connected components of a
//      plausible size, trace each outline and fit an ellipse.
//   2. Refinement on the full-size image: rays from the candidate's centre,
//      the edge on each ray where the profile crosses half way between inside
//      and the margin, a robust ellipse through the edge points, twice.
// Arithmetic follows NumPy's dtypes: the image and its samples are float32,
// geometry is float64, ellipses come back from OpenCV as float32.

import {
  adaptiveMeanThreshold, brightMask, centroids, connectedComponents, fitEllipse, fitEllipseDirect, gaussianBlur8,
  halveArea8, otsuThreshold, outerContour, sampleLinear, type Ellipse, type Gray, type GrayF,
} from "./cv";
import { PipelineError } from "./errors";
import type { Vec2 } from "./numeric";
import type { ChainSpec, Fiducial } from "./types";

const BLOCK = 25;             // local threshold window at each pyramid level, px
const MIN_LEVEL_SIDE = 360;   // smallest pyramid level, px
const MAX_SCALE = 8;          // coarsest level: 1/8 resolution
const STEP_PX = 0.25;         // sample spacing along each ray
const DOT_MARGIN: [number, number] = [1.15, 1.45];
const MAX_CANDIDATES = 1500;  // detect.py says why
export const THIN_RING = 0.6;

const f32 = Math.fround;

interface Candidate {
  cx: number;
  cy: number;
  outer: Ellipse;             // full resolution
  polarity: number;
}

/** The photo as the detectors want it, light fiducials on dark links
 * (detect.shaded): a chain with dark fiducials on light links is read inverted. */
export function shaded(gray: Gray, spec: ChainSpec): Gray {
  if (spec.fiducial_shade !== "dark") return gray;
  const data = new Uint8Array(gray.data.length);
  for (let i = 0; i < data.length; i++) data[i] = 255 - gray.data[i]!;
  return { width: gray.width, height: gray.height, data };
}

/** A search region's corners are whole multiples of this, and it is at least
 * REGION_MIN_SIDE across (detect._REGION_ALIGN, _REGION_MIN_SIDE). */
const REGION_ALIGN = 8;
const REGION_MIN_SIDE = 512;

export type Box = [number, number, number, number];

/** The fiducials a ChainSpec describes (detect_fiducials); with `region`
 * (x0, y0, x1, y1), searched only there, as the whole photo would be. */
export function detectFiducials(image: Gray, spec: ChainSpec, region: readonly number[] | null = null): Fiducial[] {
  image = shaded(image, spec);
  const box = region ? searchBox(image.width, image.height, region) : null;
  if (!box) return detectIn(image, spec, null);
  const [x0, y0, x1, y1] = box;
  const found = detectIn(crop(image, box), spec, Math.min(image.width, image.height));
  return found.map((f) => ({ ...f, center_px: [f.center_px[0] + x0, f.center_px[1] + y0] as Vec2 }));
}

/** detect.search_box: `region` grown to REGION_ALIGN-pixel corners and
 * REGION_MIN_SIDE, inside the photo; null when that is the whole photo. */
export function searchBox(width: number, height: number, region: readonly number[]): Box | null {
  if (region.length !== 4 || !region.every((v) => Number.isFinite(v))) return null;
  const [rx0, ry0, rx1, ry1] = region as [number, number, number, number];
  if (rx1 <= rx0 || ry1 <= ry0) return null;
  const span = (a: number, b: number, limit: number): [number, number] => {
    const grow = Math.max(0, (REGION_MIN_SIDE - (b - a)) / 2);
    a -= grow;
    b += grow;
    if (a < 0) {
      b -= a;
      a = 0;
    }
    if (b > limit) {
      a = Math.max(0, a - (b - limit));
      b = limit;
    }
    return [Math.max(0, Math.floor(a / REGION_ALIGN) * REGION_ALIGN), Math.min(limit, Math.ceil(b / REGION_ALIGN) * REGION_ALIGN)];
  };
  const [x0, x1] = span(rx0, rx1, width), [y0, y1] = span(ry0, ry1, height);
  if (x1 - x0 < 2 * REGION_ALIGN || y1 - y0 < 2 * REGION_ALIGN) return null;
  return x0 === 0 && y0 === 0 && x1 === width && y1 === height ? null : [x0, y0, x1, y1];
}

function crop(image: Gray, [x0, y0, x1, y1]: Box): Gray {
  const w = x1 - x0, h = y1 - y0, data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) data.set(image.data.subarray((y0 + y) * image.width + x0, (y0 + y) * image.width + x1), y * w);
  return { width: w, height: h, data };
}

function detectIn(image: Gray, spec: ChainSpec, pyramidSide: number | null): Fiducial[] {
  if (spec.fiducial === "dot") return detectDots(image, 8.0, pyramidSide);
  const ratio = spec.ring_inner_mm / spec.fiducial_mm;
  if (ratio >= THIN_RING) return detectRims(image, ratio, 8.0, pyramidSide);
  throw new PipelineError("thick ring fiducials are not supported in this build yet; use dots or the snap-fit chain");
}

/** (blurred level, scale) from full size down to about 1/8 (detect._pyramid);
 * `side`: the whole photo's shorter side when `gray` is a region of it. */
function* pyramid(gray: Gray, side: number | null = null): Generator<[Gray, number]> {
  let level = gray, scale = 1;
  for (;;) {
    yield [gaussianBlur8(level, scale === 1 ? 0.8 : 1.0), scale];
    const shorter = side === null ? Math.min(level.width, level.height) : Math.floor(side / scale);
    if (shorter < 2 * MIN_LEVEL_SIDE || scale >= MAX_SCALE) return;
    level = halveArea8(level);
    scale *= 2;
  }
}

/** Light dots on a dark link (detect.detect_dots). */
export function detectDots(gray: Gray, minDiameterPx = 8.0, pyramidSide: number | null = null): Fiducial[] {
  const candidates: Candidate[] = [];
  for (const [g, scale] of pyramid(gray, pyramidSide)) {
    const binary = adaptiveMeanThreshold(g, BLOCK, -4);
    for (const mask of [binary, brightMask(g, BLOCK)]) {
      candidates.push(...dotCandidates(mask, g.width, g.height, scale === 1 ? minDiameterPx : 10.0, scale, 1));
    }
  }
  const grayf = toFloat(gray);
  const dots: Fiducial[] = [];
  for (const c of dedupeCandidates(candidates).slice(0, MAX_CANDIDATES)) {
    const d = refineDot(grayf, c);
    if (d) dots.push(d);
  }
  return dedupe(dots);
}

/** Thin light rings found by their holes (detect.detect_rims). */
export function detectRims(gray: Gray, innerOuterRatio: number, minDiameterPx = 8.0, pyramidSide: number | null = null): Fiducial[] {
  const candidates: Candidate[] = [];
  for (const [g, scale] of pyramid(gray, pyramidSide)) {
    const binary = adaptiveMeanThreshold(g, BLOCK, -4);
    const below = invert8(binary);
    candidates.push(...dotCandidates(below, g.width, g.height, scale === 1 ? minDiameterPx : 10.0, scale, -1));
  }
  const grayf = toFloat(gray);
  const inverted = invertF(grayf);
  const rims: Fiducial[] = [];
  for (const c of dedupeCandidates(candidates).slice(0, MAX_CANDIDATES)) {
    const r = refineRim(grayf, inverted, c, innerOuterRatio);
    if (r) rims.push(r);
  }
  return dedupe(rims);
}

// ---------------------------------------------------------------------------
// Snapping a click to a fiducial (detect.snap_fiducial)

/** The fiducial a person pointed at, found and refined like any detection;
 * null if there is no clear one there. `like` is a neighbouring pin, which
 * says how big the one looked for is. `grayf` is the photo as float32, if
 * the caller keeps one (it is made otherwise). */
export function snapFiducial(gray: Gray, point: readonly [number, number], spec: ChainSpec, like: Fiducial,
  grayf: GrayF = toFloat(gray)): Fiducial | null {
  if (spec.fiducial_shade === "dark") {
    gray = shaded(gray, spec);
    grayf = invertF(grayf);
  }
  const ring = spec.fiducial === "ring" ? spec.ring_inner_mm / spec.fiducial_mm : 0;
  let found: Fiducial | null;
  if (ring >= THIN_RING) {
    // A thin rim: the dark hole under the click.
    const hole = blobUnder(gray, point, { ...like, outer_axes_px: like.inner_axes_px, polarity: -1 }, 0);
    const inverted = invertF(grayf);
    found = hole ? refineRim(grayf, inverted, hole, ring) : null;
  } else {
    if (spec.fiducial !== "dot") throw new PipelineError("thick ring fiducials are not supported in this build yet");
    const cand = blobUnder(gray, point, like, ring);
    if (cand === null) return null;
    found = refineDot(grayf, cand);
  }
  if (found === null) return null;
  const [major, minor] = like.outer_axes_px;
  if (Math.hypot(found.center_px[0] - point[0], found.center_px[1] - point[1]) > 0.5 * minor) return null;
  if (!(0.6 < found.outer_axes_px[0] / major && found.outer_axes_px[0] / major < 1.6)) return null;
  return found;
}

/** The outline of the blob under (or nearest) a point, as a start for
 * refinement (detect._blob_under). */
function blobUnder(gray: Gray, point: readonly [number, number], like: Fiducial, ringRatio: number): Candidate | null {
  const [major, minor] = like.outer_axes_px;
  const half = roundHalfEven(1.7 * major);
  const { width: w, height: h } = gray;
  const x0 = Math.max(0, Math.trunc(point[0]) - half), y0 = Math.max(0, Math.trunc(point[1]) - half);
  const x1 = Math.min(w, Math.trunc(point[0]) + half + 1), y1 = Math.min(h, Math.trunc(point[1]) + half + 1);
  if (Math.min(x1 - x0, y1 - y0) < 8) return null;
  const cw = x1 - x0, ch = y1 - y0;
  const crop = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) crop.set(gray.data.subarray((y0 + y) * w + x0, (y0 + y) * w + x1), y * cw);
  const blurred = gaussianBlur8({ width: cw, height: ch, data: crop }, 1.0).data;
  const t = otsuThreshold(blurred);
  const mask = blurred.map((v) => ((v > t) === like.polarity > 0 ? 255 : 0));
  const c = connectedComponents(mask, cw, ch);
  const px = point[0] - x0, py = point[1] - y0;
  let label = c.labels[Math.min(Math.max(Math.trunc(py), 0), ch - 1) * cw + Math.min(Math.max(Math.trunc(px), 0), cw - 1)]!;
  if (label === 0) {
    // The click fell in a ring's hole or just off the blob: the nearest one.
    const cents = centroids(c, cw);
    let best = -1, bestD = Infinity;
    for (let l = 1; l < c.count; l++) {
      const d = Math.hypot(cents[l]![0] - px, cents[l]![1] - py);
      if (d < bestD) {
        bestD = d;
        best = l;
      }
    }
    if (best < 0 || bestD > 0.7 * minor) return null;
    label = best;
  }
  const expected = (Math.PI / 4) * major * minor * (1.0 - ringRatio ** 2);
  const area = c.area[label]!;
  if (!(0.35 * expected < area && area < 2.5 * expected)) return null;
  const contour = outerContour(c, cw, ch, label);
  if (contour.length / 2 < 12) return null;
  const e = fitEllipse(contour);
  const cx = e.cx + x0, cy = e.cy + y0;
  return { cx, cy, outer: { ...e, cx, cy }, polarity: like.polarity };
}

function toFloat(gray: Gray): GrayF {
  const out = new Float32Array(gray.data.length);
  out.set(gray.data);
  return { width: gray.width, height: gray.height, data: out };
}

/** 255 - v for a mask or a photo (both exact). */
function invert8(v: Uint8Array): Uint8Array {
  const out = new Uint8Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = 255 - v[i]!;
  return out;
}

function invertF(img: GrayF): GrayF {
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < out.length; i++) out[i] = 255 - img.data[i]!;
  return { width: img.width, height: img.height, data: out };
}

/** Blobs that could be a dot: components of a plausible size and shape whose
 * outline follows an ellipse (detect._dot_candidates). */
function dotCandidates(mask: Uint8Array, w: number, h: number, minDiameterPx: number, scale: number, polarity: number): Candidate[] {
  const c = connectedComponents(mask, w, h);
  const off = (scale - 1) / 2;
  const found: Candidate[] = [];
  for (let k = 1; k < c.count; k++) {
    const lo = Math.min(c.w[k]!, c.h[k]!), hi = Math.max(c.w[k]!, c.h[k]!);
    if (!(lo >= minDiameterPx && hi <= Math.max(w, h) / 4 && hi <= 4 * lo)) continue;
    const contour = outerContour(c, w, h, k);
    if (contour.length / 2 < 12) continue;
    const e = fitEllipse(contour);
    if (Math.min(e.width, e.height) < minDiameterPx || !ellipseLike(contour, e)) continue;
    const cx = e.cx * scale + off, cy = e.cy * scale + off;
    found.push({ cx, cy, outer: { cx, cy, width: e.width * scale, height: e.height * scale, angle: e.angle }, polarity });
  }
  return found;
}

/** The outline follows the fitted ellipse, but for a few stray points (detect._ellipse_like). */
function ellipseLike(contour: Int32Array, e: Ellipse): boolean {
  const a = e.width, b = e.height;
  if (Math.min(a, b) <= 0) return false;
  const t = (e.angle * Math.PI) / 180;
  const ct = Math.cos(t), st = Math.sin(t);
  const n = contour.length / 2;
  const dev = new Float64Array(n);
  const half = Math.min(a, b) / 2;
  for (let i = 0; i < n; i++) {
    const px = contour[2 * i]! - e.cx, py = contour[2 * i + 1]! - e.cy;
    const u = px * ct + py * st, v = -px * st + py * ct;
    const r = Math.hypot(u / (a / 2), v / (b / 2));
    dev[i] = Math.abs(r - 1.0) * half;
  }
  const tol = Math.max(1.5, (0.06 * Math.min(a, b)) / 2);
  dev.sort();
  return percentile(dev, 90) < tol && dev[n - 1]! < 3 * tol;
}

/** np.percentile (linear) of sorted values. */
function percentile(sorted: Float64Array, q: number): number {
  const pos = (q / 100) * (sorted.length - 1);
  const i = Math.floor(pos), t = pos - i;
  const a = sorted[i]!, b = sorted[Math.min(i + 1, sorted.length - 1)]!;
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/** One candidate per blob, largest first: the same dot is found at several
 * levels and in several masks; a larger blob sharing a centre (a halo) must
 * not stand in for a smaller one (detect._dedupe_candidates, same_size). */
function dedupeCandidates(cands: Candidate[]): Candidate[] {
  const size = (c: Candidate) => Math.min(c.outer.width, c.outer.height);
  const sorted = cands.map((c, i) => [c, i] as const).sort((p, q) => size(q[0]) - size(p[0]) || p[1] - q[1]).map((p) => p[0]);
  const kept: Candidate[] = [];
  const claims = new Claims<Candidate>();
  for (const c of sorted) {
    const sc = size(c);
    const merges = claims.near(c.cx, c.cy).some((k) => {
      const sk = size(k);
      return !(Math.hypot(c.cx - k.cx, c.cy - k.cy) > 0.25 * Math.min(sc, sk) || c.polarity !== k.polarity || sk > 1.3 * sc);
    });
    if (!merges) {
      kept.push(c);
      claims.add(c.cx, c.cy, 0.25 * sc, c);
    }
  }
  return kept;
}

/** One fiducial per pin, the largest kept (detect._dedupe). */
function dedupe(found: Fiducial[]): Fiducial[] {
  const sorted = found.map((f, i) => [f, i] as const)
    .sort((p, q) => q[0].outer_axes_px[1] - p[0].outer_axes_px[1] || p[1] - q[1]).map((p) => p[0]);
  const kept: Fiducial[] = [];
  const claims = new Claims<Fiducial>();
  for (const r of sorted) {
    const [x, y] = r.center_px;
    const merges = claims.near(x, y).some((k) =>
      !(Math.hypot(x - k.center_px[0], y - k.center_px[1]) > 0.5 * k.outer_axes_px[1] || r.polarity !== k.polarity));
    if (!merges) {
      kept.push(r);
      claims.add(x, y, 0.5 * r.outer_axes_px[1], r);
    }
  }
  return kept;
}

/** Kept detections by grid cell, each listed in every cell its reach touches
 * (detect._Claims). */
class Claims<T> {
  private static readonly CELL = 32;
  private readonly cells = new Map<string, T[]>();

  add(x: number, y: number, reach: number, item: T): void {
    const r = reach * (1 + 1e-9) + 1e-6;
    const c = Claims.CELL;
    for (let i = Math.floor((x - r) / c); i <= Math.floor((x + r) / c); i++) {
      for (let j = Math.floor((y - r) / c); j <= Math.floor((y + r) / c); j++) {
        const key = `${i},${j}`;
        const list = this.cells.get(key);
        if (list) list.push(item);
        else this.cells.set(key, [item]);
      }
    }
  }

  near(x: number, y: number): T[] {
    return this.cells.get(`${Math.floor(x / Claims.CELL)},${Math.floor(y / Claims.CELL)}`) ?? [];
  }
}

// ---------------------------------------------------------------------------
// Refinement

/** Profiles along rays from the ellipse's centre out to sMax of its radius:
 * q[ray][sample] (float32), the rays' ends at s = 1, and the sample spacing.
 * Samples are taken when asked for, `sample(k0, k1)` filling k0..k1-1 on
 * every ray: a candidate turned down on its dot and margin is never sampled
 * across its edge. A sample is the same whenever it is taken. */
function rays(gray: GrayF, e: Ellipse, nRays: number, sMax: number, R: number) {
  const t = (e.angle * Math.PI) / 180;
  const ct = Math.cos(t), st = Math.sin(t);
  const A = e.width, B = e.height;
  const ex = new Float64Array(nRays), ey = new Float64Array(nRays);
  const step = (2 * Math.PI) / nRays;
  for (let i = 0; i < nRays; i++) {
    const phi = i * step;
    const c = Math.cos(phi), s = Math.sin(phi);
    ex[i] = (A / 2) * c * ct - (B / 2) * s * st;
    ey[i] = (A / 2) * c * st + (B / 2) * s * ct;
  }
  const nS = Math.ceil((sMax * R) / STEP_PX) + 1;
  const s = linspace32(0, sMax, nS);
  const ds = f32(s[1]! - s[0]!);
  const q: Float32Array[] = [];
  for (let i = 0; i < nRays; i++) q.push(new Float32Array(nS));
  const filled = new Uint8Array(nS);
  const sample = (k0: number, k1: number) => {
    for (let k = Math.max(0, k0); k < Math.min(k1, nS); k++) {
      if (filled[k]) continue;
      filled[k] = 1;
      const sk = s[k]!;
      for (let i = 0; i < nRays; i++) q[i]![k] = sampleLinear(gray, f32(e.cx + sk * ex[i]!), f32(e.cy + sk * ey[i]!));
    }
  };
  return { q, ex, ey, nS, ds, sample };
}

/** np.linspace(start, stop, n, dtype=float32). */
function linspace32(start: number, stop: number, n: number): Float32Array {
  const out = new Float32Array(n);
  const step = (stop - start) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = start + i * step;
  if (n > 1) out[n - 1] = stop;
  return out;
}

/** A dot from its candidate outline, or null if it does not look like a dot
 * on a link all the way round (detect._refine_dot). */
function refineDot(gray: GrayF, cand: Candidate, passes = 2, margin: [number, number] = DOT_MARGIN): Fiducial | null {
  let e = cand.outer;
  let res = 0;
  let contrastMed = 0, outsideMed = 0;
  for (let k = 0; k < passes; k++) {
    const need = k < passes - 1 ? 0.55 : 0.7;
    const R = Math.max(e.width, e.height) / 2;
    if (!Number.isFinite(R) || R < 3 || R > 0.25 * Math.max(gray.width, gray.height)) return null;
    const nRays = Math.trunc(Math.min(180, Math.max(32, Math.PI * R)));
    const { q, ex, ey, nS, ds, sample } = rays(gray, e, nRays, 1.5, R);
    const idx = (v: number) => Math.min(nS - 1, Math.max(0, roundHalfEven(v / ds)));
    sample(idx(0.1), idx(0.7) + 1);
    sample(idx(margin[0]), idx(margin[1]) + 1);
    const inside = new Float32Array(nRays), outside = new Float32Array(nRays);
    for (let i = 0; i < nRays; i++) {
      inside[i] = median32(q[i]!, idx(0.1), idx(0.7) + 1);
      outside[i] = median32(q[i]!, idx(margin[0]), idx(margin[1]) + 1);
    }
    const contrast = inside.map((v, i) => f32(v - outside[i]!));
    contrastMed = median32(contrast, 0, nRays);
    if (contrastMed < 15 || count(contrast, (v) => v > 6.0) / nRays < 0.85) return null;
    // The margin a steady fraction of the dot on most rays.
    const ratio = inside.map((v, i) => f32(f32(outside[i]! + 4) / f32(v + 4)));
    const mr = median32(ratio, 0, nRays);
    const close = f32(0.15);
    if (mr > 0.92 || count(ratio, (v) => Math.abs(f32(v - mr)) < close) / nRays < 0.6) return null;
    const level = inside.map((v, i) => f32(f32(v + outside[i]!) / 2));
    sample(idx(0.7), Math.max(idx(1.3), idx(0.7) + 1) + 1);
    const edge = crossings(q, level, idx(0.7), idx(1.3), 1.0 / ds, ds, false);
    const pts: number[] = [];
    for (let i = 0; i < nRays; i++) {
      if (Number.isFinite(edge[i]!)) pts.push(e.cx + edge[i]! * ex[i]!, e.cy + edge[i]! * ey[i]!);
    }
    if (pts.length / 2 < need * nRays) return null;
    const fit = robustEllipse(pts);
    if (fit === null || fit.inliers < need * nRays) return null;
    e = fit.ellipse;
    res = fit.rms;
    outsideMed = median32(outside, 0, nRays);
  }
  if (res > Math.max(0.6, 0.04 * Math.min(e.width, e.height))) return null;
  return {
    center_px: [e.cx, e.cy],
    outer_axes_px: [Math.max(e.width, e.height), Math.min(e.width, e.height)],
    inner_axes_px: [0, 0],
    // OpenCV's angle is its first axis's, which may be the minor one (detect._major_angle).
    angle_deg: e.angle + (e.height > e.width ? 90 : 0),
    residual_px: res,
    contrast: contrastMed,
    polarity: 1,
    surround: outsideMed,
  };
}

/** How much a rim's hole outline from the threshold mask is grown before it is
 * refined: the outline comes out small, and the rim is thin (detect._RIM_HOLE_GROWTH). */
const RIM_HOLE_GROWTH = 1.06;

/** A rim from the outline of its hole (detect._refine_rim). */
function refineRim(gray: GrayF, inverted: GrayF, hole: Candidate, ratio: number): Fiducial | null {
  const band = 1.0 / ratio - 1.0;
  const grown: Candidate = { ...hole, outer: { ...hole.outer, width: hole.outer.width * RIM_HOLE_GROWTH,
    height: hole.outer.height * RIM_HOLE_GROWTH } };
  const d = refineDot(inverted, grown, 2, [1.0 + 0.25 * band, 1.0 + 0.8 * band]);
  if (d === null) return null;
  const [major, minor] = d.outer_axes_px;
  const t = (d.angle_deg * Math.PI) / 180;
  const k = 1.25 / ratio / 2;
  const beyond = new Float32Array(48);
  for (let i = 0; i < 48; i++) {
    const phi = i * ((2 * Math.PI) / 48);
    const ex = k * (major * Math.cos(phi) * Math.cos(t) - minor * Math.sin(phi) * Math.sin(t));
    const ey = k * (major * Math.cos(phi) * Math.sin(t) + minor * Math.sin(phi) * Math.cos(t));
    beyond[i] = sampleLinear(gray, f32(d.center_px[0] + ex), f32(d.center_px[1] + ey));
  }
  return {
    ...d,
    outer_axes_px: [major / ratio, minor / ratio],
    inner_axes_px: [major, minor],
    polarity: 1,
    surround: median32(beyond, 0, 48),
  };
}

/** Per ray, where the profile crosses `level` between samples k0..k1 (rising
 * or falling), nearest to kTarget; NaN where it does not (detect._crossings). */
function crossings(q: Float32Array[], level: Float32Array, k0: number, k1: number, kTarget: number, ds: number, rising: boolean): Float64Array {
  k1 = Math.max(k1, k0 + 1);
  const out = new Float64Array(q.length).fill(NaN);
  q.forEach((row, i) => {
    const lv = level[i]!;
    let bestK = -1, bestD = Infinity;
    let fa = 0, fb = 0;
    for (let k = k0; k < k1; k++) {
      const a = f32(row[k]! - lv), b = f32(row[k + 1]! - lv);
      const cross = rising ? a < 0 && b >= 0 : a >= 0 && b < 0;
      if (!cross) continue;
      const d = Math.abs(k - kTarget);
      if (d < bestD) {
        bestD = d;
        bestK = k;
        fa = a;
        fb = b;
      }
    }
    if (bestK < 0) return;
    const den = f32(fa - fb);
    const frac = f32(fa / (den === 0 ? 1 : den));
    out[i] = (bestK + frac) * ds;
  });
  return out;
}

/** An ellipse through edge points, dropping outliers: the ellipse, the rms
 * distance of the inliers and how many (detect._robust_ellipse). */
function robustEllipse(pts: number[]): { ellipse: Ellipse; rms: number; inliers: number } | null {
  const n = pts.length / 2;
  let keep = new Array<boolean>(n).fill(true);
  const fitKept = (k: boolean[]) => {
    const sel: number[] = [];
    k.forEach((on, i) => {
      if (on) sel.push(f32(pts[2 * i]!), f32(pts[2 * i + 1]!));
    });
    return fitEllipseDirect(sel);
  };
  for (let it = 0; it < 3; it++) {
    if (count(keep, (v) => v) < 6) return null;
    const e = fitKept(keep);
    const d = ellipseDistance(pts, e);
    const absKept = d.filter((_, i) => keep[i]).map(Math.abs);
    const mad = 1.4826 * median64(absKept);
    const lim = Math.max(0.3, 3.5 * mad);
    const next = d.map((v) => Math.abs(v) < lim);
    if (next.every((v, i) => v === keep[i])) break;
    keep = next;
  }
  const kept = count(keep, (v) => v);
  if (kept < 6) return null;
  const e = fitKept(keep);
  const sel: number[] = [];
  keep.forEach((on, i) => {
    if (on) sel.push(pts[2 * i]!, pts[2 * i + 1]!);
  });
  const d = ellipseDistance(sel, e);
  let sq = 0;
  for (const v of d) sq += v * v;
  return { ellipse: e, rms: Math.sqrt(sq / d.length), inliers: kept };
}

/** Approximate signed distance of points from an ellipse, outward positive. */
function ellipseDistance(pts: number[], e: Ellipse): number[] {
  const t = (e.angle * Math.PI) / 180;
  const ct = Math.cos(t), st = Math.sin(t);
  const a2 = Math.max(e.width / 2, 1e-6), b2 = Math.max(e.height / 2, 1e-6);
  const out: number[] = [];
  for (let i = 0; i < pts.length; i += 2) {
    const px = pts[i]! - e.cx, py = pts[i + 1]! - e.cy;
    const u = px * ct + py * st, v = -px * st + py * ct;
    const r = Math.hypot(u / a2, v / b2);
    out.push(((r - 1.0) * Math.hypot(u, v)) / Math.max(r, 1e-9));
  }
  return out;
}

/** np.median of float32 values from..to (exclusive): the mean of the two
 * middle values computed in float32. */
let medianScratch = new Float32Array(256);

/** np.median of float32 values from..to (exclusive): the middle value, or
 * the mean of the two middle values computed in float32. Found by selection
 * on a scratch copy; the values (gray levels) are never NaN. */
function median32(v: Float32Array, from: number, to: number): number {
  const n = to - from, h = n >> 1;
  if (n > medianScratch.length) medianScratch = new Float32Array(2 * n);
  const a = medianScratch;
  a.set(v.subarray(from, to));
  const hi = select(a, n, h);
  if (n % 2) return hi;
  // The largest of the values below the h-th.
  let lo = a[0]!;
  for (let i = 1; i < h; i++) if (a[i]! > lo) lo = a[i]!;
  return f32(f32(lo + hi) / 2);
}

/** The k-th smallest of a[0..n-1] (Hoare's selection), leaving the k
 * smallest in a[0..k-1] and the k-th at a[k]. */
function select(a: Float32Array, n: number, k: number): number {
  let l = 0, r = n - 1;
  while (r > l) {
    if (r - l < 16) {
      for (let i = l + 1; i <= r; i++) {
        const x = a[i]!;
        let j = i - 1;
        while (j >= l && a[j]! > x) {
          a[j + 1] = a[j]!;
          j--;
        }
        a[j + 1] = x;
      }
      break;
    }
    const m = (l + r) >> 1;
    // Median of three as the pivot.
    if (a[m]! < a[l]!) swap(a, m, l);
    if (a[r]! < a[l]!) swap(a, r, l);
    if (a[r]! < a[m]!) swap(a, r, m);
    const pivot = a[m]!;
    let i = l, j = r;
    while (i <= j) {
      while (a[i]! < pivot) i++;
      while (a[j]! > pivot) j--;
      if (i <= j) {
        swap(a, i, j);
        i++;
        j--;
      }
    }
    if (k <= j) r = j;
    else if (k >= i) l = i;
    else break;
  }
  return a[k]!;
}

function swap(a: Float32Array, i: number, j: number): void {
  const t = a[i]!;
  a[i] = a[j]!;
  a[j] = t;
}

function median64(v: number[]): number {
  const s = v.slice().sort((a, b) => a - b);
  const n = s.length, h = n >> 1;
  return n % 2 ? s[h]! : (s[h - 1]! + s[h]!) / 2;
}

function count<T>(v: ArrayLike<T>, pred: (x: T) => boolean): number {
  let n = 0;
  for (let i = 0; i < v.length; i++) if (pred(v[i]!)) n++;
  return n;
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
