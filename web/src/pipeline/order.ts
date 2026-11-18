// splinewire/order.py: put unlabeled detections in chain order by walking
// about one pitch at a time, turning as little as possible. See that file for
// the reasoning; this follows it step for step. The walk that wins is chosen
// by comparing sums, medians and angles, so they are computed as NumPy
// computes them (numeric.ts) to give the same order on near ties.

import { PipelineError } from "./errors";
import { fsum, median, medianSorted, norm2, roundHalfEven, type Vec2 } from "./numeric";
import type { ChainOrder } from "./types";

export interface OrderOptions {
  maxTurnDeg?: number;
  axesPx?: readonly Vec2[] | null;        // each fiducial's outer (major, minor) diameter
  pitchPerDiameter?: number | null;       // pitch / fiducial diameter, from the spec
  nPins?: number | null;
  strength?: readonly number[] | null;    // how clearly each detection stands out (contrast)
}

// Walks start from at most this many detections (busy backgrounds only), and
// then stop at this many times the chain's pins: order.py says why.
const MAX_STARTS = 600;
const BUSY_WALK = 2;
// Radians of total turning that count as equal: order_chain says why.
const TURN_TIE = 1e-9;

export function orderChain(pointsPx: readonly Vec2[], opts: OrderOptions = {}): ChainOrder {
  const pts = pointsPx;
  const n = pts.length;
  if (n < 3) throw new PipelineError(`need at least 3 detections to find a chain, got ${n}`);
  const maxTurn = (opts.maxTurnDeg ?? 80) * (Math.PI / 180);     // as np.radians rounds it
  const near = new Near(pts);
  const nnPitch = median(near.nearest());
  const sizes = opts.axesPx && opts.pitchPerDiameter
    ? new Sizes(opts.axesPx.map((a) => a[0]), opts.axesPx.map((a) => a[1]), opts.pitchPerDiameter)
    : null;
  const strength = opts.strength ?? null;
  const nPins = opts.nPins || null;

  const starts = startsFor(pts, near, nnPitch, sizes, strength);
  const maxLen = nPins && starts.length < n ? BUSY_WALK * nPins : null;
  let bestPath: number[] | null = null;
  let bestKey: number[] | null = null;
  let bestTurn = 0;
  for (const start of starts) {
    const [path, turnTotal] = walk(start, pts, near, nnPitch, maxTurn, sizes, maxLen);
    const length = nPins ? Math.min(path.length, nPins) : path.length;
    const [off, gaps] = offPitch(path, pts);
    // fsum: walks over the same detections in another order tie exactly.
    const standsOut = strength ? (fsum(path.map((i) => strength[i]!)) * length) / path.length : 0;
    // Least turning last, and only by more than rounding: the two directions
    // along one chain turn alike, and which adds up a hair less depends on
    // the platform's atan2 (V8's and glibc's differ in the last bit).
    const key = [length, standsOut, -off, -gaps];
    if (bestKey === null || greater(key, bestKey) || (equal(key, bestKey) && turnTotal < bestTurn - TURN_TIE)) {
      bestPath = path;
      bestKey = key;
      bestTurn = turnTotal;
    }
  }
  const path = bestPath!;
  const steps = stepsOf(path, pts);
  const links: [number, number][] = [];
  const gapList: number[] = [];
  steps.forEach((s, k) => {
    const window = median(steps.slice(Math.max(0, k - 3), k + 4));
    let local: number;
    if (sizes) {
      const a = path[k]!, b = path[k + 1]!;
      const expected = (sizes.k * (sizes.major[a]! + sizes.major[b]!)) / 2;
      local = Math.min(window, expected);
    } else {
      local = Math.min(window, nnPitch * 1.3);
    }
    if (s > 1.5 * local) gapList.push(k);
    else links.push([k, k + 1]);
  });
  const on = new Set(path);
  const rejected: number[] = [];
  for (let i = 0; i < n; i++) if (!on.has(i)) rejected.push(i);
  return { indices: path, links, gaps: gapList, rejected };
}

function equal(a: readonly number[], b: readonly number[]): boolean {
  return a.every((v, i) => v === b[i]);
}

/** Python's tuple comparison a > b. */
function greater(a: readonly number[], b: readonly number[]): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i]! > b[i]!) return true;
    if (a[i]! < b[i]!) return false;
  }
  return false;
}

/** The detections to start walks from, in increasing order (order._starts). */
function startsFor(
  pts: readonly Vec2[], near: Near, nnPitch: number, sizes: Sizes | null, strength: readonly number[] | null,
  limit = MAX_STARTS,
): number[] {
  const n = pts.length;
  if (n <= limit) return Array.from({ length: n }, (_, i) => i);
  const pitch = sizes ? sizes.major.map((mj, i) => sizes.k * 0.5 * (mj + sizes.minor[i]!)) : new Array(n).fill(nnPitch);
  const count = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const p = pitch[i]!;
    for (const j of near.within(pts[i]!, 1.5 * p)) {
      const d = norm2(pts[j]![0] - pts[i]![0], pts[j]![1] - pts[i]![1]);
      let ok = d > 0.5 * p && d < 1.5 * p;
      if (ok && sizes) {
        const r = sizes.major[i]! / sizes.major[j]!;
        ok = 1 / 1.35 < r && r < 1.35;
      }
      if (ok) count[i]!++;
    }
  }
  // No neighbour last; then fewest neighbours, then standing out most.
  const rank = count.map((c) => (c === 0 ? n : c));
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) =>
    rank[a]! - rank[b]! || (strength ? strength[b]! - strength[a]! : 0) || a - b);
  return order.slice(0, limit).sort((a, b) => a - b);
}

/** Detections around a point, from points sorted by x (in place of SciPy's
 * k-d tree). Distances are computed as order.py's table computed them. */
class Near {
  private readonly byX: number[];
  private readonly xs: number[];
  private readonly balls = new Map<number, { radius: number; idx: number[]; dist: number[] }>();

  constructor(readonly pts: readonly Vec2[]) {
    this.byX = Array.from({ length: pts.length }, (_, i) => i).sort((a, b) => pts[a]![0] - pts[b]![0] || a - b);
    this.xs = this.byX.map((i) => pts[i]![0]);
  }

  /** Indices, increasing, of every detection within `radius` of p (and
   * perhaps a hair beyond: the caller filters by distance). */
  within(p: Vec2, radius: number): number[] {
    const r = radius * (1 + 1e-9) + 1e-9;
    let lo = 0, hi = this.xs.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.xs[mid]! < p[0] - r) lo = mid + 1;
      else hi = mid;
    }
    const out: number[] = [];
    for (let k = lo; k < this.xs.length && this.xs[k]! <= p[0] + r; k++) {
      const j = this.byX[k]!;
      if (norm2(this.pts[j]![0] - p[0], this.pts[j]![1] - p[1]) <= r) out.push(j);
    }
    return out.sort((a, b) => a - b);
  }

  /** Each detection's distance to its nearest other one. */
  nearest(): number[] {
    const n = this.pts.length;
    const out = new Array<number>(n).fill(Infinity);
    for (let a = 0; a < n; a++) {
      // Walk outward in x from a's place until x alone is farther than the best.
      const xa = this.pts[this.byX[a]!]![0], ya = this.pts[this.byX[a]!]![1];
      let best = Infinity;
      for (const dir of [-1, 1]) {
        for (let k = a + dir; k >= 0 && k < n; k += dir) {
          const dxa = Math.abs(this.xs[k]! - xa);
          if (dxa > best) break;
          const j = this.byX[k]!;
          const d = norm2(this.pts[j]![0] - xa, this.pts[j]![1] - ya);
          if (d < best) best = d;
        }
      }
      out[this.byX[a]!] = best;
    }
    return out;
  }

  /** What lies around detection i, out to at least `radius`: indices
   * (increasing) and distances, its own distance infinite. */
  around(i: number, radius: number): { idx: number[]; dist: number[] } {
    let ball = this.balls.get(i);
    if (ball === undefined || ball.radius < radius) {
      const idx = this.within(this.pts[i]!, radius);
      const p = this.pts[i]!;
      const dist = idx.map((j) => (j === i ? Infinity : norm2(this.pts[j]![0] - p[0], this.pts[j]![1] - p[1])));
      ball = { radius, idx, dist };
      this.balls.set(i, ball);
    }
    return ball;
  }
}

class Sizes {
  constructor(readonly major: readonly number[], readonly minor: readonly number[], readonly k: number) {}

  /** Plausible image length of a step of `pitches` pitches from fiducial i. */
  stepRange(i: number, pitches: number): [number, number] {
    const mi = this.minor[i]!, ma = this.major[i]!;
    return [0.7 * pitches * this.k * mi - 0.5 * mi, 1.3 * pitches * this.k * ma + 0.5 * mi];
  }

  /** Whether a step from i to j of image length d (gap: over a missing pin)
   * has a plausible length, joins look-alikes and stays within 1.6x of the
   * walk's median size `band`. */
  fits(i: number, j: number, d: number, gap: boolean, band: number): boolean {
    const [lo, hi] = this.stepRange(i, gap ? 2 : 1);
    const major = this.major[j]!, minor = this.minor[j]!;
    const r = this.major[i]! / major;
    const aspect = Math.abs(this.minor[i]! / this.major[i]! - minor / major);
    return lo <= d && d <= hi && 1 / 1.35 < r && r < 1.35
      && aspect < 0.15 + 2.0 / Math.min(this.minor[i]!, minor)
      && band / 1.6 < major && major < band * 1.6;
  }
}

function walk(
  start: number, pts: readonly Vec2[], near: Near, nnPitch: number, maxTurn: number, sizes: Sizes | null,
  maxLen: number | null,
): [number[], number] {
  const path = [start];
  const visited = new Set([start]);
  let pitch = sizes ? sizes.k * 0.5 * (sizes.major[start]! + sizes.minor[start]!) : nnPitch;
  let heading: Vec2 | null = null;
  const singleSteps: number[] = [];
  const pathSizes = sizes ? [sizes.major[start]!] : [];       // kept sorted, for the median
  let turnTotal = 0;
  const cosMax = maxTurn < Math.PI ? Math.cos(maxTurn) : -2;
  while (maxLen === null || path.length < maxLen) {
    const cur = path[path.length - 1]!;
    const pc = pts[cur]!;
    const { idx, dist } = near.around(cur, 2.4 * pitch);
    const band = sizes ? medianSorted(pathSizes) : 0;
    const sel: number[] = [];
    for (let k = 0; k < idx.length; k++) {
      const d = dist[k]!;
      if (!(d > 0.5 * pitch && d < 2.4 * pitch)) continue;
      if (sizes && !sizes.fits(cur, idx[k]!, d, d / pitch > 1.5, band)) continue;
      if (heading !== null) {
        const q = pts[idx[k]!]!;
        const cos = ((q[0] - pc[0]) * heading[0] + (q[1] - pc[1]) * heading[1]) / (d * Math.hypot(heading[0], heading[1]));
        if (!(cos >= cosMax - 1e-9)) continue;
      }
      sel.push(k);
    }
    const over = passesOver(idx, dist, sel.filter((k) => !(dist[k]! / pitch > 1.5)), pts, pitch, visited);
    let best = -1, bestScore = Infinity, bestTurn = 0, bestLen = 0;
    for (const k of sel) {
      const j = idx[k]!, d = dist[k]!;
      const ratio = d / pitch;
      const isGap = ratio > 1.5;
      if (visited.has(j)) continue;
      const step: Vec2 = [pts[j]![0] - pc[0], pts[j]![1] - pc[1]];
      const turn = heading === null ? 0 : Math.abs(angle(heading, step));
      if (turn > maxTurn) continue;
      let score = turn + 2.0 * Math.abs(ratio - (isGap ? 2.0 : 1.0)) + (isGap ? 1.0 : 0.0);
      if (!isGap && over.get(k)) score += 1.0;
      if (score < bestScore) {
        best = j;
        bestScore = score;
        bestTurn = turn;
        bestLen = d;
      }
    }
    if (best < 0) return [path, turnTotal];
    if (bestLen <= 1.5 * pitch) {
      singleSteps.push(bestLen);
      pitch = median(singleSteps.slice(-5));
    }
    heading = [pts[best]![0] - pc[0], pts[best]![1] - pc[1]];
    turnTotal += bestTurn;
    path.push(best);
    visited.add(best);
    if (sizes) insort(pathSizes, sizes.major[best]!);
  }
  return [path, turnTotal];
}

function insort(v: number[], x: number): void {
  let lo = 0, hi = v.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < v[mid]!) hi = mid;
    else lo = mid + 1;
  }
  v.splice(lo, 0, x);
}

function stepsOf(path: readonly number[], pts: readonly Vec2[]): number[] {
  const out: number[] = [];
  for (let k = 0; k + 1 < path.length; k++) {
    const a = pts[path[k]!]!, b = pts[path[k + 1]!]!;
    out.push(norm2(a[0] - b[0], a[1] - b[1]));
  }
  return out;
}

/** Steps more than 10% off a whole pitch, and steps of two pitches. */
function offPitch(path: readonly number[], pts: readonly Vec2[]): [number, number] {
  const steps = stepsOf(path, pts);
  if (steps.length === 0) return [0, 0];
  const med = median(steps);
  let off = 0, two = 0;
  for (const s of steps) {
    const ratio = s / med;
    const whole = Math.min(2, Math.max(1, roundHalfEven(ratio)));
    if (Math.abs(ratio - whole) > 0.1) off++;
    if (whole === 2) two++;
  }
  return [off, two];
}

/** For each step from cur to idx[k] (k in `steps`): whether some unvisited
 * detection is about one pitch from both its ends, and nearer to each than
 * they are to each other (order._passes_over). */
function passesOver(
  idx: readonly number[], dist: readonly number[], steps: readonly number[], pts: readonly Vec2[], pitch: number,
  visited: ReadonlySet<number>,
): Map<number, boolean> {
  const out = new Map<number, boolean>();
  for (const k of steps) out.set(k, false);
  const mid: number[] = [];
  for (let m = 0; m < idx.length; m++) {
    if (Math.abs(dist[m]! / pitch - 1.0) < 0.25 && !visited.has(idx[m]!)) mid.push(m);
  }
  if (mid.length === 0 || steps.length === 0) return out;
  for (const k of steps) {
    const j = idx[k]!, d = dist[k]!;
    const pj = pts[j]!;
    out.set(k, mid.some((m) => {
      const q = pts[idx[m]!]!;
      const toJ = norm2(q[0] - pj[0], q[1] - pj[1]);
      return dist[m]! < d && idx[m] !== j && Math.abs(toJ / pitch - 1.0) < 0.25 && toJ < d;
    }));
  }
  return out;
}

function angle(a: Vec2, b: Vec2): number {
  return Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
}
