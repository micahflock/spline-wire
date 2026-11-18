// splinewire/scoring.py in TypeScript: a measurement scored against a test
// target (docs/real-world-tests.md). A plaque by its pins; a gauge by its
// contact points, placed on the edge by the best rigid fit (a search along
// the edge, then iterative closest points), with a fitted circle on a round
// gauge, with and without the end points. scoring.test.ts pins it to
// Python's scores (golden/scoring.json.gz).

import { splineSamples } from "../desktop/spline";
import { contactPoints } from "./contact";
import { PipelineError } from "./errors";
import type { Vec2 } from "./numeric";
import { gaugeByName, gaugeEdge, isPlaque, plaquePins, type Gauge } from "./targets";
import { compareToTruth, type TruthComparison } from "./truth";
import type { ChainSpec } from "./types";

const SEED_MM = 2.0;
const ICP_ROUNDS = 50;
const REFINED = 8;

export interface PlaqueScore extends TruthComparison {
  target: string;
  kind: "plaque";
  pins_matched: number;
  pins_total: number;
  first_pin: number;
}

export interface GaugePart {
  points: number;
  max_mm: number;
  rms_mm: number;
  curve_max_mm: number;
  radius_mm?: number;
  radius_error_mm?: number;
  circle_rms_mm?: number;
}

export interface GaugeScore {
  target: string;
  kind: "gauge";
  all: GaugePart;
  inner: GaugePart;
  sign?: number;
}

export type Score = PlaqueScore | GaugeScore;

/** The one number a score is judged by: a plaque's worst pin, a gauge's worst
 * contact point without the ends (scoring.headline). */
export function headline(s: Score): number {
  return s.kind === "plaque" ? s.max_error_mm : s.inner.max_mm;
}

/** Score measured pins (mm) against the named target. sign: the object's side
 * of the pins; null tries both and keeps the better (an S, a straight edge). */
export function score(target: string, pins: readonly Vec2[], spec: ChainSpec, sign: number | null = null,
  gaps: readonly number[] = []): Score {
  if (isPlaque(target)) return scorePlaque(pins, plaquePins(target, spec.pitch_mm, spec.n_pins), target);
  const gauge = gaugeByName(target);
  if (!gauge) throw new PipelineError(`no target '${target}'`);
  let best: GaugeScore | null = null;
  for (const s of sign !== null ? [sign] : [1, -1]) {
    const contacts = contactPoints(pins, spec.half_width_mm, s, spec.contact === "round", gaps);
    const result = scoreGauge(contacts, gauge);
    result.sign = s;
    if (best === null || result.inner.rms_mm < best.inner.rms_mm) best = result;
  }
  return best!;
}

export function scorePlaque(pins: readonly Vec2[], truth: readonly Vec2[], target: string): PlaqueScore {
  const n = pins.length, total = truth.length;
  if (n > total) throw new PipelineError(`found ${n} pins, but ${target} has ${total}`);
  if (n < 3) throw new PipelineError(`found ${n} pins: too few to score`);
  let best: [number, TruthComparison] | null = null;
  for (let k = 0; k + n <= total; k++) {
    const r = compareToTruth(pins, truth.slice(k, k + n));
    if (best === null || r.max_error_mm < best[1].max_error_mm) best = [k, r];
  }
  const [k, r] = best!;
  return { target, kind: "plaque", pins_matched: n, pins_total: total, first_pin: k, ...r };
}

export function scoreGauge(contacts: readonly Vec2[], gauge: Gauge): GaugeScore {
  if (contacts.length < 5) throw new PipelineError(`only ${contacts.length} contact points: too few to score`);
  const edge = new Edge(gaugeEdge(gauge));
  const part = (sel: readonly Vec2[]): GaugePart => {
    const { rot, shift, dist } = register(sel, edge, gauge.closed);
    const curve = splineSamples(sel as [number, number][]).map((p) => apply(rot, shift, p as Vec2));
    const out: GaugePart = {
      points: sel.length,
      max_mm: Math.max(...dist),
      rms_mm: Math.sqrt(dist.reduce((a, d) => a + d * d, 0) / dist.length),
      curve_max_mm: Math.max(...edge.nearest(curve).dist),
    };
    if (gauge.radius_mm !== undefined) {
      const { center, radius } = fitCircle(sel);
      const res = sel.map((p) => Math.abs(Math.hypot(p[0] - center[0], p[1] - center[1]) - radius));
      out.radius_mm = radius;
      out.radius_error_mm = radius - gauge.radius_mm;
      out.circle_rms_mm = Math.sqrt(res.reduce((a, r) => a + r * r, 0) / res.length);
    }
    return out;
  };
  return { target: gauge.name, kind: "gauge", all: part(contacts), inner: part(contacts.slice(1, -1)) };
}

/** Least-squares circle through points, geometric distances (Kasa, then Gauss–Newton). */
export function fitCircle(pts: readonly Vec2[]): { center: Vec2; radius: number } {
  // Kasa: 2x cx + 2y cy + c = x² + y², by the normal equations.
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (const [x, y] of pts) {
    const row = [2 * x, 2 * y, 1], rhs = x * x + y * y;
    for (let i = 0; i < 3; i++) {
      b[i]! += row[i]! * rhs;
      for (let j = 0; j < 3; j++) A[i]![j]! += row[i]! * row[j]!;
    }
  }
  const [cx0, cy0, c0] = solve3(A, b) as [number, number, number];
  let p: number[] = [cx0, cy0, Math.sqrt(Math.max(c0 + cx0 * cx0 + cy0 * cy0, 1e-12))];
  let lambda = 1e-6;
  const cost = (q: number[]) => pts.reduce((a, [x, y]) => a + (Math.hypot(x - q[0]!, y - q[1]!) - q[2]!) ** 2, 0);
  let f = cost(p);
  for (let it = 0; it < 200; it++) {
    const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], Jtr = [0, 0, 0];
    for (const [x, y] of pts) {
      const d = Math.max(Math.hypot(x - p[0]!, y - p[1]!), 1e-300);
      const r = d - p[2]!;
      const J = [-(x - p[0]!) / d, -(y - p[1]!) / d, -1];
      for (let i = 0; i < 3; i++) {
        Jtr[i]! += J[i]! * r;
        for (let j = 0; j < 3; j++) JtJ[i]![j]! += J[i]! * J[j]!;
      }
    }
    for (let i = 0; i < 3; i++) JtJ[i]![i]! *= 1 + lambda;
    const step = solve3(JtJ, Jtr.map((v) => -v));
    const q = p.map((v, i) => v + step[i]!) as number[];
    const fq = cost(q);
    if (fq <= f) {
      const small = Math.max(...step.map(Math.abs)) < 1e-13 * (1 + Math.abs(q[2]!));
      p = q;
      f = fq;
      lambda = Math.max(lambda / 10, 1e-12);
      if (small) break;
    } else {
      lambda *= 10;
      if (lambda > 1e12) break;
    }
  }
  return { center: [p[0]!, p[1]!], radius: p[2]! };
}

function solve3(A: number[][], b: number[]): number[] {
  const m = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < 3; c++) {
    let piv = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[piv]![c]!)) piv = r;
    [m[c], m[piv]] = [m[piv]!, m[c]!];
    for (let r = c + 1; r < 3; r++) {
      const f = m[r]![c]! / m[c]![c]!;
      for (let k = c; k < 4; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = m[r]![3]!;
    for (let k = r + 1; k < 3; k++) s -= m[r]![k]! * x[k]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

type Rot = readonly [number, number, number, number];      // [r00, r01, r10, r11]

function apply(rot: Rot, shift: Vec2, p: Vec2): Vec2 {
  return [rot[0] * p[0] + rot[1] * p[1] + shift[0], rot[2] * p[0] + rot[3] * p[1] + shift[1]];
}

/** Rotation and shift taking a onto b, least squares, no reflection. */
function procrustes(a: readonly Vec2[], b: readonly Vec2[]): { rot: Rot; shift: Vec2 } {
  const n = a.length;
  let ax = 0, ay = 0, bx = 0, by = 0;
  for (let i = 0; i < n; i++) {
    ax += a[i]![0]; ay += a[i]![1]; bx += b[i]![0]; by += b[i]![1];
  }
  ax /= n; ay /= n; bx /= n; by /= n;
  let h00 = 0, h01 = 0, h10 = 0, h11 = 0;          // H = sum (b - cb)(a - ca)^T
  for (let i = 0; i < n; i++) {
    const px = a[i]![0] - ax, py = a[i]![1] - ay, qx = b[i]![0] - bx, qy = b[i]![1] - by;
    h00 += qx * px; h01 += qx * py; h10 += qy * px; h11 += qy * py;
  }
  const th = Math.atan2(h10 - h01, h00 + h11);
  const c = Math.cos(th), s = Math.sin(th);
  const rot: Rot = [c, -s, s, c];
  return { rot, shift: [bx - (c * ax - s * ay), by - (s * ax + c * ay)] };
}

/** numpy.allclose with its default rtol. */
function close(a: readonly number[], b: readonly number[], atol: number): boolean {
  return a.every((v, i) => Math.abs(v - b[i]!) <= atol + 1e-5 * Math.abs(b[i]!));
}

function register(pts: readonly Vec2[], edge: Edge, closed: boolean): { rot: Rot; shift: Vec2; dist: number[] } {
  const e = edge.points;
  const sEdge = [0];
  for (let i = 1; i < e.length; i++) sEdge.push(sEdge[i - 1]! + Math.hypot(e[i]![0] - e[i - 1]![0], e[i]![1] - e[i - 1]![1]));
  const length = sEdge[sEdge.length - 1]!;
  const sPts = [0];
  for (let i = 1; i < pts.length; i++) sPts.push(sPts[i - 1]! + Math.hypot(pts[i]![0] - pts[i - 1]![0], pts[i]![1] - pts[i - 1]![1]));
  const starts: { rough: number; rot: Rot; shift: Vec2 }[] = [];
  const count = Math.ceil((length + 1e-9) / SEED_MM);
  for (let k = 0; k < count; k++) {
    const s0 = k * SEED_MM;
    for (const dir of [1, -1]) {
      const target = sPts.map((sp) => {
        let s = s0 + dir * sp;
        s = closed ? ((s % length) + length) % length : Math.min(Math.max(s, 0), length);
        return interp(s, sEdge, e);
      });
      const { rot, shift } = procrustes(pts, target);
      const moved = pts.map((p) => apply(rot, shift, p));
      let sum = 0;
      for (const p of moved) sum += edge.nearestVertex(p).d ** 2;
      starts.push({ rough: sum / pts.length, rot, shift });
    }
  }
  starts.sort((a, b) => a.rough - b.rough);
  let best: { rms: number; rot: Rot; shift: Vec2; dist: number[] } | null = null;
  for (const start of starts.slice(0, REFINED)) {
    let { rot, shift } = start;
    for (let it = 0; it < ICP_ROUNDS; it++) {
      const { near } = edge.nearest(pts.map((p) => apply(rot, shift, p)));
      const next = procrustes(pts, near);
      const done = close(next.rot, rot, 1e-10) && close(next.shift, shift, 1e-9);
      rot = next.rot;
      shift = next.shift;
      if (done) break;
    }
    const { dist } = edge.nearest(pts.map((p) => apply(rot, shift, p)));
    const rms = Math.sqrt(dist.reduce((a, d) => a + d * d, 0) / dist.length);
    if (best === null || rms < best.rms) best = { rms, rot, shift, dist };
  }
  return best!;
}

/** numpy.interp along the edge's arc length (sEdge increasing, perhaps with repeats). */
function interp(s: number, sEdge: readonly number[], e: readonly Vec2[]): Vec2 {
  const n = sEdge.length;
  if (s <= sEdge[0]!) return e[0]!;
  if (s >= sEdge[n - 1]!) return e[n - 1]!;
  let lo = 0, hi = n - 1;                   // the last j with sEdge[j] <= s
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sEdge[mid]! <= s) lo = mid;
    else hi = mid;
  }
  const t = (s - sEdge[lo]!) / (sEdge[lo + 1]! - sEdge[lo]!);
  return [e[lo]![0] + t * (e[lo + 1]![0] - e[lo]![0]), e[lo]![1] + t * (e[lo + 1]![1] - e[lo]![1])];
}

/** A dense polyline with a grid for nearest-vertex queries (cKDTree's job in Python). */
class Edge {
  readonly points: readonly Vec2[];
  private readonly cell = 1.0;
  private readonly x0: number;
  private readonly y0: number;
  private readonly nx: number;
  private readonly ny: number;
  private readonly bins: number[][];

  constructor(points: readonly Vec2[]) {
    this.points = points;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of points) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    this.x0 = x0; this.y0 = y0;
    this.nx = Math.floor((x1 - x0) / this.cell) + 1;
    this.ny = Math.floor((y1 - y0) / this.cell) + 1;
    this.bins = Array.from({ length: this.nx * this.ny }, () => []);
    points.forEach(([x, y], i) => this.bins[this.bin(x, y)]!.push(i));
  }

  private bin(x: number, y: number): number {
    return this.index(y, this.y0, this.ny) * this.nx + this.index(x, this.x0, this.nx);
  }

  /** The nearest vertex (lowest index on a tie) and its distance. The search
   * starts at p's cell (p moved onto the grid if outside it: no closer to any
   * vertex) and goes out ring by ring, until a ring cannot hold a closer one. */
  nearestVertex(p: Vec2): { i: number; d: number } {
    const ci = this.index(p[0], this.x0, this.nx), cj = this.index(p[1], this.y0, this.ny);
    let best = -1, bestD2 = Infinity;
    const visit = (i: number, j: number) => {
      if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return;
      for (const k of this.bins[j * this.nx + i]!) {
        const q = this.points[k]!;
        const d2 = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
        if (d2 < bestD2 || (d2 === bestD2 && k < best)) {
          bestD2 = d2;
          best = k;
        }
      }
    };
    for (let r = 0; ; r++) {
      if (best >= 0 && (r - 1) * this.cell > Math.sqrt(bestD2)) break;
      if (r === 0) visit(ci, cj);
      for (let i = ci - r; r > 0 && i <= ci + r; i++) {
        visit(i, cj - r);
        visit(i, cj + r);
      }
      for (let j = cj - r + 1; r > 0 && j <= cj + r - 1; j++) {
        visit(ci - r, j);
        visit(ci + r, j);
      }
      if (ci - r <= 0 && cj - r <= 0 && ci + r >= this.nx - 1 && cj + r >= this.ny - 1) break;
    }
    return { i: best, d: Math.sqrt(bestD2) };
  }

  private index(v: number, v0: number, n: number): number {
    return Math.min(n - 1, Math.max(0, Math.floor((v - v0) / this.cell)));
  }

  /** Each point's distance from the polyline and the nearest point on it,
   * on the segments either side of the nearest vertex (scoring._nearest). */
  nearest(pts: readonly Vec2[]): { dist: number[]; near: Vec2[] } {
    const e = this.points, last = e.length - 2;
    const dist: number[] = [], near: Vec2[] = [];
    for (const p of pts) {
      const idx = this.nearestVertex(p).i;
      let bd = Infinity, bq: Vec2 = e[idx]!;
      for (const lo0 of [idx - 1, idx]) {
        const lo = Math.min(Math.max(lo0, 0), last);
        const a = e[lo]!, b = e[lo + 1]!;
        const ux = b[0] - a[0], uy = b[1] - a[1];
        const t = Math.min(Math.max(((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / Math.max(ux * ux + uy * uy, 1e-18), 0), 1);
        const q: Vec2 = [a[0] + t * ux, a[1] + t * uy];
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        if (d < bd) {
          bd = d;
          bq = q;
        }
      }
      dist.push(bd);
      near.push(bq);
    }
    return { dist, near };
  }
}
