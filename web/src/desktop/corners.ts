// Sharp corners: points a person marks on the drawing where the target has a
// corner (a step, a ledge), so the curve has a corner there instead of a
// smooth spline rounding it off.
//
// A chain cannot follow a sharp corner. Its sleeves (or a plain chain's
// rounded link ends) stop short of a hollow corner, and over a ridge the link
// that spans it rests on the ridge with its arm, its pin lifted clear: the
// contact points there are off the corner, 0.6-0.9 mm in a hollow and about
// 4 mm over a ridge with the snap-fit chain (corners.test.ts, wrap-sim.ts).
// So the corner is rebuilt from the walls either side of it:
//
//   - a wall is the line the sleeves of two neighbouring pins rest on: the line
//     through the two pins, moved one sleeve radius towards the object. That
//     holds while both pins touch the wall, wherever along it they are, so a
//     wall two pins long is enough;
//   - the pins between the two walls' links (the zone) touch the corner, or
//     nothing, so their points are left out;
//   - the corner is where the two walls meet. The zone's end pins touch the
//     walls, so their points become their sleeves' touching points on them.
//
// Which pins form the zone is not marked, only a point at the corner. The
// zone holds the marked pin and up to two more either side, three at most,
// and is chosen by what the chain could not do: no pin may sit closer to the
// object the two walls make than its sleeve, and no link may pass through it.
// Of the zones left, the one whose corner is farthest from every pin is taken:
// a wall along a link that crosses the corner meets the other wall short of
// it, nearer the chain. Where no zone fits (the curve is straight there), the
// curve only kinks at the point.
//
// Simulated with the snap-fit chain pressed onto single corners of 45-120°
// and staircases (corners.test.ts), the corner is found to 1.2 mm at worst, 0
// in the median, where each wall has two pins resting on it: walls about three
// pitches (40 mm) long or more. A wall one pin rests on has no direction to
// read; shorter steps come out wrong by several mm.
//
// The points are the pins' own when there is one point per pin (round contact:
// the snap-fit chain; a plain chain bent only away from the object); otherwise
// the points themselves stand for the pins, with no offset.
//
// Marks are kept as positions in mm (the marked point's), not as indices: a
// re-measured photo (a new revision) moves the points a little, and a mark
// stays with the point nearest it, within MARK_REACH_MM.

import type { Point } from "../relay/messages";

/** A mark finds its point within this distance, in mm. */
export const MARK_REACH_MM = 3;
/** How far into the object a pin or link may seem to be, in mm: the pins are
 * measured to about 0.1 mm. */
const FIT_TOL_MM = 0.2;
/** A link's arm, in sleeve radii: the snap-fit chain's is 2.5 mm against a 4.39 mm
 * sleeve, the O-ring chain's as wide as its rounded ends. */
const ARM = 0.5;
/** How far from the marked pin a corner may be, in pitches: an acute hollow's
 * corner lies far beyond the pins that cannot reach into it. */
const REACH = 3;
/** Walls that turn less than this (degrees) meet too far away to be a corner. */
const MIN_TURN_DEG = 12;

/** The curve with its corners: the points to fit through, and where it breaks. */
export interface Shaped {
  /** Points in chain order, corners included. */
  points: Point[];
  /** Indices into `points` where the curve has a corner: each piece ends on one
   * and the next starts on it. Ascending, never the first or the last point. */
  corners: number[];
  /** Per mark, in the order given: the index into the curve's own points it
   * resolved to, or -1 if no point is near it. */
  marked: number[];
  /** Per corner: the curve's own point it was marked on, and the points it
   * replaced (from..to; none, to < from, where the curve only kinks there). */
  zones: { source: number; from: number; to: number }[];
  /** Indices into the curve's own points that a corner replaced: those that
   * rounded it off, and the two either side moved onto their walls. */
  replaced: number[];
}

interface Frame {
  /** The pins, or the points when there is not one point per pin. */
  pins: Point[];
  /** Which side of them the object is: +1 their left, -1 their right. */
  side: number;
  /** From a pin to the wall it rests on, mm: the sleeve, or 0 for points. */
  reach: number;
}

function frame(points: readonly Point[], pins: readonly Point[], halfWidth: number): Frame {
  // The points lie off the pin line on the object's side.
  let s = 0;
  for (const p of points) {
    let best = Infinity, cross_ = 0;
    for (let i = 0; i + 1 < pins.length; i++) {
      const a = pins[i]!, d = sub(pins[i + 1]!, a);
      const t = Math.max(0, Math.min(1, dot(sub(p, a), d) / Math.max(dot(d, d), 1e-12)));
      const e = sub(p, [a[0] + t * d[0], a[1] + t * d[1]]);
      if (Math.hypot(e[0], e[1]) < best) {
        best = Math.hypot(e[0], e[1]);
        cross_ = cross(d, e);
      }
    }
    s += cross_;
  }
  const side = s >= 0 ? 1 : -1;
  if (pins.length !== points.length || pins.length < 3) return { pins: points.map(copy), side, reach: 0 };
  return { pins: pins.map(copy), side, reach: halfWidth };
}

/** The index of the point nearest `at`, or -1 if none is within MARK_REACH_MM. */
export function nearestPoint(points: readonly Point[], at: Point, reach = MARK_REACH_MM): number {
  let best = -1, bestD = reach;
  points.forEach((p, i) => {
    const d = Math.hypot(p[0] - at[0], p[1] - at[1]);
    if (d <= bestD) {
      best = i;
      bestD = d;
    }
  });
  return best;
}

interface Wall { at: Point; dir: Point }

/** The wall the sleeves of pins i and j rest on (j = i ± 1), passing by pin j. */
function wall(f: Frame, i: number, j: number): Wall {
  const [a, b] = i < j ? [f.pins[i]!, f.pins[j]!] : [f.pins[j]!, f.pins[i]!];
  const d = unit(sub(b, a));
  const left: Point = [-d[1], d[0]];
  const p = f.pins[j]!, o = f.side * f.reach;
  return { at: [p[0] + o * left[0], p[1] + o * left[1]], dir: d };
}

interface Fit { from: number; to: number; corner: Point; first: Point; last: Point; dir1: Point; dir2: Point }

/** The sharp corner at point k, its zone after `after` (no zone may reach back to it). */
function fit(f: Frame, k: number, after: number): Fit | null {
  const n = f.pins.length;
  const pitch = medianLink(f.pins);
  let best: Fit | null = null, bestDepth = -Infinity;
  for (let a = k - 2; a <= k; a++) {
    for (let b = k; b <= k + 2 && b - a <= 2; b++) {
      if (a < 1 || b > n - 2 || a <= after) continue;
      const w1 = wall(f, a - 1, a);
      const w2 = wall(f, b + 1, b);
      const turn = Math.asin(Math.min(1, Math.abs(cross(w1.dir, w2.dir))));
      if (turn < (MIN_TURN_DEG * Math.PI) / 180 && dot(w1.dir, w2.dir) > 0) continue;
      const c = meet(w1, w2);
      if (!c) continue;
      // Ahead of both walls' links: past pin a on the first wall, before pin b
      // on the second, unless that pin's sleeve rests on the corner itself
      // (a ridge's tip, touched from beyond it).
      const tol = 0.05 * pitch;
      const onTip = (i: number) => Math.abs(Math.hypot(c[0] - f.pins[i]![0], c[1] - f.pins[i]![1]) - f.reach) < FIT_TOL_MM;
      if (dot(sub(c, w1.at), w1.dir) < -tol && !onTip(a)) continue;
      if (dot(sub(c, w2.at), w2.dir) > tol && !onTip(b)) continue;
      if (Math.hypot(c[0] - f.pins[k]![0], c[1] - f.pins[k]![1]) > REACH * pitch) continue;
      // Nothing of the chain may be inside the object this corner makes: no
      // pin nearer to it than the sleeve, no link through it.
      const inside = depthIn(f, w1, c, w2);
      const sleeve = f.reach, arm = ARM * sleeve;
      let ok = true, depth = Infinity;
      for (let i = a - 1; i <= b + 1 && ok; i++) {
        const d = inside(f.pins[i]!);
        if (d < sleeve - FIT_TOL_MM) ok = false;
        depth = Math.min(depth, Math.hypot(c[0] - f.pins[i]![0], c[1] - f.pins[i]![1]));
        if (i <= b) {
          for (let s = 1; s < 8 && ok; s++) {
            const p = f.pins[i]!, q = f.pins[i + 1]!;
            if (inside([p[0] + (s / 8) * (q[0] - p[0]), p[1] + (s / 8) * (q[1] - p[1])]) < arm - FIT_TOL_MM) ok = false;
          }
        }
      }
      // Of the corners that fit, the deepest: walls along links that cross the
      // corner meet it short, nearer the chain.
      if (ok && depth > bestDepth) {
        bestDepth = depth;
        best = { from: a, to: b, corner: c, first: w1.at, last: w2.at, dir1: w1.dir, dir2: w2.dir };
      }
    }
  }
  return best;
}

/** Signed distance from a point to the object the two walls make, meeting at
 * c: positive outside it (the chain's side), negative inside. */
function depthIn(f: Frame, w1: Wall, c: Point, w2: Wall): (p: Point) => number {
  const side = f.side;
  // Walls turning towards the object close round it (a ridge: inside both
  // half-planes); turning away, they open round it (a hollow: inside either).
  const ridge = cross(w1.dir, w2.dir) * side > 0;
  return (p: Point) => {
    const v = sub(p, c);
    const in1 = cross(w1.dir, v) * side > 0;
    const in2 = cross(w2.dir, v) * side > 0;
    const d = Math.min(rayDist(v, w1.dir, -1), rayDist(v, w2.dir, 1));
    return (ridge ? in1 && in2 : in1 || in2) ? -d : d;
  };
}

/** Distance from v to the ray from the origin along dir * way. */
function rayDist(v: Point, dir: Point, way: number): number {
  const t = Math.max(0, way * dot(v, dir));
  return Math.hypot(v[0] - way * t * dir[0], v[1] - way * t * dir[1]);
}

/** The curve's points with the marked corners made sharp. `marks` are
 * positions in mm (see the top of this file); marks that find no point, or the
 * same point as an earlier one, are ignored. */
export function shape(points: readonly Point[], pins: readonly Point[], halfWidth: number, marks: readonly Point[]): Shaped {
  const marked = marks.map((m) => nearestPoint(points, m));
  const ks = [...new Set(marked.filter((k) => k > 0 && k < points.length - 1))].sort((x, y) => x - y);
  const f = frame(points, pins, halfWidth);
  const out: Point[] = [];
  const corners: number[] = [];
  const replaced: number[] = [];
  const zones: Shaped["zones"] = [];
  let next = 0;                 // the first of the curve's points not yet copied
  for (const k of ks) {
    if (k < next) continue;     // inside the last corner's zone
    const z = fit(f, k, next - 1);
    for (; next < (z ? z.from : k); next++) out.push(copy(points[next]!));
    if (!z) {
      // No corner fits here: the curve kinks at the point itself.
      corners.push(out.length);
      zones.push({ source: k, from: k, to: k - 1 });
      out.push(copy(points[k]!));
      next = k + 1;
      continue;
    }
    // The zone's end pins touch the walls: their points are where they touch.
    // A zone of one pin touches both walls, either side of the corner.
    // A sleeve resting on a ridge's tip from beyond it touches no wall: no point.
    if (dot(sub(z.corner, z.first), z.dir1) > 0) out.push(z.first);
    corners.push(out.length);
    zones.push({ source: k, from: z.from, to: z.to });
    out.push(z.corner);
    if (dot(sub(z.last, z.corner), z.dir2) > 0) out.push(z.last);
    for (let i = z.from; i <= z.to; i++) replaced.push(i);
    next = z.to + 1;
  }
  for (; next < points.length; next++) out.push(copy(points[next]!));
  return { points: dedupe(out, corners), corners, marked, zones, replaced };
}

/** Consecutive points closer than 0.01 mm merged (a pin touching its wall at
 * the corner itself), keeping the corner indices right. */
function dedupe(points: Point[], corners: number[]): Point[] {
  const keep: Point[] = [];
  const map: number[] = [];
  for (const p of points) {
    const last = keep[keep.length - 1];
    if (last && Math.hypot(p[0] - last[0], p[1] - last[1]) < 0.01) {
      map.push(keep.length - 1);
      continue;
    }
    map.push(keep.length);
    keep.push(p);
  }
  for (let i = 0; i < corners.length; i++) corners[i] = map[corners[i]!]!;
  return keep;
}

/** The pieces between corners, each a list of points; neighbours share the corner point. */
export function pieces(s: Pick<Shaped, "points" | "corners">): Point[][] {
  const out: Point[][] = [];
  let start = 0;
  for (const c of s.corners) {
    out.push(s.points.slice(start, c + 1));
    start = c;
  }
  out.push(s.points.slice(start));
  return out;
}

function meet(w1: Wall, w2: Wall): Point | null {
  const den = cross(w1.dir, w2.dir);
  if (Math.abs(den) < 1e-9) return null;
  const t = cross(sub(w2.at, w1.at), w2.dir) / den;
  return [w1.at[0] + t * w1.dir[0], w1.at[1] + t * w1.dir[1]];
}

function medianLink(pins: readonly Point[]): number {
  const d = pins.slice(1).map((p, i) => Math.hypot(p[0] - pins[i]![0], p[1] - pins[i]![1])).sort((x, y) => x - y);
  return d[d.length >> 1] ?? 0;
}

const copy = (p: Point): Point => [p[0], p[1]];
const sub = (a: Point, b: Point): Point => [a[0] - b[0], a[1] - b[1]];
const dot = (a: Point, b: Point) => a[0] * b[0] + a[1] * b[1];
const cross = (a: Point, b: Point) => a[0] * b[1] - a[1] * b[0];
function unit(v: Point): Point {
  const l = Math.hypot(v[0], v[1]);
  return [v[0] / l, v[1] / l];
}

/** The marks after a click on the curve's own point i: a mark on it, or a
 * corner whose zone holds it, goes; otherwise i is marked. The ends cannot be. */
export function toggleMark(points: readonly Point[], s: Pick<Shaped, "marked" | "zones">, marks: readonly Point[], i: number): Point[] {
  const zone = s.zones.find((z) => z.source === i || (z.from <= i && i <= z.to));
  if (zone) return marks.filter((_, j) => s.marked[j] !== zone.source);
  if (s.marked.includes(i)) return marks.filter((_, j) => s.marked[j] !== i);
  if (i <= 0 || i >= points.length - 1) return marks.slice();
  return [...marks, copy(points[i]!)];
}
