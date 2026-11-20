import { describe, expect, it } from "vitest";
import { MARK_REACH_MM, nearestPoint, pieces, shape, toggleMark } from "./corners";
import { splineSamples } from "./spline";
import { SNAP_LINK, staircase, wrap, type P } from "./wrap-sim";
import { contactPoints } from "../pipeline/contact";

const R = SNAP_LINK.sleeve;

/** A corner turning `deg` (counter-clockwise positive: the chain, on the left,
 * in a hollow; negative: over a ridge), with walls 60 mm long. */
function corner(deg: number): P[] {
  const t = (deg * Math.PI) / 180;
  return [[-60, 0], [0, 0], [60 * Math.cos(t), 60 * Math.sin(t)]];
}

/** The snap-fit chain pressed onto `profile`, and its contact points (the object is on the right). */
function chainOn(profile: P[], start: number) {
  const pins = wrap(profile, start);
  return { pins, points: contactPoints(pins, R, -1, true) as P[] };
}

const dist = (a: P, b: P) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const nearest = (p: P, set: readonly P[]) => Math.min(...set.map((q) => dist(p, q)));

/** The profile, densely, from where the chain's first point touches it to its last. */
function profileSpan(profile: P[], points: P[]): P[] {
  const dense: P[] = [];
  for (let i = 0; i + 1 < profile.length; i++) {
    const [a, b] = [profile[i]!, profile[i + 1]!];
    const n = Math.ceil(dist(a, b) / 0.1);
    for (let k = 0; k < n; k++) dense.push([a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n]);
  }
  const at = (q: P) => dense.reduce((best, p, i) => (dist(p, q) < dist(dense[best]!, q) ? i : best), 0);
  return dense.slice(at(points[0]!), at(points[points.length - 1]!) + 1);
}

/** How far the drawn curve and the profile are apart at worst (both ways), mm. */
function deviation(curve: P[], span: P[]): number {
  return Math.max(Math.max(...curve.map((c) => nearest(c, span))), Math.max(...span.map((s) => nearest(s, curve))));
}

/** Mark the point nearest each corner of the profile; null if a corner lies by an end of the chain. */
function markCorners(profile: P[], points: P[]): { marks: P[]; vertices: P[] } | null {
  const marks: P[] = [];
  const vertices: P[] = [];
  for (const v of profile.slice(1, -1)) {
    const k = points.reduce((best, p, i) => (dist(p, v) < dist(points[best]!, v) ? i : best), 0);
    if (k < 2 || k > points.length - 3) {
      if (nearest(v, points) < 15) return null;
      continue;
    }
    marks.push(points[k]!);
    vertices.push(v);
  }
  return { marks, vertices };
}

function drawn(points: P[], corners: number[] = []): P[] {
  return pieces({ points, corners }).flatMap((p) => splineSamples(p, 20) as P[]);
}

describe("sharp corners on the simulated snap-fit chain", () => {
  // Every 2 mm of where the chain starts puts the corner somewhere else along a link.
  const starts = [2, 4, 6, 8, 10, 12, 14];

  for (const deg of [-100, -90, -60, -45, 45, 60, 90, 120]) {
    it(`finds a ${Math.abs(deg)}° ${deg < 0 ? "ridge" : "hollow"} the chain rounds off`, () => {
      const before: number[] = [], after: number[] = [], errors: number[] = [];
      for (const start of starts) {
        const profile = corner(deg);
        const { pins, points } = chainOn(profile, start);
        const m = markCorners(profile, points)!;
        const s = shape(points, pins, R, m.marks);
        expect(s.corners).toHaveLength(1);
        errors.push(dist(s.points[s.corners[0]!]!, m.vertices[0]!));
        const span = profileSpan(profile, points);
        before.push(deviation(drawn(points), span));
        after.push(deviation(drawn(s.points, s.corners), span));
      }
      // Before: the spline rounds the corner off by 1-14 mm.
      expect(Math.max(...before)).toBeGreaterThan(1);
      expect(Math.max(...errors)).toBeLessThan(1.2);
      expect(Math.max(...after)).toBeLessThan(1);
    });
  }

  // Steps 40 mm or more: two pins or more rest on every wall.
  for (const [tread, rise] of [[40, 40], [60, 40]] as const) {
    it(`finds the corners of ${tread} × ${rise} mm steps, all marked`, () => {
      const errors: number[] = [], after: number[] = [], before: number[] = [];
      for (const start of starts) {
        const profile = staircase(3, tread, rise);
        const { pins, points } = chainOn(profile, start);
        const m = markCorners(profile, points);
        if (!m) continue;
        const s = shape(points, pins, R, m.marks);
        const found = s.corners.map((i) => s.points[i]!);
        for (const v of m.vertices) errors.push(nearest(v, found));
        const span = profileSpan(profile, points);
        before.push(deviation(drawn(points), span));
        after.push(deviation(drawn(s.points, s.corners), span));
      }
      expect(errors.length).toBeGreaterThan(10);
      expect(Math.min(...before)).toBeGreaterThan(4);
      expect(Math.max(...errors)).toBeLessThan(1.5);
      expect(Math.max(...after)).toBeLessThan(1);
    });
  }
});

describe("shape", () => {
  const line = (n: number): P[] => Array.from({ length: n }, (_, i) => [13 * i, 0]);

  it("leaves a curve with no marks as it is", () => {
    const points = line(5);
    const s = shape(points, points, 0, []);
    expect(s.points).toEqual(points);
    expect(s.corners).toEqual([]);
    expect(pieces(s)).toEqual([points]);
  });

  it("only kinks a straight curve, at the point marked", () => {
    const points = line(6);
    const s = shape(points, points, 0, [[26.5, 0.5]]);
    expect(s.points).toEqual(points);
    expect(s.corners).toEqual([2]);
    expect(s.replaced).toEqual([]);
    expect(pieces(s)).toEqual([points.slice(0, 3), points.slice(2)]);
  });

  it("ignores marks at the ends, far from any point, or twice on one point", () => {
    const points = line(5);
    const s = shape(points, points, 0, [[0, 0], [52, 0], [26, MARK_REACH_MM + 1], [13, 0], [13.2, 0]]);
    expect(s.marked).toEqual([0, 4, -1, 1, 1]);
    expect(s.corners).toEqual([1]);
  });

  it("finds the corner from the points alone when there is not one per pin", () => {
    // An L of points, 13 mm apart, with the corner point rounded off.
    const points: P[] = [[-39, 0], [-26, 0], [-13, 0], [-3, 3], [0, 13], [0, 26], [0, 39]].map(([x, y]) => [x!, y!]);
    // Fewer pins than points, off the points on the chain's side (up and left).
    const pins: P[] = points.slice(0, -1).map(([x, y]) => [x - 2, y + 2]);
    const s = shape(points, pins, R, [[-3, 3]]);
    expect(s.corners).toHaveLength(1);
    const c = s.points[s.corners[0]!]!;
    expect(dist(c, [0, 0])).toBeLessThan(1e-9);
    expect(s.replaced).toContain(3);
  });

  it("splits into pieces that share their corner", () => {
    const s = { points: [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] as P[], corners: [1, 3] };
    expect(pieces(s)).toEqual([[[0, 0], [1, 0]], [[1, 0], [2, 0], [3, 0]], [[3, 0], [4, 0]]]);
  });

  it("toggles a mark by a click on a point, on its corner, or on a point the corner replaced", () => {
    const profile = corner(90);
    const { pins, points } = chainOn(profile, 6);
    const k = points.reduce((b, p, i) => (dist(p, [0, 0]) < dist(points[b]!, [0, 0]) ? i : b), 0);
    let marks = toggleMark(points, shape(points, pins, R, []), [], k);
    expect(marks).toEqual([points[k]]);
    const s = shape(points, pins, R, marks);
    expect(s.zones).toHaveLength(1);
    const { from, to } = s.zones[0]!;
    expect(to).toBeGreaterThan(from);
    for (const i of [k, from, to]) expect(toggleMark(points, s, marks, i)).toEqual([]);
    // The ends cannot be corners.
    marks = toggleMark(points, shape(points, pins, R, []), [], 0);
    expect(marks).toEqual([]);
  });

  it("finds a mark's point within reach only", () => {
    const points = line(3);
    expect(nearestPoint(points, [13.5, 2])).toBe(1);
    expect(nearestPoint(points, [2, 2])).toBe(0);
    expect(nearestPoint(points, [6.5, 0])).toBe(-1);
  });
});
