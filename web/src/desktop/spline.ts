// The smooth curve drawn through the points: a port of splinewire/contact.py
// spline_samples, which is SciPy's CubicSpline with not-a-knot ends over the
// chord length. Drawing only: the files and the clipboard carry the points,
// and Fusion fits its own spline through them. spline.test.ts pins this to
// the samples Python wrote into the measurement fixture.

import type { Point } from "../relay/messages";

/** Dense samples of the spline through `points`, `perSegment` per gap between
 * points, both ends included. Fewer than 3 points: the points themselves, as Python. */
export function splineSamples(points: readonly Point[], perSegment = 12): Point[] {
  const n = points.length;
  if (n < 3) return points.map(([x, y]) => [x, y]);
  const s = [0];
  for (let i = 1; i < n; i++) {
    const [ax, ay] = points[i - 1]!;
    const [bx, by] = points[i]!;
    s.push(s[i - 1]! + Math.hypot(bx - ax, by - ay));
  }
  // Two points at the same place give a zero-length gap, which the spline
  // cannot cross (SciPy refuses it). Never seen from a real chain; draw straight.
  for (let i = 1; i < n; i++) if (!(s[i]! > s[i - 1]!)) return points.map(([x, y]) => [x, y]);
  const fx = cubic(s, points.map((p) => p[0]));
  const fy = cubic(s, points.map((p) => p[1]));
  const count = perSegment * (n - 1) + 1;
  const end = s[n - 1]!;
  const step = end / (count - 1);
  const out: Point[] = [];
  for (let k = 0; k < count; k++) {
    const t = k === count - 1 ? end : k * step;    // numpy's linspace ends exactly on the stop
    out.push([fx(t), fy(t)]);
  }
  return out;
}

/** SciPy's CubicSpline(x, y, bc_type="not-a-knot") for one coordinate. */
function cubic(x: readonly number[], y: readonly number[]): (t: number) => number {
  const n = x.length;
  const dx: number[] = [];
  const slope: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(x[i + 1]! - x[i]!);
    slope.push((y[i + 1]! - y[i]!) / dx[i]!);
  }
  const d = slopes(x, dx, slope);
  return (t: number) => {
    // The interval holding t: x[i] <= t < x[i+1], the last one for the end itself.
    let lo = 0;
    let hi = n - 2;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (x[mid]! <= t) lo = mid;
      else hi = mid - 1;
    }
    const i = lo;
    const h = dx[i]!;
    const c = (d[i]! + d[i + 1]! - 2 * slope[i]!) / h;
    const c3 = c / h;
    const c2 = (slope[i]! - d[i]!) / h - c;
    const u = t - x[i]!;
    return ((c3 * u + c2) * u + d[i]!) * u + y[i]!;
  };
}

/** The first derivative at each point, from the system SciPy solves. */
function slopes(x: readonly number[], dx: readonly number[], slope: readonly number[]): number[] {
  const n = x.length;
  if (n === 3) {
    // Not-a-knot at both ends of three points is the parabola through them;
    // SciPy solves this 3×3 system for it.
    const a = [
      [1, 1, 0],
      [dx[1]!, 2 * (dx[0]! + dx[1]!), dx[0]!],
      [0, 1, 1],
    ];
    const b = [2 * slope[0]!, 3 * (dx[0]! * slope[1]! + dx[1]! * slope[0]!), 2 * slope[1]!];
    return solve3(a, b);
  }
  // Tridiagonal: row i has lower[i] at column i-1, diag[i] at i, upper[i] at i+1.
  const lower = new Array<number>(n).fill(0);
  const diag = new Array<number>(n).fill(0);
  const upper = new Array<number>(n).fill(0);
  const b = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    lower[i] = dx[i]!;
    diag[i] = 2 * (dx[i - 1]! + dx[i]!);
    upper[i] = dx[i - 1]!;
    b[i] = 3 * (dx[i]! * slope[i - 1]! + dx[i - 1]! * slope[i]!);
  }
  const d0 = x[2]! - x[0]!;
  diag[0] = dx[1]!;
  upper[0] = d0;
  b[0] = ((dx[0]! + 2 * d0) * dx[1]! * slope[0]! + dx[0]! ** 2 * slope[1]!) / d0;
  const m = n - 1;
  const dn = x[m]! - x[m - 2]!;
  diag[m] = dx[m - 2]!;
  lower[m] = dn;
  b[m] = (dx[m - 1]! ** 2 * slope[m - 2]! + (2 * dn + dx[m - 1]!) * dx[m - 2]! * slope[m - 1]!) / dn;
  // Thomas algorithm. The rows are diagonally dominant except the two end
  // rows, whose pivots stay well away from zero for chord-length parameters.
  for (let i = 1; i < n; i++) {
    const w = lower[i]! / diag[i - 1]!;
    diag[i] = diag[i]! - w * upper[i - 1]!;
    b[i] = b[i]! - w * b[i - 1]!;
  }
  const out = new Array<number>(n).fill(0);
  out[m] = b[m]! / diag[m]!;
  for (let i = m - 1; i >= 0; i--) out[i] = (b[i]! - upper[i]! * out[i + 1]!) / diag[i]!;
  return out;
}

function solve3(a: number[][], b: number[]): number[] {
  // Gaussian elimination with partial pivoting.
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < 3; col++) {
    let p = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r]![col]!) > Math.abs(m[p]![col]!)) p = r;
    [m[col], m[p]] = [m[p]!, m[col]!];
    for (let r = col + 1; r < 3; r++) {
      const f = m[r]![col]! / m[col]![col]!;
      for (let c = col; c < 4; c++) m[r]![c] = m[r]![c]! - f * m[col]![c]!;
    }
  }
  const out = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let v = m[r]![3]!;
    for (let c = r + 1; c < 3; c++) v -= m[r]![c]! * out[c]!;
    out[r] = v / m[r]![r]!;
  }
  return out;
}
