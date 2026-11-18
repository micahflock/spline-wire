// output.dxf_text: the curve as DXF in mm (layer CURVE: one spline through
// the points; layer CURVE_POINTS: the points), the file ezdxf writes. The
// file around the entities is ezdxf's own (dxf-template.ts, made from its
// output by web/scripts/golden_dxf.py), so Fusion gets exactly what it has
// been importing; the spline is computed as ezdxf's
// add_cad_spline_control_frame computes it: a cubic B-spline interpolating
// the points, knots at their chord-length parameters, zero second derivative
// at both ends (ezdxf.math.cad_fit_point_interpolation).

import { DXF_TEMPLATE, EZDXF_VERSION } from "./dxf-template";
import type { Vec2 } from "./numeric";
import { floatRepr } from "./pyjson";

const FIRST_HANDLE = 0x31;      // the SPLINE's; the POINTs follow it

export interface DxfStamp {
  /** When the file is written: the dates and time stamps in it. */
  now: Date;
  /** Two GUIDs, as {XXXXXXXX-...}; random if not given. */
  guids?: [string, string];
}

export function dxfText(points: readonly Vec2[], stamp: DxfStamp = { now: new Date() }): string {
  const n = points.length;
  const { knots, control } = cadSpline(points);
  const g = (code: number) => String(code).padStart(3, " ");
  const lines: string[] = [];
  const add = (code: number, value: string) => lines.push(g(code), value);
  add(0, "SPLINE");
  add(5, hex(FIRST_HANDLE));
  add(330, "17");
  add(100, "AcDbEntity");
  add(8, "CURVE");
  add(100, "AcDbSpline");
  add(70, "0");
  add(71, "3");
  add(72, String(knots.length));
  add(73, String(control.length));
  add(74, "0");
  for (const k of knots) add(40, floatRepr(k));
  for (const [x, y] of control) {
    add(10, floatRepr(x));
    add(20, floatRepr(y));
    add(30, "0.0");
  }
  points.forEach(([x, y], k) => {
    add(0, "POINT");
    add(5, hex(FIRST_HANDLE + 1 + k));
    add(330, "17");
    add(100, "AcDbEntity");
    add(8, "CURVE_POINTS");
    add(100, "AcDbPoint");
    add(10, floatRepr(x));
    add(20, floatRepr(y));
    add(30, "0.0");
  });
  const jd = floatRepr(julianDate(stamp.now));
  const [fp, ver] = stamp.guids ?? [guid(), guid()];
  const values: Record<string, string> = {
    ENTITIES: lines.join("\n"),
    TDCREATE: jd, TDUPDATE: jd, FINGERPRINTGUID: fp, VERSIONGUID: ver,
    H0: hex(FIRST_HANDLE + n + 1), H1: hex(FIRST_HANDLE + n + 2), H2: hex(FIRST_HANDLE + n + 3),
    HANDSEED: hex(FIRST_HANDLE + n + 4),
    MARKER: `${EZDXF_VERSION} @ ${isoUtc(stamp.now)}`,
  };
  return DXF_TEMPLATE.replace(/\{\{([A-Z0-9]+)\}\}/g, (_, key: string) => {
    if (!(key in values)) throw new Error(`the DXF template has an unknown field ${key}`);
    return values[key]!;
  });
}

function hex(h: number): string {
  return h.toString(16).toUpperCase();
}

function guid(): string {
  return `{${crypto.randomUUID().toUpperCase()}}`;
}

/** ezdxf.tools.juliandate of a local date and time (to the second). */
export function julianDate(d: Date): number {
  const y = d.getFullYear() + (d.getMonth() + 1 - 2.85) / 12.0;
  const A = Math.floor(367.0 * y) - 1.75 * Math.floor(y) + d.getDate();
  const B = Math.floor(A) - 0.75 * Math.floor(y / 100.0);
  const day = (d.getHours() * 3600.0 + d.getMinutes() * 60.0 + d.getSeconds()) / 86400.0;
  return Math.floor(B) + 1721115.0 + day;
}

/** datetime.now(timezone.utc).isoformat(): microseconds when not zero, "+00:00". */
function isoUtc(d: Date): string {
  const base = d.toISOString().slice(0, 19);
  const us = d.getUTCMilliseconds() * 1000;
  return `${base}${us ? "." + String(us).padStart(6, "0") : ""}+00:00`;
}

/** The control points and knots ezdxf makes for a cubic spline through the points. */
export function cadSpline(points: readonly Vec2[]): { knots: number[]; control: Vec2[] } {
  const n = points.length - 1;
  const p = 3;
  // Chord-length parameters.
  const d = points.slice(1).map((q, i) => Math.hypot(q[0] - points[i]![0], q[1] - points[i]![1]));
  const total = d.reduce((a, b) => a + b, 0);
  const t = [0];
  let s = 0;
  for (const v of d.slice(0, -1)) {
    s += v;
    t.push(s / total);
  }
  t.push(1);
  // Natural knots, constrained: the interior parameters.
  const knots = [0, 0, 0, 0, ...t.slice(1, n), 1, 1, 1, 1];
  const count = n + 3;
  const rows = t.map((u) => basisVector(knots, count, u));
  // Second derivative zero at both ends (Piegl & Tiller 3.9, 3.10).
  const up1 = knots[p + 1]!, up2 = knots[p + 2]!;
  const f1 = (p * (p - 1)) / up1;
  const c1 = [f1 / up1, (-f1 * (up1 + up2)) / (up1 * up2), f1 / up2, ...new Array<number>(n).fill(0)];
  const m = knots.length - 1;
  const ump1 = knots[m - p - 1]!, ump2 = knots[m - p - 2]!;
  const f2 = (p * (p - 1)) / (1.0 - ump1);
  const c2 = [...new Array<number>(n).fill(0), f2 / (1.0 - ump2), (-f2 * (2.0 - ump1 - ump2)) / (1.0 - ump1) / (1.0 - ump2), f2 / (1.0 - ump1)];
  rows.splice(1, 0, c1);
  rows.splice(rows.length - 1, 0, c2);
  const rhs: Vec2[] = points.map((q) => [q[0], q[1]]);
  rhs.splice(1, 0, [0, 0]);
  rhs.splice(rhs.length - 1, 0, [0, 0]);
  return { knots, control: solve(rows, rhs) };
}

/** The values of the count basis functions at u (Cox–de Boor, Piegl & Tiller A2.2). */
function basisVector(knots: readonly number[], count: number, u: number): number[] {
  const p = 3;
  // The span: knots[span] <= u < knots[span + 1], the last one for u = 1.
  let span = count - 1;
  if (u < knots[count]!) {
    let lo = p, hi = count;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (u < knots[mid]!) hi = mid;
      else lo = mid;
    }
    span = lo;
  }
  const N = [1, 0, 0, 0];
  const left = [0, 0, 0, 0], right = [0, 0, 0, 0];
  for (let j = 1; j <= p; j++) {
    left[j] = u - knots[span + 1 - j]!;
    right[j] = knots[span + j]! - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = N[r]! / (right[r + 1]! + left[j - r]!);
      N[r] = saved + right[r + 1]! * temp;
      saved = left[j - r]! * temp;
    }
    N[j] = saved;
  }
  const row = new Array<number>(count).fill(0);
  for (let k = 0; k <= p; k++) row[span - p + k] = N[k]!;
  return row;
}

/** Solve A X = B for two right-hand columns (Gaussian elimination, partial pivoting). */
function solve(A: number[][], B: Vec2[]): Vec2[] {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const b = B.map((r) => [r[0], r[1]]);
  for (let k = 0; k < n; k++) {
    let piv = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(a[i]![k]!) > Math.abs(a[piv]![k]!)) piv = i;
    [a[k], a[piv]] = [a[piv]!, a[k]!];
    [b[k], b[piv]] = [b[piv]!, b[k]!];
    for (let i = k + 1; i < n; i++) {
      const f = a[i]![k]! / a[k]![k]!;
      if (f === 0) continue;
      for (let j = k; j < n; j++) a[i]![j] = a[i]![j]! - f * a[k]![j]!;
      b[i]![0] = b[i]![0]! - f * b[k]![0]!;
      b[i]![1] = b[i]![1]! - f * b[k]![1]!;
    }
  }
  const x: Vec2[] = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sx = b[i]![0]!, sy = b[i]![1]!;
    for (let j = i + 1; j < n; j++) {
      sx -= a[i]![j]! * x[j]![0];
      sy -= a[i]![j]! * x[j]![1];
    }
    x[i] = [sx / a[i]![i]!, sy / a[i]![i]!];
  }
  return x;
}
