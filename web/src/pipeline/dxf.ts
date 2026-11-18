// splinewire/dxf.py: the curve as a small DXF of our own, byte for byte as
// Python writes it (dxf.test.ts, session.test.ts): one cubic spline through
// the curve points on layer CURVE, the points on CURVE_POINTS, in mm. DXF
// R2000 with what CAD programs expect of one and nothing more, every handle
// fixed and nothing depending on the time (decision D19). The spline passes
// through every point: global cubic interpolation, chord-length knots, zero
// second derivative at both ends, as CAD programs build one from fit points.

import { fixed, type Vec2 } from "./numeric";

const ENTITY_HANDLE = 0x100;
const H = {
  vport_table: "8", ltype_table: "5", layer_table: "2", style_table: "3", view_table: "6", ucs_table: "7",
  appid_table: "9", dimstyle_table: "A", block_record_table: "1",
  ltype_byblock: "14", ltype_bylayer: "15", ltype_continuous: "16",
  layer_0: "10", layer_curve: "11", layer_points: "12",
  style_standard: "13", appid_acad: "17", dimstyle_standard: "18",
  model_record: "1F", paper_record: "1B",
  model_block: "20", model_endblk: "21", paper_block: "1C", paper_endblk: "1D",
  root_dict: "C", group_dict: "D",
};

type Tag = [number, string];

/** The DXF file's text for a curve through the points (mm). */
export function dxfText(points: readonly Vec2[]): string {
  const tags: Tag[] = [];
  const add = (code: number, value: string) => tags.push([code, value]);
  const section = (name: string) => {
    add(0, "SECTION");
    add(2, name);
  };
  const table = (name: string, handle: string, count: number) => {
    add(0, "TABLE");
    add(2, name);
    add(5, handle);
    add(330, "0");
    add(100, "AcDbSymbolTable");
    add(70, String(count));
  };
  const record = (kind: string, handle: string, owner: string, subclass: string) => {
    add(0, kind);
    add(5, handle);
    add(330, owner);
    add(100, "AcDbSymbolTableRecord");
    add(100, subclass);
  };
  const many = (list: Tag[]) => list.forEach(([c, v]) => add(c, v));

  const entityCount = points.length >= 2 ? 1 + points.length : points.length;
  section("HEADER");
  for (const [name, code, value] of [["$ACADVER", 1, "AC1015"], ["$HANDSEED", 5, hex(ENTITY_HANDLE + entityCount)],
    ["$INSUNITS", 70, "4"], ["$MEASUREMENT", 70, "1"]] as const) {
    add(9, name);
    add(code, value);
  }
  add(0, "ENDSEC");
  section("CLASSES");
  add(0, "ENDSEC");

  section("TABLES");
  table("VPORT", H.vport_table, 0);
  add(0, "ENDTAB");
  table("LTYPE", H.ltype_table, 3);
  for (const [name, handle, text] of [["ByBlock", H.ltype_byblock, ""], ["ByLayer", H.ltype_bylayer, ""],
    ["Continuous", H.ltype_continuous, "Solid line"]] as const) {
    record("LTYPE", handle, H.ltype_table, "AcDbLinetypeTableRecord");
    many([[2, name], [70, "0"], [3, text], [72, "65"], [73, "0"], [40, "0.0"]]);
  }
  add(0, "ENDTAB");
  table("LAYER", H.layer_table, 3);
  for (const [name, handle, color] of [["0", H.layer_0, "7"], ["CURVE", H.layer_curve, "7"],
    ["CURVE_POINTS", H.layer_points, "1"]] as const) {
    record("LAYER", handle, H.layer_table, "AcDbLayerTableRecord");
    many([[2, name], [70, "0"], [62, color], [6, "Continuous"], [370, "-3"]]);
  }
  add(0, "ENDTAB");
  table("STYLE", H.style_table, 1);
  record("STYLE", H.style_standard, H.style_table, "AcDbTextStyleTableRecord");
  many([[2, "Standard"], [70, "0"], [40, "0.0"], [41, "1.0"], [50, "0.0"], [71, "0"], [42, "2.5"], [3, "txt"], [4, ""]]);
  add(0, "ENDTAB");
  for (const [name, handle] of [["VIEW", H.view_table], ["UCS", H.ucs_table]] as const) {
    table(name, handle, 0);
    add(0, "ENDTAB");
  }
  table("APPID", H.appid_table, 1);
  record("APPID", H.appid_acad, H.appid_table, "AcDbRegAppTableRecord");
  many([[2, "ACAD"], [70, "0"], [0, "ENDTAB"]]);
  table("DIMSTYLE", H.dimstyle_table, 1);
  many([[100, "AcDbDimStyleTable"], [0, "DIMSTYLE"], [105, H.dimstyle_standard], [330, H.dimstyle_table],
    [100, "AcDbSymbolTableRecord"], [100, "AcDbDimStyleTableRecord"], [2, "Standard"], [70, "0"], [0, "ENDTAB"]]);
  table("BLOCK_RECORD", H.block_record_table, 2);
  for (const [name, handle] of [["*Model_Space", H.model_record], ["*Paper_Space", H.paper_record]] as const) {
    record("BLOCK_RECORD", handle, H.block_record_table, "AcDbBlockTableRecord");
    add(2, name);
  }
  add(0, "ENDTAB");
  add(0, "ENDSEC");

  section("BLOCKS");
  for (const [name, rec, block, endblk] of [["*Model_Space", H.model_record, H.model_block, H.model_endblk],
    ["*Paper_Space", H.paper_record, H.paper_block, H.paper_endblk]] as const) {
    many([[0, "BLOCK"], [5, block], [330, rec], [100, "AcDbEntity"], [8, "0"], [100, "AcDbBlockBegin"], [2, name],
      [70, "0"], [10, "0.0"], [20, "0.0"], [30, "0.0"], [3, name], [1, ""],
      [0, "ENDBLK"], [5, endblk], [330, rec], [100, "AcDbEntity"], [8, "0"], [100, "AcDbBlockEnd"]]);
  }
  add(0, "ENDSEC");

  section("ENTITIES");
  let handle = ENTITY_HANDLE;
  if (points.length >= 2) {
    const { knots, control } = cadSpline(points);
    many([[0, "SPLINE"], [5, hex(handle)], [330, H.model_record], [100, "AcDbEntity"], [8, "CURVE"], [100, "AcDbSpline"],
      [210, "0.0"], [220, "0.0"], [230, "1.0"], [70, "8"], [71, "3"], [72, String(knots.length)],
      [73, String(control.length)], [74, "0"]]);
    for (const k of knots) add(40, num(k, 9));
    for (const [x, y] of control) many([[10, num(x, 6)], [20, num(y, 6)], [30, "0.0"]]);
    handle++;
  }
  for (const [x, y] of points) {
    many([[0, "POINT"], [5, hex(handle)], [330, H.model_record], [100, "AcDbEntity"], [8, "CURVE_POINTS"],
      [100, "AcDbPoint"], [10, num(x, 6)], [20, num(y, 6)], [30, "0.0"]]);
    handle++;
  }
  add(0, "ENDSEC");

  section("OBJECTS");
  many([[0, "DICTIONARY"], [5, H.root_dict], [330, "0"], [100, "AcDbDictionary"], [281, "1"], [3, "ACAD_GROUP"],
    [350, H.group_dict], [0, "DICTIONARY"], [5, H.group_dict], [330, H.root_dict], [100, "AcDbDictionary"], [281, "1"]]);
  add(0, "ENDSEC");
  add(0, "EOF");
  return tags.map(([c, v]) => `${String(c).padStart(3, " ")}\n${v}\n`).join("");
}

function hex(h: number): string {
  return h.toString(16).toUpperCase();
}

/** f"{x:.{digits}f}", with no "-0.000000". */
function num(x: number, digits: number): string {
  const s = fixed(x, digits);
  return s.startsWith("-") && /^-[0.]*$/.test(s) ? s.slice(1) : s;
}

/** Knots and control points of the cubic B-spline through the points (dxf.cad_spline). */
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
