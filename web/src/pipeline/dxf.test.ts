// dxf.ts against splinewire/dxf.py (golden/dxf.json.gz, web/scripts/golden_dxf.py):
// the same file, byte for byte.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { dxfText } from "./dxf";
import type { Vec2 } from "./numeric";

const golden = JSON.parse(gunzipSync(readFileSync(new URL("./golden/dxf.json.gz", import.meta.url))).toString("utf-8")) as
  { curves: { points: Vec2[]; dxf: string }[] };

describe("the DXF, as Python writes it", () => {
  golden.curves.forEach((c, n) => {
    it(`curve ${n} (${c.points.length} points)`, () => {
      expect(dxfText(c.points)).toBe(c.dxf);
    });
  });
});

describe("the DXF with sharp corners", () => {
  const pts: Vec2[] = [[0, 0], [10, 1], [20, 0], [20, 10], [30, 12], [40, 10], [50, 10]];

  it("is Python's file without corners", () => {
    expect(dxfText(pts, [])).toBe(dxfText(pts));
  });

  it("has one spline per piece, joined at the corners, and every point", () => {
    const text = dxfText(pts, [2, 5]);
    expect(text.match(/\nSPLINE\n/g)).toHaveLength(3);
    expect(text.match(/\nPOINT\n/g)).toHaveLength(pts.length);
    // The pieces' control points start and end on the corners.
    const controls = text.split("\nSPLINE\n").slice(1).map((e) => {
      const xs = [...e.matchAll(/ 10\n([-\d.]+)\n 20\n([-\d.]+)\n/g)].map((m) => [Number(m[1]), Number(m[2])]);
      return xs.slice(0, xs.length - (e.includes("POINT") ? (e.match(/\nPOINT\n/g)?.length ?? 0) : 0));
    });
    expect(controls[0]![0]).toEqual([0, 0]);
    expect(controls[0]!.at(-1)).toEqual([20, 0]);
    expect(controls[1]![0]).toEqual([20, 0]);
    expect(controls[2]!.at(-1)).toEqual([50, 10]);
  });

  it("takes a piece of two points (a straight wall)", () => {
    expect(dxfText([[0, 0], [10, 0], [10, 10], [20, 12]], [1, 2]).match(/\nSPLINE\n/g)).toHaveLength(3);
  });
});
