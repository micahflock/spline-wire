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
