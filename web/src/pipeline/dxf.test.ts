// dxf.ts against output.dxf_text (golden/dxf.json.gz, web/scripts/golden_dxf.py):
// the same file line for line, but for the dates, GUIDs and time stamps
// (which differ every time ezdxf writes one too) and the last digits of the
// spline's numbers.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { dxfText, julianDate } from "./dxf";
import type { Vec2 } from "./numeric";

const golden = JSON.parse(gunzipSync(readFileSync(new URL("./golden/dxf.json.gz", import.meta.url))).toString("utf-8")) as
  { ezdxf: string; curves: { points: Vec2[]; dxf: string }[] };

/** The CLASSES section with its CLASS records sorted: ezdxf orders them by
 * Python's hash seed, so two of them swap places from run to run. */
function sortedClasses(lines: string[]): string[] {
  const start = lines.indexOf("CLASSES"), end = lines.indexOf("ENDSEC", start);
  const records: string[][] = [];
  for (let i = start + 1; i < end - 1; i++) {
    if (lines[i] === "  0" && lines[i + 1] === "CLASS") records.push([]);
    records[records.length - 1]?.push(lines[i]!);
  }
  const sorted = records.map((r) => r.join("\n")).sort().join("\n").split("\n");
  return [...lines.slice(0, start + 1), ...sorted, ...lines.slice(end - 1)];
}

/** The lines that differ every time ezdxf writes a file. */
function varying(lines: string[]): Set<number> {
  const out = new Set<number>();
  lines.forEach((l, i) => {
    if (["$TDCREATE", "$TDUPDATE", "$FINGERPRINTGUID", "$VERSIONGUID"].includes(lines[i - 2] ?? "")) out.add(i);
    if (/^\d+(\.\d+)+ @ /.test(l)) out.add(i);
  });
  return out;
}

describe("DXF as ezdxf writes it", () => {
  golden.curves.forEach((c, n) => {
    it(`curve ${n} (${c.points.length} points)`, () => {
      const got = sortedClasses(dxfText(c.points).split("\n"));
      const want = sortedClasses(c.dxf.split("\n"));
      expect(got.length).toBe(want.length);
      const skip = varying(want);
      let worst = 0;
      want.forEach((w, i) => {
        if (skip.has(i)) return;
        if (got[i] === w) return;
        // A number: the spline's control points, computed in another order.
        expect(/^-?\d+\.\d+(e[-+]\d+)?$/.test(w), `line ${i}: ${got[i]} against ${w}`).toBe(true);
        const d = Math.abs(Number(got[i]) - Number(w));
        worst = Math.max(worst, d);
        expect(d, `line ${i}: ${got[i]} against ${w}`).toBeLessThan(1e-9 * Math.max(1, Math.abs(Number(w))));
      });
    });
  });

  it("dates as ezdxf's juliandate, GUIDs and time stamps in its form", () => {
    // ezdxf: juliandate(datetime(2026, 11, 18, 3, 31, 3)) == 2461363.1465625
    expect(julianDate(new Date(2026, 10, 18, 3, 31, 3, 500))).toBe(2461363.1465625);
    const text = dxfText([[0, 0], [10, 2], [20, 0]], { now: new Date(Date.UTC(2026, 0, 2, 3, 4, 5, 6)), guids: ["{A}", "{B}"] });
    expect(text).toContain(`${golden.ezdxf} @ 2026-01-02T03:04:05.006000+00:00`);
    expect(text).toContain("$FINGERPRINTGUID\n  2\n{A}\n");
    expect(text).toContain("$VERSIONGUID\n  2\n{B}\n");
    expect(text.endsWith("  0\nEOF\n")).toBe(true);
  });
});
