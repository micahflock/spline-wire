// Hand edits against Python (golden/edits.json.gz, web/scripts/golden_edits.py):
// pins removed and added by clicks, snapped to the dot under the click or,
// under glare, placed by the click and fitted to the pitch. Same pins in the
// same order, same added-pin bookkeeping, warnings and errors word for word,
// geometry within 1e-4 mm. Detection on the same photos is checked too.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { detectFiducials } from "./detect";
import { editsFromJson } from "./edits";
import { PipelineError } from "./errors";
import { readGrayPng } from "./golden/png";
import { measureDetections } from "./measure";
import type { Vec2 } from "./numeric";
import type { ChainSpec, Fiducial } from "./types";

interface Case {
  photo: string; chain: ChainSpec; focal_px: number; size: [number, number]; detected: number[][]; detected_is_full: boolean;
  runs: { name: string; edits: { add: number[][]; remove: number[][] }; error?: string;
    result?: { fiducials: number[][]; indices: number[]; gaps: number[]; manual: [number, number, boolean][];
      pins_mm: Vec2[]; contacts_mm: Vec2[]; warnings: string[]; notes: string[] } }[];
}

const cases = JSON.parse(
  gunzipSync(readFileSync(new URL("./golden/edits.json.gz", import.meta.url))).toString("utf-8"),
) as Case[];

function fiducial(r: number[]): Fiducial {
  const [cx, cy, ma, mi, ia, ib, angle, residual, contrast, polarity, surround] = r as number[] & { length: 11 };
  return {
    center_px: [cx!, cy!], outer_axes_px: [ma!, mi!], inner_axes_px: [ia!, ib!], angle_deg: angle!,
    residual_px: residual!, contrast: contrast!, polarity: polarity!, surround: surround!,
  };
}

function worst(a: readonly Vec2[], b: readonly Vec2[]): number {
  expect(a.length).toBe(b.length);
  return Math.max(0, ...a.map((p, i) => Math.hypot(p[0] - b[i]![0], p[1] - b[i]![1])));
}

describe("hand edits against Python", () => {
  const images = new Map<string, ReturnType<typeof readGrayPng>>();
  const photo = (name: string) => {
    if (!images.has(name)) images.set(name, readGrayPng(new URL(`./golden/${name}`, import.meta.url)));
    return images.get(name)!;
  };
  let worstMm = 0;

  for (const c of cases) {
    if (c.detected_is_full) {
      it(`${c.photo}: detection matches`, () => {
        const got = detectFiducials(photo(c.photo), c.chain);
        expect(got.length).toBe(c.detected.length);
        got.forEach((f, i) => {
          const w = fiducial(c.detected[i]!);
          expect(Math.abs(f.center_px[0] - w.center_px[0]) + Math.abs(f.center_px[1] - w.center_px[1])).toBeLessThan(1e-6);
        });
      }, 60000);
    }
    for (const run of c.runs) {
      it(`${c.photo}: ${run.name}`, () => {
        const image = photo(c.photo);
        // As Python's measure got them: not rounded as editsFromJson rounds the page's clicks.
        const pair = (p: number[]): Vec2 => [p[0]!, p[1]!];
        const edits = { add_px: run.edits.add.map(pair), remove_px: run.edits.remove.map(pair) };
        let m;
        try {
          m = measureDetections(c.detected.map(fiducial), c.size, c.chain, c.focal_px, "inside", edits, image);
        } catch (err) {
          if (!(err instanceof PipelineError)) throw err;
          expect(err.message).toBe(run.error);
          return;
        }
        expect(run.error).toBeUndefined();
        const want = run.result!;
        expect(m.fiducials.length).toBe(want.fiducials.length);
        m.fiducials.forEach((f, i) => {
          const w = fiducial(want.fiducials[i]!);
          expect(Math.hypot(f.center_px[0] - w.center_px[0], f.center_px[1] - w.center_px[1])).toBeLessThan(1e-6);
          expect(f.outer_axes_px[0]).toBeCloseTo(w.outer_axes_px[0], 6);
        });
        expect(m.order.indices).toEqual(want.indices);
        expect(m.order.gaps).toEqual(want.gaps);
        expect(m.manual.map((mp) => [mp.fiducial, mp.edit, mp.snapped])).toEqual(want.manual);
        expect(m.warnings).toEqual(want.warnings);
        expect(m.notes).toEqual(want.notes);
        const d = Math.max(worst(m.rectification.pins_mm, want.pins_mm), worst(m.contacts_mm, want.contacts_mm));
        worstMm = Math.max(worstMm, d);
        expect(d).toBeLessThan(1e-4);
      }, 60000);
    }
  }

  it("reports", () => {
    console.log(`[edits] worst pin or contact difference ${worstMm.toExponential(1)} mm`);
  });
});

describe("edits from the page's JSON", () => {
  it("are checked as Python checks them", () => {
    expect(editsFromJson({ add: [[1.234, 5.675]], remove: [] })).toEqual({ add_px: [[1.23, 5.67]], remove_px: [] });
    expect(() => editsFromJson([])).toThrow("edits must be an object with add and remove lists");
    expect(() => editsFromJson({ add: [[1]] })).toThrow("add: each point is [x, y] in pixels");
    expect(() => editsFromJson({ remove: [[1, 2]] }, [0, 0])).toThrow("remove: a point lies outside the photo");
    expect(() => editsFromJson({ add: new Array(301).fill([1, 1]) })).toThrow("at most 300 points");
  });
});
