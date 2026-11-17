// The TypeScript pipeline against Python's, on the cv_benchmark scenes:
// golden/scenes.json.gz holds Python's detections and measurements
// (web/scripts/golden.py). From the same detections the port must give the
// same chain (pins, links, gaps, rejected, exactly), the same warnings,
// notes and errors (word for word), and the same geometry to 1e-4 mm. The
// geometry cannot match more closely than the solver stops: SciPy's lm ends
// within xtol (1e-8) of the minimum, and where the minimum is flat (steep
// tilt, focal length estimated) the 25 starts stop up to ~1e-6 apart with
// costs equal to 1e-13, so the start that wins is chosen by rounding, which
// differs between V8, glibc and Pyodide. The largest differences are printed.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { afterAll, describe, expect, it } from "vitest";
import { PipelineError } from "./errors";
import { measureDetections } from "./measure";
import { median, type Vec2 } from "./numeric";
import { orderChain } from "./order";
import type { ChainSpec, Fiducial, Side } from "./types";

interface Run {
  name: string;
  spec: ChainSpec;
  focal_px: number | null;
  side: Side | Vec2;
  error?: string;
  result?: {
    indices: number[]; links: [number, number][]; gaps: number[]; rejected: number[];
    pins_mm: Vec2[]; focal_px: number; tilt_deg: number; residual_rms_mm: number; residual_max_mm: number;
    depth_mm: number[]; object_side: Side; object_sign: number; contacts_mm: Vec2[];
    warnings: string[]; notes: string[];
  };
}

interface Scene {
  id: string;
  size: [number, number];
  focal_px: number;
  fiducials: number[][];
  plain_order?: { indices: number[]; links: [number, number][]; gaps: number[] };
  runs: Run[];
}

const golden = JSON.parse(
  gunzipSync(readFileSync(new URL("./golden/scenes.json.gz", import.meta.url))).toString("utf-8"),
) as { scenes: Scene[] };

function fiducials(rows: number[][]): Fiducial[] {
  return rows.map(([cx, cy, major, minor, contrast, polarity, surround]) => ({
    center_px: [cx!, cy!], outer_axes_px: [major!, minor!], inner_axes_px: [0, 0], angle_deg: 0,
    residual_px: 0, contrast: contrast!, polarity: polarity!, surround: surround!,
  }));
}

const TOL_MM = 1e-4;
const worst = { known: 0, estimated: 0, focalRel: 0 };
let compared = 0, errors = 0;
const times: number[] = [];

function maxDiff(a: readonly Vec2[], b: readonly Vec2[]): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  a.forEach((p, i) => {
    worst = Math.max(worst, Math.abs(p[0] - b[i]![0]), Math.abs(p[1] - b[i]![1]));
  });
  return worst;
}

describe("pipeline after detection, against Python", () => {
  afterAll(() => {
    console.log(`[golden] ${compared} measurements and ${errors} errors matched; worst pin or contact `
      + `difference ${worst.known.toExponential(1)} mm (focal known), ${worst.estimated.toExponential(1)} mm `
      + `(estimated), focal ${worst.focalRel.toExponential(1)} relative; time per measurement median `
      + `${median(times).toFixed(1)} ms, max ${Math.max(...times).toFixed(0)} ms`);
  });

  it("has the scenes", () => {
    expect(golden.scenes.length).toBeGreaterThanOrEqual(30);
  });

  for (const scene of golden.scenes) {
    const fids = fiducials(scene.fiducials);
    for (const run of scene.runs) {
      it(`${scene.id} ${run.name}`, () => {
        let got;
        try {
          const t0 = performance.now();
          got = measureDetections(fids, scene.size, run.spec, run.focal_px, run.side);
          times.push(performance.now() - t0);
        } catch (err) {
          if (!(err instanceof PipelineError)) throw err;
          expect(err.message).toBe(run.error);
          errors++;
          return;
        }
        expect(run.error).toBeUndefined();
        const want = run.result!;
        expect(got.order.indices).toEqual(want.indices);
        expect(got.order.links).toEqual(want.links);
        expect(got.order.gaps).toEqual(want.gaps);
        expect(got.order.rejected).toEqual(want.rejected);
        expect(got.warnings).toEqual(want.warnings);
        expect(got.notes).toEqual(want.notes);
        expect(got.object_side).toBe(want.object_side);
        expect(got.object_sign).toBe(want.object_sign);
        const r = got.rectification;
        const kind = run.focal_px === null ? "estimated" : "known";
        const diff = Math.max(maxDiff(r.pins_mm, want.pins_mm), maxDiff(got.contacts_mm, want.contacts_mm),
          Math.abs(r.residual_max_mm - want.residual_max_mm), Math.abs(r.residual_rms_mm - want.residual_rms_mm));
        worst[kind] = Math.max(worst[kind], diff);
        expect(diff).toBeLessThan(TOL_MM);
        const focalRel = Math.abs(r.focal_px - want.focal_px) / want.focal_px;
        worst.focalRel = Math.max(worst.focalRel, focalRel);
        expect(focalRel).toBeLessThan(kind === "known" ? 1e-15 : 1e-4);
        expect(Math.abs(r.tilt_deg - want.tilt_deg)).toBeLessThan(1e-3);
        r.depth_mm.forEach((d, i) => expect(Math.abs(d - want.depth_mm[i]!) / d).toBeLessThan(1e-4));
        compared++;
      });
    }
    if (scene.plain_order) {
      const want = scene.plain_order;
      it(`${scene.id} order without sizes`, () => {
        const o = orderChain(fids.map((f) => f.center_px), { maxTurnDeg: 80 });
        expect(o.indices).toEqual(want.indices);
        expect(o.links).toEqual(want.links);
        expect(o.gaps).toEqual(want.gaps);
      });
    }
  }
});
