// detect.ts against splinewire/detect.py.
//
// Always: two small photos committed with Python's detections
// (golden/detect.json.gz, web/scripts/golden_detect.py): the same fiducials in
// the same order, every field.
//
// With SPLINEWIRE_FULL_GOLDEN=1 and the renders cached in out/ (by
// experiments/cv_benchmark.py and web/scripts/golden.py): all 191 scenes of
// golden/scenes.json.gz from pixels to curve, the TypeScript detections then
// measured by measure.ts and compared with Python's measurements. About 8
// minutes; it prints how many scenes agree and the largest differences.

import { existsSync, readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { detectFiducials } from "./detect";
import { PipelineError } from "./errors";
import { readGrayPng } from "./golden/png";
import { measureDetections } from "./measure";
import type { Vec2 } from "./numeric";
import type { ChainSpec, Fiducial, Side } from "./types";

const read = (name: string) =>
  JSON.parse(gunzipSync(readFileSync(new URL(`./golden/${name}`, import.meta.url))).toString("utf-8"));

const PX = 1e-6;

function expectSameFiducials(got: Fiducial[], want: Fiducial[]): number {
  expect(got.length).toBe(want.length);
  let worst = 0;
  got.forEach((g, i) => {
    const w = want[i]!;
    expect(g.polarity).toBe(w.polarity);
    for (const [a, b] of [[g.center_px, w.center_px], [g.outer_axes_px, w.outer_axes_px], [g.inner_axes_px, w.inner_axes_px]] as const) {
      worst = Math.max(worst, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]));
    }
    expect(Math.abs(g.contrast - w.contrast)).toBeLessThan(1e-3);
    expect(Math.abs(g.surround - w.surround)).toBeLessThan(1e-3);
  });
  expect(worst).toBeLessThan(PX);
  return worst;
}

describe("detection against Python, committed photos", () => {
  for (const rec of read("detect.json.gz") as { photo: string; chain: ChainSpec; fiducials: Fiducial[] }[]) {
    it(rec.photo, () => {
      const img = readGrayPng(new URL(`./golden/${rec.photo}`, import.meta.url));
      expectSameFiducials(detectFiducials(img, rec.chain), rec.fiducials);
    }, 60000);
  }
});

interface Run {
  name: string; spec: ChainSpec; focal_px: number | null; side: Side | Vec2; error?: string;
  result?: { indices: number[]; pins_mm: Vec2[]; contacts_mm: Vec2[]; warnings: string[] };
}
interface Scene { id: string; size: [number, number]; fiducials: number[][]; runs: Run[] }

const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};
const full = env.SPLINEWIRE_FULL_GOLDEN === "1";

describe.skipIf(!full)("from pixels to curve, all scenes (SPLINEWIRE_FULL_GOLDEN=1)", () => {
  it("matches Python on every cached render", () => {
    const scenes = (read("scenes.json.gz") as { scenes: Scene[] }).scenes;
    let photos = 0, sameDetections = 0, runs = 0, sameRuns = 0, worstPx = 0, worstMm = 0, ms = 0;
    const differ: string[] = [];
    for (const sc of scenes) {
      const path = new URL(sc.id.startsWith("snap")
        ? `../../../out/golden-snap/${sc.id}.png` : `../../../out/cv-benchmark/dot/${sc.id}.png`, import.meta.url);
      if (!existsSync(path)) continue;
      photos++;
      const img = readGrayPng(path);
      const t0 = performance.now();
      const got = detectFiducials(img, sc.runs[0]!.spec);
      ms = Math.max(ms, performance.now() - t0);
      const want = sc.fiducials;
      let worst = got.length === want.length ? 0 : Infinity;
      if (worst === 0) {
        got.forEach((f, i) => {
          const w = want[i]!;
          worst = Math.max(worst, Math.abs(f.center_px[0] - w[0]!), Math.abs(f.center_px[1] - w[1]!),
            Math.abs(f.outer_axes_px[0] - w[2]!), Math.abs(f.outer_axes_px[1] - w[3]!));
          if (f.polarity !== w[5]) worst = Infinity;
        });
      }
      if (worst < PX) sameDetections++;
      else differ.push(`${sc.id}: ${got.length} against ${want.length} fiducials, worst ${worst.toExponential(1)} px`);
      if (Number.isFinite(worst)) worstPx = Math.max(worstPx, worst);
      for (const run of sc.runs) {
        runs++;
        let m;
        try {
          m = measureDetections(got, sc.size, run.spec, run.focal_px, run.side);
        } catch (err) {
          if (!(err instanceof PipelineError)) throw err;
          if (err.message === run.error) sameRuns++;
          else differ.push(`${sc.id} ${run.name}: error ${err.message}`);
          continue;
        }
        const r = run.result;
        if (!r || m.order.indices.join() !== r.indices.join() || m.warnings.join() !== r.warnings.join()) {
          differ.push(`${sc.id} ${run.name}: chain or warnings differ`);
          continue;
        }
        let d = 0;
        m.rectification.pins_mm.forEach((p, i) => (d = Math.max(d, Math.hypot(p[0] - r.pins_mm[i]![0], p[1] - r.pins_mm[i]![1]))));
        m.contacts_mm.forEach((p, i) => (d = Math.max(d, Math.hypot(p[0] - r.contacts_mm[i]![0], p[1] - r.contacts_mm[i]![1]))));
        worstMm = Math.max(worstMm, d);
        if (d < 1e-4) sameRuns++;
        else differ.push(`${sc.id} ${run.name}: ${d.toExponential(1)} mm`);
      }
    }
    console.log(`[detect] ${photos} photos: detections identical in ${sameDetections} (worst ${worstPx.toExponential(1)} px); `
      + `${sameRuns} of ${runs} measurements match Python (worst ${worstMm.toExponential(1)} mm); slowest detection ${ms.toFixed(0)} ms`);
    for (const d of differ) console.log(`[detect]   ${d}`);
    expect(photos).toBeGreaterThan(0);
    expect(differ).toEqual([]);
  }, 3600000);
});
