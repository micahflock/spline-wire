// live.ts against splinewire/live.py on every golden scene's blobs, scaled
// to the overlay's 640 px work frame, with the focal length known and estimated.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { expect, it } from "vitest";
import { analyse, type LiveAnalysis } from "./live";
import type { Vec2 } from "./numeric";
import type { ChainSpec } from "./types";

interface LiveCase { points: Vec2[]; diameters: number[]; size: [number, number]; focal_px: number | null; chain: ChainSpec; result: LiveAnalysis }

const scenes = (JSON.parse(
  gunzipSync(readFileSync(new URL("./golden/scenes.json.gz", import.meta.url))).toString("utf-8"),
) as { scenes: { id: string; live: LiveCase[] }[] }).scenes;

// 382 frames take ~5 s on a slow machine: its own timeout, as the other golden tests have.
it("analyses frames as Python does", { timeout: 60_000 }, () => {
  let n = 0, tilted = 0, ms = 0;
  for (const sc of scenes) {
    for (const c of sc.live) {
      const t0 = performance.now();
      const got = analyse(c.points, c.diameters, c.size, c.chain, c.focal_px);
      ms = Math.max(ms, performance.now() - t0);
      expect(got.order, sc.id).toEqual(c.result.order);
      expect(got.gaps, sc.id).toEqual(c.result.gaps);
      // Rounded to 0.1 degree and 0.001 mm: a value on a rounding edge may land either side.
      if (c.result.tilt_deg === null) expect(got.tilt_deg).toBeNull();
      else expect(Math.abs(got.tilt_deg! - c.result.tilt_deg)).toBeLessThanOrEqual(0.1 + 1e-9);
      if (c.result.link_residual_max_mm === null) expect(got.link_residual_max_mm).toBeNull();
      else expect(Math.abs(got.link_residual_max_mm! - c.result.link_residual_max_mm)).toBeLessThanOrEqual(0.001 + 1e-9);
      if (got.tilt_deg !== null) tilted++;
      n++;
    }
  }
  console.log(`[live] ${n} frames (${tilted} rectified) match Python; slowest ${ms.toFixed(1)} ms`);
  expect(n).toBeGreaterThan(300);
});
