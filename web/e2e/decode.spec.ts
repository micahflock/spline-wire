// Decoding in the browser against Pillow (src/pipeline/decode.ts): JPEGs as a
// phone writes them (e2e/decode_scene.py: gray; colour with 4:2:0 chroma;
// colour stored on its side with EXIF orientation 6; a realistic scene),
// decoded by Chromium, made gray with Pillow's formula, then detected and
// measured by the TypeScript pipeline in the page. Compared with what Python
// reads from the same bytes: the gray pixels, the fiducials and the pins.

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { build } from "esbuild";
import { expect, test } from "@playwright/test";

const OUT = join(test.info === undefined ? "" : "", "test-results", "decode");

test.describe.configure({ mode: "serial" });

test("Chromium decodes phone JPEGs as Pillow does, and measures the same pins", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  execFileSync("uv", ["run", "python", "e2e/decode_scene.py", OUT], { stdio: "inherit" });
  const native = JSON.parse(readFileSync(join(OUT, "decode.json"), "utf-8")) as {
    chain: Record<string, unknown>;
    photos: Record<string, { width: number; height: number; focal_px: number; fiducials: number[][]; pins_px: number[][]; pins_mm: number[][] }>;
  };
  const bundle = await build({ entryPoints: ["e2e/decode-entry.ts"], bundle: true, write: false, format: "iife", target: "es2022" });
  await page.route("http://decode.test/**", (route) => {
    const name = new URL(route.request().url()).pathname.slice(1);
    if (name === "") return route.fulfill({ contentType: "text/html", body: "<!doctype html><title>decode</title>" });
    return route.fulfill({ body: readFileSync(join(OUT, name)) });
  });
  await page.goto("http://decode.test/");
  await page.addScriptTag({ content: bundle.outputFiles[0]!.text });

  for (const [name, want] of Object.entries(native.photos)) {
    const got = await page.evaluate(async ({ name, chain, focal }) => {
      const p = (globalThis as any).pipeline;
      const t0 = performance.now();
      const gray = await p.decodeToGray(await (await fetch(`/${name}.jpg`)).blob());
      const decodeMs = performance.now() - t0;
      const ref = new Uint8Array(await (await fetch(`/${name}.gray`)).arrayBuffer());
      let differ = 0, worst = 0;
      for (let i = 0; i < ref.length; i++) {
        const d = Math.abs(ref[i]! - gray.data[i]!);
        if (d) differ++;
        if (d > worst) worst = d;
      }
      const spec = p.specFromDict(chain);
      const t1 = performance.now();
      const fids = p.detectFiducials(gray, spec);
      const detectMs = performance.now() - t1;
      const m = p.measureDetections(fids, [gray.width, gray.height], spec, focal);
      return {
        width: gray.width, height: gray.height, differ, worst, decodeMs, detectMs,
        fiducials: fids.map((f: any) => f.center_px), pinsPx: m.pins_px, pinsMm: m.rectification.pins_mm,
      };
    }, { name, chain: native.chain, focal: want.focal_px });

    expect([got.width, got.height], `${name}: upright size`).toEqual([want.width, want.height]);
    expect(got.fiducials.length, `${name}: fiducials`).toBe(want.fiducials.length);
    expect(got.pinsPx.length, `${name}: pins`).toBe(want.pins_px.length);
    // Same direction along the chain, then pin by pin.
    const px = Math.max(...got.pinsPx.map((p: number[], i: number) => Math.hypot(p[0]! - want.pins_px[i]![0]!, p[1]! - want.pins_px[i]![1]!)));
    const mm = Math.max(...got.pinsMm.map((p: number[], i: number) => Math.hypot(p[0]! - want.pins_mm[i]![0]!, p[1]! - want.pins_mm[i]![1]!)));
    console.log(`[decode] ${name} ${got.width}x${got.height}: ${got.differ} of ${got.width * got.height} gray pixels differ `
      + `(${(100 * got.differ / (got.width * got.height)).toFixed(3)}%, at most ${got.worst} levels); pins ${px.toFixed(4)} px, `
      + `${mm.toFixed(5)} mm from Python's; decode ${got.decodeMs.toFixed(0)} ms, detect ${got.detectMs.toFixed(0)} ms`);
    expect(px, `${name}: pins against Python's, px`).toBeLessThan(0.05);
    expect(mm, `${name}: pins against Python's, mm`).toBeLessThan(0.005);
  }
});
