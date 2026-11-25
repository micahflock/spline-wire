// The core worker end to end, both engines: the TypeScript port (engine.ts,
// the default) and the Python core under Pyodide (worker.ts, ?engine=python),
// each in Chromium measuring synthetic photos, compared with native Python.
// The worker is reached through the desktop page (the site's root), whose
// desktop.engine() loads the measuring code on demand; the page itself, with
// a photo dropped on it, is tested in desktop.spec.ts.
//
//   npm run e2e         (after npm run assets)
//
// Timings and memory are printed as "[core] ..." lines.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import type { CoreClient } from "../src/core/client";
import type { Progress } from "../src/core/protocol";
import type { ChainSpec, MeasureOptions, Point } from "../src/core/types";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

interface NativePhoto {
  photo: string;
  truth: { pin_points: Point[] };
  cli_truth: { max_error_mm: number };
  pins_px: Point[];
  focal_px: number;
  pins_mm: Point[];
  tsv: string;
}
interface Native { chain: ChainSpec; photos: Record<"pipe" | "tilted" | "heic", NativePhoto> }

// S1: the browser's OpenCV 4.11 and native OpenCV 5.0 put pins up to 0.043 px
// apart on these photos (identical on the same OpenCV).
const PX_TOLERANCE = 0.05;
// The browser's truth error may exceed the CLI's by this much (the same 0.05 px
// at ~0.1 mm/px is 0.005 mm).
const MM_TOLERANCE = 0.005;

declare global {
  interface Window {
    core: CoreClient; chains: Promise<Record<string, ChainSpec>>; progressLog: Progress[];
    desktop: { engine(): Promise<{ core: CoreClient; chains: Promise<Record<string, ChainSpec>>; progress: Progress[] }> };
  }
}

/** The desktop page's engine, as window.core, window.chains and window.progressLog. */
async function openCore(p: Page, engine: "typescript" | "python"): Promise<void> {
  await p.goto(engine === "python" ? "./?engine=python" : "./");
  await p.evaluate(async () => {
    const e = await window.desktop.engine();
    Object.assign(window, { core: e.core, chains: e.chains, progressLog: e.progress });
  });
}

let native: Native;
let page: Page;
let engine: "typescript" | "python" = "typescript";
const log = (...a: unknown[]) => console.log(`[core ${engine}]`, ...a);
/** Every URL the page and its workers requested. */
const requests: string[] = [];
const mib = (b: number | null) => (b == null ? "?" : `${(b / 2 ** 20).toFixed(0)} MiB`);

test.describe.configure({ mode: "serial" });

for (const kind of ["typescript", "python"] as const) {
test.describe(kind, () => {
  test.beforeAll(() => {
    engine = kind;
  });

  test.beforeAll(async ({ browser }) => {
    const out = fs.mkdtempSync(path.join(os.tmpdir(), "spline-wire-e2e-"));
    execFileSync("uv", ["run", "python", "web/e2e/native.py", out], { cwd: ROOT, stdio: "inherit" });
    native = JSON.parse(fs.readFileSync(path.join(out, "native.json"), "utf8")) as Native;
    const context = await browser.newContext();
    // The photos are served under /__e2e/<name>.jpg, outside the site.
    await context.route("**/__e2e/*", (route) => {
      const name = path.basename(new URL(route.request().url()).pathname, ".jpg") as keyof Native["photos"];
      return route.fulfill({ path: native.photos[name].photo, contentType: "image/jpeg" });
    });
    context.on("request", (r) => requests.push(r.url()));      // workers' too
    page = await context.newPage();
    page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
    await openCore(page, engine);
  });

  /** core.measure in the page, with the photo fetched from /__e2e/. Returns the document and the time. */
  async function measure(id: string, photo: string, options: MeasureOptions) {
    return page.evaluate(async ([id, photo, options]) => {
      const t0 = performance.now();
      const doc = await window.core.measure(id, options, {
        name: `${photo}.jpg`, bytes: () => fetch(`__e2e/${photo}.jpg`).then((r) => r.arrayBuffer()),
      });
      return { doc, ms: performance.now() - t0 };
    }, [id, photo, options] as const);
  }

  const stats = () => page.evaluate(() => window.core.call("stats", {}));
  const maxDistance = (a: Point[], b: Point[]) => Math.max(...a.map((p, i) => Math.hypot(p[0] - b[i]![0], p[1] - b[i]![1])));

  test("loads, reporting progress", async () => {
    // Nothing has started the load yet: the desktop page loads the engine only for a photo.
    const { progress, result } = await page.evaluate(async () => {
      const result = await window.core.preload();
      return { progress: window.progressLog, result };
    });
    const again = await page.evaluate(() => window.core.preload());
    expect(again.ms).toBe(result.ms);            // idempotent: the same load
    if (engine === "typescript") {
      // Nothing to download or start: ready at once.
      expect(progress.map((p) => p.stage)).toEqual(["ready"]);
      expect(result.versions).toMatchObject({ engine: "typescript", splinewire: expect.any(String) });
      log(`load ${result.ms} ms, versions ${JSON.stringify(result.versions)}`);
      return;
    }
    const at = (stage: string) => (progress.find((p) => p.stage === stage) as unknown as { t: number }).t;
    const stages = `runtime ${at("packages") - at("runtime")} ms, packages ${at("starting") - at("packages")} ms, ` +
      `imports ${at("ready") - at("starting")} ms`;
    expect(progress.map((p) => p.stage)).toEqual(["runtime", "packages", "starting", "ready"]);
    const fractions = progress.map((p) => p.fraction);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(fractions.at(-1)).toBe(1);
    expect(result.versions).toMatchObject({ pyodide: expect.any(String), opencv: expect.any(String) });
    log(`cold load ${(result.ms / 1000).toFixed(1)} s (${stages}), versions ${JSON.stringify(result.versions)},`,
        `heap ${mib((await stats()).heap_bytes)}`);
  });

  for (const name of ["pipe", "tilted"] as const) {
    test(`measures the ${name} photo as the CLI does`, async () => {
      const n = native.photos[name];
      const { doc, ms } = await measure(name, name, {
        chain: native.chain, truth: { name: "truth.json", pin_points: n.truth.pin_points },
      });
      expect(doc.ok, doc.error ?? "").toBe(true);
      expect(doc.result!.pins).toBe(native.chain.n_pins);
      const px = maxDistance(doc.view!.pins.map((p) => [p.x, p.y]), n.pins_px);
      const truth = doc.result!.truth!;
      expect(px).toBeLessThan(PX_TOLERANCE);
      expect(truth.max_error_mm!).toBeLessThan(n.cli_truth.max_error_mm + MM_TOLERANCE);
      log(`${name} ${doc.info.width}x${doc.info.height}: measured in ${(ms / 1000).toFixed(2)} s (open + detect + solve);`,
          `pins ${px.toFixed(4)} px from native; truth max ${truth.max_error_mm!.toFixed(4)} mm`,
          `(CLI ${n.cli_truth.max_error_mm.toFixed(4)} mm); heap ${mib((await stats()).heap_bytes)}`);
    });
  }

  test("an edit measures again without detecting again", async () => {
    const opts = { chain: native.chain };
    const { doc: first } = await measure("pipe", "pipe", opts);
    const pin = first.view!.pins[4]!;
    const p: Point = [pin.x, pin.y];
    const t = await page.evaluate(async () => {
      // A fresh detection, for comparison: the photo opened again and measured.
      await window.core.call("close", { photo: "pipe" });
      const t0 = performance.now();
      await window.core.measure("pipe", { chain: (await window.chains).snap! },
        { name: "pipe.jpg", bytes: () => fetch("__e2e/pipe.jpg").then((r) => r.arrayBuffer()) });
      return performance.now() - t0;
    });
    const removed = await measure("pipe", "pipe", { ...opts, edits: { remove: [p], add: [] } });
    expect(removed.doc.result!.pins).toBe(native.chain.n_pins - 1);
    expect(removed.doc.view!.removed).toEqual([{ x: p[0], y: p[1], edit: 0 }]);
    const added = await measure("pipe", "pipe", { ...opts, edits: { remove: [p], add: [p] } });
    expect(added.doc.result!.pins).toBe(native.chain.n_pins);
    const back = added.doc.view!.pins.find((m) => m.edit === 0)!;
    expect(back.snapped).toBe(true);
    expect(Math.hypot(back.x - p[0], back.y - p[1])).toBeLessThan(0.5);
    const undone = await measure("pipe", "pipe", opts);
    expect(undone.doc.curve!.pins).toEqual(first.curve!.pins);
    // Detection is most of a measurement: an edit that detected again would take as long.
    for (const r of [removed, added, undone]) expect(r.ms).toBeLessThan(t / 2);
    log(`re-measure after an edit: removed ${removed.ms.toFixed(0)} ms, added back ${added.ms.toFixed(0)} ms,`,
        `undone ${undone.ms.toFixed(0)} ms; opening and detecting again took ${t.toFixed(0)} ms`);
  });

  test("outputs: every kind, named after the photo, the tsv as Python writes it", async () => {
    const { files, names, ms, again } = await page.evaluate(async () => {
      const t0 = performance.now();
      const out = await window.core.call("outputs", { photo: "pipe" });
      const t1 = performance.now();
      await window.core.call("outputs", { photo: "pipe" });
      return { ...out, ms: t1 - t0, again: performance.now() - t1 };
    });
    expect(Object.keys(files).sort()).toEqual(["csv", "dxf", "fusion_csv", "json", "svg", "tsv"]);
    expect(names).toEqual({ json: "pipe.json", csv: "pipe-curve.csv", svg: "pipe-curve.svg",
                            dxf: "pipe-curve.dxf", fusion_csv: "pipe-fusion-cm.csv" });
    expect(files.dxf).toContain("SPLINE");
    // The same table as native Python's, row for row. The browser's OpenCV
    // (4.11) is not the native one, so digits may differ by the pins' < 0.05 px.
    const browser = files.tsv.trimEnd().split("\n");
    const python = native.photos.pipe.tsv.trimEnd().split("\n");
    expect(browser[0]).toBe("x_mm\ty_mm");
    expect(browser.length).toBe(python.length);
    const worst = Math.max(...browser.slice(1).map((row, i) => {
      const [a, b] = [row.split("\t").map(Number), python[i + 1]!.split("\t").map(Number)];
      return Math.max(Math.abs(a[0]! - b[0]!), Math.abs(a[1]! - b[1]!));
    }));
    const same = browser.filter((row, i) => row === python[i]).length - 1;
    expect(worst).toBeLessThan(MM_TOLERANCE);
    log(`outputs: first call ${ms.toFixed(0)} ms, then ${again.toFixed(0)} ms;`,
        `tsv rows identical to native Python: ${same}/${browser.length - 1}, worst ${worst.toFixed(4)} mm`);
  });

  test("a file that is not a photo is a failure for the person", async () => {
    const failure = await page.evaluate(async () => {
      try {
        await window.core.measure("notes", { chain: (await window.chains).snap! },
          { name: "notes.txt", bytes: async () => new TextEncoder().encode("shopping list").buffer as ArrayBuffer });
        return null;
      } catch (e) {
        return (e as { failure?: unknown }).failure;
      }
    });
    expect(failure).toMatchObject({ kind: "person", message: expect.stringContaining("notes.txt is not a photo") });
  });

  test("an iPhone HEIC photo: the HEIF decoder loads on demand and keeps the EXIF focal length", async () => {
    // The TypeScript engine decodes HEIC with libheif in wasm (pipeline/heif.ts); Python with pillow-heif.
    const decoder = engine === "typescript" ? /libheif/ : /pillow_heif/;
    const loaded = (re: RegExp) => requests.filter((u) => re.test(u));
    expect(loaded(decoder)).toEqual([]);             // JPEG photos never load it
    const { doc, ms } = await measure("heic", "heic", { chain: native.chain });
    log(`HEIC: measured in ${(ms / 1000).toFixed(2)} s, decoder included`);
    expect(loaded(decoder).length).toBeGreaterThan(0);
    // Chromium cannot decode HEIC itself, and the TypeScript engine no longer hands it to Pyodide.
    if (engine === "typescript") expect(loaded(/pyodide|pillow_heif/)).toEqual([]);
    expect(doc.ok).toBe(true);
    expect(doc.info.format).toBe("HEIF");
    expect(doc.result!.focal_source).toBe("exif");
    expect(maxDistance(doc.view!.pins.map((p) => [p.x, p.y] as Point), native.photos.heic.pins_px)).toBeLessThan(PX_TOLERANCE);
  });

  test("a third photo closes the first, and measuring it again reopens it", async () => {
    const opts = { chain: native.chain };
    await measure("one", "pipe", opts);
    await measure("two", "tilted", opts);
    const before = await stats();
    await measure("three", "pipe", opts);
    const after = await stats();
    expect(after.open).toEqual(["two", "three"]);
    const { doc, reopened, ms } = await page.evaluate(async () => {
      let reopened = 0;
      const t0 = performance.now();
      const doc = await window.core.measure("one", { chain: (await window.chains).snap! }, {
        name: "pipe.jpg", bytes: () => { reopened++; return fetch("__e2e/pipe.jpg").then((r) => r.arrayBuffer()); },
      });
      return { doc, reopened, ms: performance.now() - t0 };
    });
    expect(reopened).toBe(1);
    expect(doc.ok).toBe(true);
    expect(maxDistance(doc.view!.pins.map((p) => [p.x, p.y]), native.photos.pipe.pins_px)).toBeLessThan(PX_TOLERANCE);
    expect((await stats()).open).toEqual(["three", "one"]);
    log(`heap with two photos open ${mib(before.heap_bytes)}, after the third ${mib(after.heap_bytes)},`,
        `after reopening the first ${mib((await stats()).heap_bytes)}; reopen + measure ${(ms / 1000).toFixed(2)} s`);
  });

  test("analyse puts a measurement's pins back in chain order", async () => {
    const { doc } = await measure("pipe", "pipe", { chain: native.chain });
    const pins = doc.view!.pins;
    const shuffled = [...pins.slice(7), ...pins.slice(0, 7)].reverse();
    const frame = {
      points: shuffled.map((p): Point => [p.x, p.y]), diameters: shuffled.map((p) => 2 * p.r),
      width: doc.view!.width, height: doc.view!.height, chain: native.chain, focal_px: null,
    };
    const { a, ms, known } = await page.evaluate(async ([frame, focal]) => {
      await window.core.call("analyse", frame);            // the first call may pay for imports
      const t0 = performance.now();
      const a = await window.core.call("analyse", frame);
      const t1 = performance.now();
      await window.core.call("analyse", { ...frame, focal_px: focal });
      return { a, ms: t1 - t0, known: performance.now() - t1 };
    }, [frame, native.photos.pipe.focal_px] as const);
    const ks = a.order.map((i) => shuffled[i]!.k);
    const forward = pins.map((p) => p.k);
    expect([forward, [...forward].reverse()]).toContainEqual(ks);
    expect(a.tilt_deg).not.toBeNull();
    log(`analyse round trip for ${pins.length} points: ${ms.toFixed(0)} ms with the focal length estimated,`,
        `${known.toFixed(0)} ms with it known; tilt ${a.tilt_deg}°`);
  });

  test("loads warm in a new tab", async () => {
    const tab = await page.context().newPage();
    await openCore(tab, engine);
    const result = await tab.evaluate(() => window.core.preload());
    log(`warm load ${(result.ms / 1000).toFixed(1)} s (HTTP cache warm)`);
    await tab.close();
  });
});
}
