// The desktop page measuring a photo itself (decision D9), end to end in
// Chromium at 1280×800: opening the page loads nothing of the engine; a picked
// photo shows the engine's progress, then the photo with its pins, compared
// with native Python; the mouse fixes pins (click), zooms (wheel) and pans
// (drag); the object side; Add to the list, where the curve is marked
// "measured here", copies as Python's table and offers the files; a reload
// keeps it. Last, dropped files: one that is not a photo and a photo with too
// few pins show the core's message as is.
//
//   npm run e2e         (after npm run assets)
//
// Prints "[desktop] ..." lines: the JavaScript loaded at open and once the
// engine is needed, and the time from a photo to its pins. Screenshots go to
// test-results/desktop/ (or DESKTOP_SHOTS).

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import type { Point } from "../src/core/types";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SHOTS = process.env.DESKTOP_SHOTS ?? path.resolve(ROOT, "web/test-results/desktop");
/** The browser's OpenCV against native: S1, e2e/core.spec.ts. Pins are drawn rounded to 0.01 px. */
const PX_TOLERANCE = 0.06;
const MM_TOLERANCE = 0.005;

interface NativePhoto { photo: string; pins_px: Point[]; tsv: string }
interface Timing { name: string; status: string; ms: number; sinceDrop: number }
interface DesktopWindow { desktop: { timings(): Promise<Timing[]> } }

const log = (...a: unknown[]) => console.log("[desktop]", ...a);
const kib = (b: number) => `${(b / 1024).toFixed(1)} KiB`;

let pipe: NativePhoto;
let broken: string;               // a crop with too few pins
let context: BrowserContext;
let page: Page;
let jsAtOpen: Set<string>;

test.describe.configure({ mode: "serial" });
test.use({ actionTimeout: 30_000 });

test.beforeAll(async ({ browser }) => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "spline-wire-desktop-"));
  execFileSync("uv", ["run", "python", "web/e2e/native.py", out], { cwd: ROOT, stdio: "inherit" });
  pipe = (JSON.parse(fs.readFileSync(path.join(out, "native.json"), "utf8")) as { photos: { pipe: NativePhoto } }).photos.pipe;
  broken = path.join(out, "crop.jpg");
  execFileSync("uv", ["run", "python", "-c", [
    "import sys; from PIL import Image",
    "im = Image.open(sys.argv[1]); w, h = im.size",
    "im.crop((0, 0, 1850, h)).save(sys.argv[2], 'JPEG', quality=90)",
  ].join("\n"), pipe.photo, broken], { cwd: ROOT, stdio: "inherit" });
  fs.mkdirSync(SHOTS, { recursive: true });
  context = await browser.newContext({ viewport: { width: 1280, height: 800 }, permissions: ["clipboard-read", "clipboard-write"] });
  page = await context.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[page error]", e.message));
});

test.afterAll(async () => {
  await context?.close();
});

const pins = () => page.locator(".here .ov-layer circle[data-k]");
const viewBox = async () => (await page.locator(".here .photo-svg").getAttribute("viewBox"))!.split(" ").map(Number);
const timings = () => page.evaluate(() => (window as unknown as DesktopWindow).desktop.timings());

/** The scripts the page has loaded so far, Pyodide's own left out. */
async function scripts(): Promise<{ name: string; bytes: number }[]> {
  return page.evaluate(() => performance.getEntriesByType("resource")
    .filter((e) => /\.js(\?|$)/.test(e.name) && !/\/pyodide\//.test(e.name))
    .map((e) => ({ name: new URL(e.name).pathname, bytes: (e as PerformanceResourceTiming).decodedBodySize })));
}

async function center(locator: ReturnType<Page["locator"]>): Promise<{ x: number; y: number }> {
  const b = (await locator.boundingBox())!;
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
}

async function pinPositions(): Promise<Point[]> {
  return page.evaluate(() => [...document.querySelectorAll<SVGCircleElement>(".here .ov-layer circle[data-k]")]
    .sort((a, b) => Number(a.dataset.k) - Number(b.dataset.k))
    .map((c): [number, number] => [Number(c.getAttribute("cx")), Number(c.getAttribute("cy"))]));
}

/** A file dropped on the page, as the browser delivers a drag from the desktop. */
async function drop(name: string, bytes: Buffer, type: string): Promise<void> {
  await page.evaluate(([name, b64, type]) => {
    const data = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const dt = new DataTransfer();
    dt.items.add(new File([data], name, { type }));
    for (const kind of ["dragenter", "dragover", "drop"]) {
      document.querySelector("main")!.dispatchEvent(new DragEvent(kind, { dataTransfer: dt, bubbles: true, cancelable: true }));
    }
  }, [name, bytes.toString("base64"), type] as const);
}

test("opens without loading any of the measuring engine", async () => {
  await page.goto("./");
  await expect(page.locator("#link-text")).not.toHaveText("Starting…");
  await page.waitForLoadState("networkidle");
  const js = await scripts();
  const all = await page.evaluate(() => performance.getEntriesByType("resource").map((e) => new URL(e.name).pathname));
  expect(all.filter((n) => /pyodide|wheels|core-manifest|chains\.json|worker/.test(n))).toEqual([]);
  jsAtOpen = new Set(js.map((s) => s.name));
  // Nor does any of it carry the engine's code: the core client (which makes
  // the worker), the photo editor and the phone's measuring logic come in here.ts's chunks.
  for (const { name } of js) {
    const text = await (await page.request.get(name)).text();
    expect(text, name).not.toMatch(/new Worker|the measuring engine was restarted|Starting Python|pointercancel/);
  }
  // GitHub Pages cannot send the headers for cross-origin isolation (COOP and
  // COEP), and neither does `vite preview`: everything here runs without it.
  expect(await page.evaluate(() => self.crossOriginIsolated)).toBe(false);
  await expect(page.locator(".here-offer")).toContainText("Drop it on this page");
  log(`at open: ${js.length} scripts, ${kib(js.reduce((n, s) => n + s.bytes, 0))} (${js.map((s) => path.basename(s.name)).join(", ")})`);
});

test("a picked photo: the engine's progress, then the pins as native Python finds them", async () => {
  const t0 = Date.now();
  await page.setInputFiles("#pick", pipe.photo);
  await expect(page.locator(".here .engine")).toBeVisible();
  await expect(page.locator(".here [data-out=stage]")).toHaveText(/Starting Python|Loading NumPy|Starting Spline Wire/, { timeout: 30_000 });
  await expect(page.locator('.item.draft')).toContainText("pipe.jpg");
  await expect(pins()).toHaveCount(13, { timeout: 180_000 });
  await expect(page.locator(".here .engine")).toBeHidden();
  const wall = Date.now() - t0;
  const t = (await timings()).at(-1)!;
  const px = await pinPositions();
  const worst = Math.max(...px.map((p, i) => Math.hypot(p[0] - pipe.pins_px[i]![0], p[1] - pipe.pins_px[i]![1])));
  expect(worst).toBeLessThan(PX_TOLERANCE);
  await expect(page.locator(".here [data-out=status]")).toHaveText(/^Measured: \d+ points\.$/);
  await expect(page.locator(".here .nums")).toContainText("13 of 13");
  await expect(page.locator(".here [data-here=add]")).toBeEnabled();
  const js = await scripts();
  const later = js.filter((s) => !jsAtOpen.has(s.name));
  log(`photo to pins: ${(t.sinceDrop / 1000).toFixed(1)} s from the pick (engine load included; ${(wall / 1000).toFixed(1)} s in the test),`,
      `of which measuring ${(t.ms / 1000).toFixed(2)} s; pins ${worst.toFixed(3)} px from native`);
  log(`once the engine is needed: ${later.length} more scripts, ${kib(later.reduce((n, s) => n + s.bytes, 0))}`,
      `(${later.map((s) => path.basename(s.name)).join(", ")}), then Pyodide and its packages`);
});

test("the mouse: a click removes a pin and Undo brings it back; the wheel zooms and a drag pans, adding nothing", async () => {
  const before = await viewBox();
  await page.mouse.click(...Object.values(await center(page.locator('.here .ov-layer circle[data-k="5"]'))) as [number, number]);
  await expect(pins()).toHaveCount(12);
  await expect(page.locator(".here .ov-removed")).toHaveCount(1);
  await expect(page.locator(".here .nums")).toContainText("12 of 13");
  expect(await viewBox()).toEqual(before);           // an edit keeps the view
  const removed = (await timings()).at(-1)!.ms;
  await page.mouse.move(5, 400);
  await page.screenshot({ path: path.join(SHOTS, "measure-here-editing-light.png") });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.screenshot({ path: path.join(SHOTS, "measure-here-editing-dark.png") });
  await page.emulateMedia({ colorScheme: "light" });
  await page.getByRole("button", { name: "Undo the last pin change" }).click();
  await expect(pins()).toHaveCount(13);
  await expect(page.locator(".here .ov-removed")).toHaveCount(0);

  const c = await center(page.locator(".here .photo-svg"));
  await page.mouse.move(c.x, c.y);
  await page.mouse.wheel(0, -400);
  await expect.poll(async () => (await viewBox())[2]).toBeLessThan(before[2]! * 0.8);
  const zoomed = await viewBox();
  await page.mouse.move(c.x, c.y);
  await page.mouse.down();
  await page.mouse.move(c.x + 60, c.y + 30, { steps: 6 });
  await page.mouse.move(c.x + 120, c.y + 60, { steps: 6 });
  await page.mouse.up();
  const panned = await viewBox();
  expect(panned[0]).toBeLessThan(zoomed[0]!);
  expect(panned[2]).toBeCloseTo(zoomed[2]!, 3);
  await page.waitForTimeout(300);
  await expect(pins()).toHaveCount(13);
  await expect(page.locator(".here .ov-hand, .here .ov-mine, .here .ov-removed")).toHaveCount(0);

  // A click on empty photo adds a pin by hand; Reset takes it away.
  await page.getByRole("button", { name: "Fit the chain to the view" }).click();
  const box = (await page.locator(".here .photo-svg").boundingBox())!;
  await page.mouse.click(box.x + 24, box.y + 24);
  await expect(page.locator(".here .ov-hand, .here .ov-mine")).toHaveCount(1);
  await page.getByRole("button", { name: "Remove all pin changes" }).click();
  await expect(page.locator(".here .ov-hand, .here .ov-mine")).toHaveCount(0);
  await expect(pins()).toHaveCount(13);
  log(`re-measure after a click: ${removed} ms`);
});

test("the object side: Outside, a clicked point, Inside", async () => {
  await page.getByRole("button", { name: "Outside", exact: true }).click();
  await expect(page.locator(".here [data-out=side-now]")).toContainText("outside of the bend");
  await page.getByRole("button", { name: "Click its side" }).click();
  await expect(page.locator(".here [data-out=hint]")).toContainText("object's side");
  const p = await page.evaluate(() => {
    const cs = [...document.querySelectorAll<SVGCircleElement>(".here .ov-layer circle.ov-pin")];
    const mid = cs[6]!.getBoundingClientRect(), a = cs[0]!.getBoundingClientRect(), z = cs[12]!.getBoundingClientRect();
    const ends = { x: (a.x + z.x) / 2, y: (a.y + z.y) / 2 };
    const m = { x: mid.x + mid.width / 2, y: mid.y + mid.height / 2 };
    return { x: m.x + (ends.x - m.x) * 0.5, y: m.y + (ends.y - m.y) * 0.5 };
  });
  await page.mouse.click(p.x, p.y);
  await expect(page.locator(".here .ov-side")).toHaveCount(1);
  await expect(page.locator(".here [data-out=side-now]")).toContainText("from your click");
  await expect(pins()).toHaveCount(13);
  await page.getByRole("button", { name: "Inside", exact: true }).click();
  await expect(page.locator(".here [data-out=side-now]")).toContainText("inside of the bend");
});

test("Add to the list: marked as measured here, Python's table on the clipboard, and the files", async () => {
  await page.locator(".here [data-here=add]").click();
  await expect(page.locator(".here")).toHaveCount(0);
  await expect(page.locator(".item.draft")).toHaveCount(0);
  const item = page.locator(".item[data-id]").first();
  await expect(item.locator(".name")).toHaveText("pipe.jpg");
  await expect(item.locator(".tag-here")).toHaveText("measured here");
  await expect(item).toHaveAttribute("aria-current", "true");
  await expect(page.locator("#curve-title")).toHaveText("pipe.jpg");
  await expect(page.locator(".view .meta")).toContainText("measured here");
  await page.locator('[data-act="copy"]').click();
  await expect(page.locator("#toast")).toContainText("Copied");
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  const rows = copied.trimEnd().split("\n"), native = pipe.tsv.trimEnd().split("\n");
  expect(rows[0]).toBe("x_mm\ty_mm");
  expect(rows.length).toBe(native.length);
  const worst = Math.max(...rows.slice(1).map((r, i) => {
    const [a, b] = [r.split("\t").map(Number), native[i + 1]!.split("\t").map(Number)];
    return Math.max(Math.abs(a[0]! - b[0]!), Math.abs(a[1]! - b[1]!));
  }));
  expect(worst).toBeLessThan(MM_TOLERANCE);
  await expect(page.locator("#exp a[data-kind]")).toHaveCount(6);
  const download = page.waitForEvent("download");
  await page.locator('#exp a[data-kind="dxf"]').click();
  const dxf = await download;
  expect(dxf.suggestedFilename()).toBe("pipe-curve.dxf");
  expect(fs.readFileSync((await dxf.path())!, "utf8")).toContain("SPLINE");
  await page.screenshot({ path: path.join(SHOTS, "measured-here-light.png") });
  log(`clipboard table ${rows.length - 1} rows, worst ${worst.toFixed(4)} mm from native Python`);
});

test("a reload keeps the curve and its mark", async () => {
  await page.reload();
  const item = page.locator(".item[data-id]").first();
  await expect(item.locator(".tag-here")).toHaveText("measured here");
  await expect(page.locator("#curve-title")).toHaveText("pipe.jpg");
});

test("dropped files: one that is not a photo, and a photo with too few pins, show the core's message as is", async () => {
  await drop("notes.txt", Buffer.from("shopping list"), "text/plain");
  await expect(page.locator(".here .problem")).toBeVisible({ timeout: 120_000 });
  await expect(page.locator(".here .problem b")).toHaveText("Couldn't read this photo.");
  await expect(page.locator(".here .problem .raw")).toContainText("notes.txt is not a photo");
  await expect(page.locator("#drop-hint")).toBeHidden();
  const notPhoto = (await timings()).at(-1)!;

  await drop("crop.jpg", fs.readFileSync(broken), "image/jpeg");
  await expect(page.locator("#toast")).toContainText("notes.txt was not added");
  await expect(page.locator(".here .problem .raw")).toContainText("taking the perspective out needs at least", { timeout: 60_000 });
  await expect(page.locator(".here .problem b")).toContainText("pins in a row");
  await expect(page.locator(".here .problem .advice")).toContainText("click each pin");
  await expect(page.locator(".here .ov-unused").first()).toBeVisible();
  await expect(page.locator(".here [data-here=add]")).toBeDisabled();
  const crop = (await timings()).at(-1)!;
  await page.screenshot({ path: path.join(SHOTS, "measure-here-error-light.png") });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.screenshot({ path: path.join(SHOTS, "measure-here-error-dark.png") });
  await page.emulateMedia({ colorScheme: "light" });

  // A curve picked from the list leaves the photo where it is; Discard drops it.
  await page.locator(".item[data-id]").first().click();
  await expect(page.locator("#curve-title")).toHaveText("pipe.jpg");
  await page.locator(".item.draft").click();
  await expect(page.locator(".here .problem .raw")).toContainText("taking the perspective out");
  await page.getByRole("button", { name: "Discard" }).click();
  await expect(page.locator(".item.draft")).toHaveCount(0);
  await expect(page.locator("#curve-title")).toHaveText("pipe.jpg");
  log(`drop to answer: not a photo ${notPhoto.sinceDrop} ms (the engine loading again after the reload),`,
      `then a cropped photo ${crop.sinceDrop} ms (engine loaded)`);
});
