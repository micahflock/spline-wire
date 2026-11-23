// The phone page end to end, at a phone's size with touch (Pixel 7 in Chromium):
// take a synthetic photo while the engine loads, check its pins against native
// Python, edit by touch, change the side, copy the points, reload, an error;
// then pair with a minimal desktop through the dev relay and send the curve.
// Last, the second visit's load time with and without the service worker.
//
//   npm run e2e         (after npm run assets; needs web/relay's npm install)
//
// Timings are printed as "[phone] ..." lines; screenshots go to
// test-results/phone/ (or PHONE_SHOTS).

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { devices, expect, test, type BrowserContext, type Page } from "@playwright/test";
import type { CoreClient } from "../src/core/client";
import type { Point } from "../src/core/types";
import { unpack } from "../src/relay/files";
import type { Curve } from "../src/relay/messages";
import { startDemo, type Demo } from "../relay/demo/serve";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SHOTS = process.env.PHONE_SHOTS ?? path.resolve(ROOT, "web/test-results/phone");
const PHONE = devices["Pixel 7"];
/** The browser's OpenCV against native: S1, e2e/core.spec.ts. Pins are drawn rounded to 0.01 px. */
const PX_TOLERANCE = 0.06;
const MM_TOLERANCE = 0.005;

interface NativePhoto { photo: string; pins_px: Point[]; tsv: string; cli_truth: { max_error_mm: number } }
interface PhoneWindow { phone: { core: CoreClient; timings: { id: string; status: string; ms: number }[]; engine: { readyAt: number; ms: number } | null } }

const log = (...a: unknown[]) => console.log("[phone]", ...a);
const mib = (b: number | null) => (b == null ? "?" : `${(b / 2 ** 20).toFixed(0)} MiB`);

/** A small JPEG in three bands, red, green and blue (made in the page): a photo whose colour shows. */
async function colourJpeg(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(async () => {
    const c = document.createElement("canvas");
    c.width = 600;
    c.height = 400;
    const g = c.getContext("2d")!;
    ["#e01010", "#10c010", "#1010e0"].forEach((fill, i) => {
      g.fillStyle = fill;
      g.fillRect(i * 200, 0, 200, 400);
    });
    const blob = await new Promise<Blob>((r) => c.toBlob((b) => r(b!), "image/jpeg", 0.95));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let s = "";
    for (const x of bytes) s += String.fromCharCode(x);
    return btoa(s);
  });
  return Buffer.from(b64, "base64");
}

/** The RGB of the middle of each band of the photo an SVG <image> shows (its blob: address);
 * null while it shows none (the page clears it while the photo is made again). */
async function bandColours(page: Page, image: string): Promise<number[][] | null> {
  return page.locator(image).evaluate(async (el) => {
    const href = el.getAttribute("href");
    if (!href?.startsWith("blob:")) return null;
    let bmp: ImageBitmap;
    try {
      bmp = await createImageBitmap(await (await fetch(href)).blob());
    } catch {
      return null;                                 // replaced (and released) meanwhile
    }
    const c = document.createElement("canvas");
    c.width = bmp.width;
    c.height = bmp.height;
    const g = c.getContext("2d")!;
    g.drawImage(bmp, 0, 0);
    return [1, 3, 5].map((k) => [...g.getImageData(Math.round((k / 6) * bmp.width), Math.round(bmp.height / 2), 1, 1).data.slice(0, 3)]);
  });
}

const spread = (rgb: number[]) => Math.max(...rgb) - Math.min(...rgb);
/** The most and the least colourful band, or null while no photo is shown. */
const spreads = async (page: Page, image: string) => {
  const bands = await bandColours(page, image);
  return bands && { most: Math.max(...bands.map(spread)), least: Math.min(...bands.map(spread)) };
};


let pipe: NativePhoto;
let broken: string;               // a crop with too few pins and no EXIF
let lensPhoto: string;            // a back-camera photo with full EXIF, no chain
let selfie: string;
let darkRims: string;
let context: BrowserContext;
let page: Page;
let photoId: string;
/** The engine's files for photoId as last measured (the Copy points test). */
let photoFiles: Record<string, string>;

test.describe.configure({ mode: "serial" });
// A button that stays disabled is a failure, not something to wait ten minutes for.
test.use({ actionTimeout: 30_000 });

test.beforeAll(async ({ browser }) => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "spline-wire-phone-"));
  execFileSync("uv", ["run", "python", "web/e2e/native.py", out], { cwd: ROOT, stdio: "inherit" });
  pipe = (JSON.parse(fs.readFileSync(path.join(out, "native.json"), "utf8")) as { photos: { pipe: NativePhoto } }).photos.pipe;
  // The left part of the pipe photo: three dots, two of them in a row; saved
  // without EXIF, so the focal length is unknown too.
  broken = path.join(out, "crop.jpg");
  execFileSync("uv", ["run", "python", "-c", [
    "import sys; from PIL import Image",
    "im = Image.open(sys.argv[1]); w, h = im.size",
    "im.crop((0, 0, 1850, h)).save(sys.argv[2], 'JPEG', quality=90)",
  ].join("\n"), pipe.photo, broken], { cwd: ROOT, stdio: "inherit" });
  // Two small photos as a phone's camera app writes them (D17): one from the
  // back camera, to teach the focal length, and a selfie, which must not.
  // The snap-fit chain printed the other way round: dark rims on light links.
  execFileSync("uv", ["run", "splinewire", "synth", "--shape", "pipe", "--dark-fiducials", "--size", "2000x1500",
    "--out", path.join(out, "dark")], { cwd: ROOT, stdio: "inherit" });
  darkRims = path.join(out, "dark", "pipe.jpg");
  lensPhoto = path.join(out, "lens.jpg");
  selfie = path.join(out, "selfie.jpg");
  execFileSync("uv", ["run", "python", "-c", [
    "import sys; from PIL import Image",
    "for path, f35, mm, lens in [(sys.argv[1], 26, 4.25, 'Phone back camera 4.25mm f/1.8'), (sys.argv[2], 30, 2.7, 'Phone front camera 2.7mm f/2.2')]:",
    "    e = Image.Exif(); e[0x010F] = 'Maker'; e[0x0110] = 'Phone'",
    "    x = e.get_ifd(0x8769); x[0xA405] = f35; x[0x920A] = mm; x[0xA434] = lens",
    "    Image.new('L', (400, 300), 128).save(path, 'JPEG', exif=e)",
  ].join("\n"), lensPhoto, selfie], { cwd: ROOT, stdio: "inherit" });
  fs.mkdirSync(SHOTS, { recursive: true });
  context = await browser.newContext({ ...PHONE, permissions: ["clipboard-read", "clipboard-write"] });
  page = await context.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[page error]", e.message));
});

test.afterAll(async () => {
  await context?.close();
});

const pins = () => page.locator("#overlay circle[data-k]");
const selectedId = () => page.locator("#photos li.on").getAttribute("data-id");
const viewBox = async () => (await page.locator("#photo").getAttribute("viewBox"))!.split(" ").map(Number);
const lastMs = () => page.evaluate(() => (window as unknown as PhoneWindow).phone.timings.at(-1)!.ms);

async function pinPositions(): Promise<Point[]> {
  return page.evaluate(() => [...document.querySelectorAll<SVGCircleElement>("#overlay circle[data-k]")]
    .filter((c) => c.classList.contains("ov-pin") || c.classList.contains("ov-mine"))
    .sort((a, b) => Number(a.dataset.k) - Number(b.dataset.k))
    .map((c): [number, number] => [Number(c.getAttribute("cx")), Number(c.getAttribute("cy"))]));
}

async function center(selector: string): Promise<{ x: number; y: number }> {
  const b = (await page.locator(selector).first().boundingBox())!;
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
}

/** One or two fingers, through the DevTools protocol: Playwright's touchscreen only taps. */
async function touch(paths: { x: number; y: number }[][], steps = 8): Promise<void> {
  const cdp = await context.newCDPSession(page);
  const at = (i: number) => paths.map((p, id) => {
    const [a, b] = [p[0]!, p[p.length - 1]!];
    return { x: a.x + ((b.x - a.x) * i) / steps, y: a.y + ((b.y - a.y) * i) / steps, id };
  });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: at(0) });
  for (let i = 1; i <= steps; i++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: at(i) });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await cdp.detach();
}

async function tapButton(name: string | RegExp): Promise<void> {
  const b = page.getByRole("button", { name }).first();
  await b.scrollIntoViewIfNeeded();
  await b.tap();
}

test("a photo taken while the engine loads is measured when it is ready", async () => {
  // The TypeScript engine is ready almost at once, sooner than the check below
  // can see its notice: hold its worker's script back so the photo really is
  // taken while it loads.
  let release!: () => void;
  const held = new Promise<void>((r) => { release = r; });
  await page.route(/\/assets\/engine-[^/]*\.js$/, async (route) => {
    await held;
    await route.continue();
  });
  const t0 = Date.now();
  await page.goto("phone/");
  await expect(page.locator("#engine")).toBeVisible();
  await page.setInputFiles("#take", pipe.photo);
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", /loading|queued/);
  release();
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 180_000 });
  photoId = (await selectedId())!;
  const engine = await page.evaluate(() => (window as unknown as PhoneWindow).phone.engine);
  const ms = await lastMs();
  await expect(pins()).toHaveCount(13);
  const px = await pinPositions();
  const fwd = Math.max(...px.map((p, i) => Math.hypot(p[0] - pipe.pins_px[i]![0], p[1] - pipe.pins_px[i]![1])));
  expect(fwd).toBeLessThan(PX_TOLERANCE);
  await expect(page.locator("#numbers")).toContainText("13 of 13");
  await expect(page.locator("#numbers")).toContainText("from the photo");
  await expect(page.locator("#r-tag")).toHaveText(/OK|CHECK/);
  const stats = await page.evaluate(() => (window as unknown as PhoneWindow).phone.core.call("stats", {}));
  log(`first visit: engine ready ${(engine!.readyAt / 1000).toFixed(1)} s after the page opened (load ${(engine!.ms / 1000).toFixed(1)} s);`,
      `photo measured ${(ms / 1000).toFixed(2)} s; pins ${fwd.toFixed(3)} px from native (CLI truth error`,
      `${pipe.cli_truth.max_error_mm.toFixed(4)} mm); heap ${mib(stats.heap_bytes)}; ${(Date.now() - t0) / 1000} s in all`);
  await page.locator("#result").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(SHOTS, "measured-light.png"), fullPage: true });
});

test("a tap on a pin removes it, Undo brings it back", async () => {
  await page.locator("#photo").scrollIntoViewIfNeeded();
  const before = await viewBox();
  await page.touchscreen.tap(...Object.values(await center('#overlay circle[data-k="5"]')) as [number, number]);
  await expect(pins()).toHaveCount(12);
  await expect(page.locator("#overlay .ov-removed")).toHaveCount(1);
  await expect(page.locator("#numbers")).toContainText("12 of 13");
  const removed = await lastMs();
  expect(await viewBox()).toEqual(before);         // an edit keeps the view
  await tapButton("Undo the last pin change");
  await expect(pins()).toHaveCount(13);
  await expect(page.locator("#overlay .ov-removed")).toHaveCount(0);
  const undone = await lastMs();
  expect((await pinPositions()).length).toBe(13);
  // A tap on empty photo adds a pin by hand; Reset removes it.
  const b = (await page.locator("#photo").boundingBox())!;
  await page.touchscreen.tap(b.x + 12, b.y + 12);
  await expect(page.locator("#overlay .ov-hand, #overlay .ov-mine")).toHaveCount(1);
  await tapButton("Remove all pin changes");
  await expect(page.locator("#overlay .ov-hand, #overlay .ov-mine")).toHaveCount(0);
  await expect(pins()).toHaveCount(13);
  log(`re-measure after a tap: removed ${removed} ms, undone ${undone} ms`);
});

test("a drag pans and a pinch zooms, and neither adds a pin", async () => {
  await page.locator("#photo").scrollIntoViewIfNeeded();
  const c = await center("#photo");
  const start = await viewBox();
  await touch([[{ x: c.x, y: c.y }, { x: c.x + 90, y: c.y + 40 }]]);
  const panned = await viewBox();
  expect(panned[0]).toBeLessThan(start[0]!);
  expect(panned[2]).toBeCloseTo(start[2]!, 3);
  await touch([[{ x: c.x - 30, y: c.y }, { x: c.x - 120, y: c.y }], [{ x: c.x + 30, y: c.y }, { x: c.x + 120, y: c.y }]]);
  const zoomed = await viewBox();
  expect(zoomed[2]).toBeLessThan(panned[2]! / 2);
  await page.waitForTimeout(300);
  await expect(pins()).toHaveCount(13);
  await expect(page.locator("#overlay .ov-hand, #overlay .ov-mine, #overlay .ov-removed")).toHaveCount(0);
  await tapButton("Fit the chain to the view");
  expect((await viewBox())[2]).toBeCloseTo(start[2]!, 1);
});

test("the object side: Outside, a tapped point, Inside", async () => {
  await tapButton(/^Outside$/);
  await expect(page.locator("#side-now")).toContainText("outside of the bend");
  await tapButton("Tap its side");
  await expect(page.locator("#hint")).toContainText("object's side");
  await page.locator("#photo").scrollIntoViewIfNeeded();
  // The pipe's chain bends round the object: its middle pin's far side, away from the ends, is inside.
  const p = await page.evaluate(() => {
    const cs = [...document.querySelectorAll<SVGCircleElement>("#overlay circle.ov-pin")];
    const mid = cs[6]!.getBoundingClientRect(), a = cs[0]!.getBoundingClientRect(), z = cs[12]!.getBoundingClientRect();
    const ends = { x: (a.x + z.x) / 2, y: (a.y + z.y) / 2 };
    const m = { x: mid.x + mid.width / 2, y: mid.y + mid.height / 2 };
    return { x: m.x + (ends.x - m.x) * 0.5, y: m.y + (ends.y - m.y) * 0.5 };
  });
  await page.touchscreen.tap(p.x, p.y);
  await expect(page.locator("#overlay .ov-side")).toHaveCount(1);
  await expect(page.locator("#side-now")).toContainText("from your tap");
  await expect(pins()).toHaveCount(13);                // the side tap is not an edit
  await tapButton(/^Inside$/);
  await expect(page.locator("#side-now")).toContainText("inside of the bend");
  await expect(page.locator("#overlay .ov-side")).toHaveCount(0);
});

test("Copy points puts the Fusion add-in's table on the clipboard", async () => {
  await expect(page.locator("#photos li.on")).toHaveAttribute("data-status", "done");
  await tapButton("Copy points");
  await expect(page.locator("#toast")).toContainText("Copied");
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  const core = await page.evaluate((id) => (window as unknown as PhoneWindow).phone.core.call("outputs", { photo: id }), photoId);
  expect(copied).toBe(core.files.tsv);                 // byte for byte what Python writes
  photoFiles = core.files;
  const rows = copied.trimEnd().split("\n"), native = pipe.tsv.trimEnd().split("\n");
  expect(rows[0]).toBe("x_mm\ty_mm");
  expect(rows.length).toBe(native.length);
  const worst = Math.max(...rows.slice(1).map((r, i) => {
    const [a, b] = [r.split("\t").map(Number), native[i + 1]!.split("\t").map(Number)];
    return Math.max(Math.abs(a[0]! - b[0]!), Math.abs(a[1]! - b[1]!));
  }));
  expect(worst).toBeLessThan(MM_TOLERANCE);
  await expect(page.locator("#downloads button")).toHaveCount(5);
  const download = page.waitForEvent("download");
  await tapButton("Download the DXF file");
  expect((await download).suggestedFilename()).toBe("pipe-curve.dxf");
});

test("a reload keeps the photo and its measurement", async () => {
  await page.reload();
  const item = page.locator(`#photos li[data-id="${photoId}"]`);
  await expect(item).toHaveAttribute("data-status", "done");
  await expect(pins()).toHaveCount(13);                // from the stored measurement, before the engine is back
  await expect(page.locator("#numbers")).toContainText("13 of 13");
  await expect(page.locator("#photo-img")).toHaveAttribute("href", /^blob:/);
});

test("settings learn a focal length per lens, not from a selfie, and offer it as the saved one", async () => {
  await page.locator("#settings summary").tap();
  // The synthetic photo has the 35 mm tag but names no camera or lens: nothing to remember it under (D17).
  await expect(page.locator("#s-offer")).toBeHidden();
  await page.setInputFiles("#s-teach", selfie);
  await expect(page.locator("#toast")).toContainText("front camera");
  await expect(page.locator("#s-lenses li")).toHaveCount(0);
  await page.setInputFiles("#s-teach", lensPhoto);
  await expect(page.locator("#toast")).toContainText("Learned 26 mm for Phone back camera 4.25mm f/1.8");
  await expect(page.locator("#s-lenses li")).toHaveCount(1);
  await expect(page.locator("#s-offer-text")).toHaveText("Use 26 mm from your last photo (Phone back camera 4.25mm f/1.8) for photos without camera data?");
  await expect(page.locator("#s-chain")).toHaveValue("snap");
  await page.locator("#settings").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(SHOTS, "settings-light.png"), fullPage: true });
  await tapButton("Use it");
  await expect(page.locator("#s-focal")).toHaveValue("26");
  await expect(page.locator("#s-offer")).toBeHidden();
  await page.locator("#settings summary").tap();
});

test("an error is shown as is, and the photo can still be fixed by hand", async () => {
  await page.setInputFiles("#choose", broken);
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "error", { timeout: 120_000 });
  await expect(page.locator("#r-problem")).toContainText("pins in a row");
  await expect(page.locator("#r-problem .raw")).toContainText("taking the perspective out needs at least");
  await expect(page.locator("#r-tag")).toHaveText("FAILED");
  await expect(page.locator("#overlay .ov-unused").first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Copy points" })).toBeDisabled();
  await page.screenshot({ path: path.join(SHOTS, "error-light.png"), fullPage: true });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.screenshot({ path: path.join(SHOTS, "error-dark.png"), fullPage: true });
  await page.locator(`#photos li[data-id="${photoId}"] button.open`).tap();
  await expect(page.locator("#numbers")).toContainText("13 of 13");
  await page.locator("#result").scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(SHOTS, "measured-dark.png"), fullPage: true });
  await page.emulateMedia({ colorScheme: "light" });
  // Nothing wider than the phone.
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(PHONE.viewport.width);
});

test("dark rims on light links are found once Colours says so", async () => {
  await page.setInputFiles("#choose", darkRims);
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "error", { timeout: 120_000 });
  await expect(page.locator("#r-problem")).toContainText("No chain found");
  await expect(page.locator("#r-problem")).toContainText("choose that under Colours in Settings");
  await page.locator("#settings summary").tap();
  await page.locator("#s-shade").selectOption("dark");
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 120_000 });
  await expect(page.locator("#numbers")).toContainText("13 of 13");
  // Back to the light chain for what follows, with the light photo open (so it is not measured again).
  await page.locator(`#photos li[data-id="${photoId}"] button.open`).tap();
  await page.locator("#s-shade").selectOption("light");
  await page.locator("#settings summary").tap();
  await expect(page.locator(`#photos li[data-id="${photoId}"]`)).toHaveAttribute("data-status", "done");
});

test.describe("paired with a computer", () => {
  let demo: Demo;
  let desk: BrowserContext;          // the computer's browser: its storage keeps the pairing and the curves
  let desktop: Page;
  let over = false;

  test.beforeAll(async ({ browser }) => {
    demo = await startDemo();
    desk = await browser.newContext();
  });
  test.afterAll(async () => {
    over = true;
    await page.goto("phone/");                         // no ?relay=: this build has no relay, so the page stops trying
    await desk?.close();
    await demo?.close();
  });

  /** The desktop page (the site's root) on the test relay. */
  const openDesktop = async (ctx: BrowserContext = desk) => {
    const p = await ctx.newPage();
    p.on("pageerror", (e) => log(`[desktop error] ${e.message}`));
    await p.goto(`./?relay=${encodeURIComponent(demo.relay.url)}`);
    return p;
  };
  const received = (p: Page = desktop) => p.evaluate(() =>
    JSON.parse(localStorage.getItem("splinewire-desktop:inbox") ?? "[]") as Curve[]);
  const items = (p: Page = desktop) => p.locator("#list button.item[data-id]");
  const phoneItem = (id: string) => page.locator(`#photos li[data-id="${id}"]`);
  /** A photo added on the phone (from the library), measured; its id. */
  const addPhoto = async () => {
    const before = await page.locator("#photos li").count();
    await page.setInputFiles("#choose", pipe.photo);
    await expect(page.locator("#photos li")).toHaveCount(before + 1);
    const id = (await page.locator("#photos li").first().getAttribute("data-id"))!;
    await expect(phoneItem(id)).toHaveAttribute("data-status", "done", { timeout: 60_000 });
    return id;
  };

  test("pairs by the scanned link; a photo from before is sent with the button, with its files", async () => {
    desktop = await openDesktop();
    const link = (await desktop.locator("[data-pair-link]").first().textContent())!;
    const secret = new URL(link).hash;                 // #pair=<secret>
    expect(secret).toMatch(/^#pair=/);
    page.on("websocket", (ws) => {
      ws.on("close", () => { if (!over) log(`relay socket closed: ${ws.url()}`); });
    });
    await page.goto(`phone/?relay=${encodeURIComponent(demo.relay.url)}${secret}`);
    await expect(page).toHaveURL(/\/phone\/\?relay=[^#]*$/);   // the secret is taken off the address
    await expect(page.locator("#link-text")).toHaveText("Computer connected");
    await expect(phoneItem(photoId)).toHaveAttribute("data-status", "done");
    await phoneItem(photoId).locator("button.open").tap();
    await expect(page.locator("#send-note")).toContainText("Taken before this phone was paired");
    // The engine is new after the navigation: the files sent are the ones kept with the photo when it was measured.
    await tapButton("Send to computer");
    await expect(items()).toHaveCount(1, { timeout: 60_000 });
    await expect(page.locator("#send-note")).toHaveText(/Delivered to the computer/);
    await expect(page.getByRole("button", { name: "Send to computer" })).toBeHidden();
    const [curve] = await received();
    expect(curve!.rev).toBe(1);
    expect(curve!.points_mm.length).toBeGreaterThan(10);
    expect(curve!.photo).toBe(path.basename(pipe.photo));
    const files = await unpack(curve!.files!);
    // The same files the engine wrote for this photo, the DXF included (nothing in it changes with time).
    for (const kind of ["json", "csv", "svg", "dxf", "fusion_csv"] as const) {
      expect(files[kind], kind).toBe(photoFiles[kind]);
    }
    log("link states:", JSON.stringify(await page.evaluate(() => (window as unknown as { phone: { link: unknown } }).phone.link)));
  });

  test("a change on the phone replaces the curve on the computer, also one made while it is closed", async () => {
    const id = (await received())[0]!.id;
    await tapButton(/^Outside$/);                     // the object side changed
    await expect.poll(async () => (await received()).map((c) => [c.id, c.rev, c.object_side])).toEqual([[id, 2, "outside"]]);
    await expect(items()).toHaveCount(1);
    await expect(desktop.locator("#toast")).toContainText("Changed on the phone");
    await expect(page.locator("#send-note")).toHaveText("Delivered to the computer, with your changes.");
    await desktop.close();                             // the computer's page is closed
    await expect(page.locator("#link-text")).toHaveText("Computer not open");
    await page.locator("#photo").scrollIntoViewIfNeeded();
    await page.touchscreen.tap(...Object.values(await center('#overlay circle[data-k="3"]')) as [number, number]);
    await expect(pins()).toHaveCount(12);
    await expect(page.locator("#send-note")).toHaveText(/Queued: the computer's page is not open. The change goes/);
    await expect(page.locator('#outbox li[data-state="queued-no-computer"]')).toHaveCount(1);
    await expect(page.locator('#outbox li button')).toHaveCount(0);       // a change to a delivered curve is not held back
    desktop = await openDesktop();
    await expect.poll(async () => (await received()).map((c) => [c.id, c.rev, c.diagnostics.missing_pins])).toEqual([[id, 3, 1]]);
    await expect(page.locator("#send-note")).toHaveText(/Delivered/);
    await tapButton("Undo the last pin change");
    await expect(pins()).toHaveCount(13);
    await expect.poll(async () => (await received()).map((c) => [c.rev, c.diagnostics.missing_pins])).toEqual([[4, 0]]);
    await expect(items()).toHaveCount(1);
    await page.screenshot({ path: path.join(SHOTS, "paired-light.png"), fullPage: true });
  });

  test("a photo taken while paired goes by itself; stars and deletions stay in step both ways", async () => {
    const curveOf = async (photo: string) => (await received()).find((c) => c.photo === path.basename(pipe.photo) && c.id !== photo);
    const first = (await received())[0]!.id;
    const b = await addPhoto();                         // no button: it goes once measured
    await expect(items()).toHaveCount(2, { timeout: 30_000 });
    await expect(page.locator("#send-note")).toHaveText(/Delivered to the computer/);
    const bCurve = (await curveOf(first))!.id;

    // A star set on the phone shows on the computer; one taken off there goes off on the phone.
    await phoneItem(photoId).locator("button.star").tap();
    await expect(desktop.locator(`button.item[data-id="${first}"] .starred`)).toHaveCount(1);
    await desktop.locator(`button.item[data-id="${first}"]`).click();
    await expect(desktop.locator('[data-act="star"]')).toHaveAttribute("aria-pressed", "true");
    await desktop.locator('[data-act="star"]').click();
    await expect(phoneItem(photoId).locator("button.star")).toHaveAttribute("aria-pressed", "false");
    // Starred on the computer while the phone's page is closed: the phone has it when it opens.
    await desktop.locator('[data-act="star"]').click();
    await page.goto("phone/");                          // the phone's page reloaded, paired
    await page.goto(`phone/?relay=${encodeURIComponent(demo.relay.url)}`);
    await expect(page.locator("#link-text")).toHaveText("Computer connected");
    await expect(phoneItem(photoId).locator("button.star")).toHaveAttribute("aria-pressed", "true");
    await phoneItem(photoId).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(SHOTS, "paired-star.png") });
    await desktop.screenshot({ path: path.join(SHOTS, "desktop-star.png") });

    // Deleted on the computer: the photo goes from the phone.
    await desktop.locator(`button.item[data-id="${bCurve}"]`).click();
    await desktop.locator('[data-act="delete"]').click();
    await expect(desktop.locator("#confirm-body")).toContainText("the photo goes from the phone too");
    await desktop.locator("#confirm-yes").click();
    await expect(phoneItem(b)).toHaveCount(0);
    await expect(page.locator("#toast")).toContainText("Deleted on the computer");

    // Deleted on the phone: the curve goes from the computer, also while its page is closed.
    const c = await addPhoto();
    await expect(items()).toHaveCount(2, { timeout: 30_000 });
    await desktop.close();
    page.once("dialog", (d) => {
      expect(d.message()).toContain("from this phone and the computer");
      void d.accept();
    });
    await phoneItem(c).locator("button.del").tap();
    await expect(phoneItem(c)).toHaveCount(0);
    desktop = await openDesktop();
    await expect(items()).toHaveCount(1);
    expect((await received()).map((x) => x.id)).toEqual([first]);
    await desktop.close();
  });

  test("a name given on either side shows on both, and the files take it", async () => {
    desktop = await openDesktop();
    const first = (await received())[0]!.id;
    const row = desktop.locator(`button.item[data-id="${first}"]`);
    await row.click();
    // Renamed on the phone, in the photo's header.
    await phoneItem(photoId).locator("button.open").tap();
    await tapButton("Rename");
    const field = page.getByRole("textbox", { name: "Name of the curve" });
    await field.fill("Pipe 1.5in, left");
    await field.press("Enter");
    await expect(page.locator("#r-name")).toHaveText("Pipe 1.5in, left");
    await expect(phoneItem(photoId).locator(".name")).toHaveText("Pipe 1.5in, left");
    await expect(row.locator(".name")).toHaveText("Pipe 1.5in, left");
    await expect(desktop.locator("#curve-title")).toHaveText("Pipe 1.5in, left");
    await expect(desktop.locator(".view .meta")).toContainText("pipe.jpg");          // the photo's name, for reference
    await expect(desktop.locator('#exp a[data-kind="dxf"]')).toHaveAttribute("download", "Pipe 1.5in, left-curve.dxf");
    // Renamed on the computer: Enter keeps it.
    await desktop.getByRole("button", { name: "Rename" }).click();
    await desktop.getByRole("textbox", { name: "Name of the curve" }).fill("Pipe, left");
    await desktop.keyboard.press("Enter");
    await expect(desktop.locator("#curve-title")).toHaveText("Pipe, left");
    await expect(phoneItem(photoId).locator(".name")).toHaveText("Pipe, left");
    await expect(page.locator("#r-name")).toHaveText("Pipe, left");
    // Escape does not; a click elsewhere does, and the click still goes where it was meant.
    await desktop.getByRole("button", { name: "Rename" }).click();
    await desktop.getByRole("textbox", { name: "Name of the curve" }).fill("Not this");
    await desktop.keyboard.press("Escape");
    await expect(desktop.locator("#curve-title")).toHaveText("Pipe, left");
    await desktop.getByRole("button", { name: "Rename" }).click();
    await desktop.getByRole("textbox", { name: "Name of the curve" }).fill("Pipe bend");
    await desktop.getByRole("button", { name: /Star/ }).click();
    await expect(desktop.locator("#curve-title")).toHaveText("Pipe bend");
    await expect(desktop.getByRole("button", { name: /Star/ })).toHaveAttribute("aria-pressed", "false");   // starred before: the click unstarred it
    await expect(phoneItem(photoId).locator(".name")).toHaveText("Pipe bend");
    // The phone's own files take the name too.
    const download = page.waitForEvent("download");
    await page.locator("#downloads button", { hasText: "CSV" }).first().tap();
    expect((await download).suggestedFilename()).toBe("Pipe bend-curve.csv");
    // An empty name gives the photo's back, on both.
    await tapButton("Rename");
    await field.fill("");
    await field.press("Enter");
    await expect(page.locator("#r-name")).toHaveText("pipe.jpg");
    await expect(row.locator(".name")).toHaveText("pipe.jpg");
    await desktop.close();
  });

  test("a code scanned while the page is open pairs it, without loading the page again", async ({ browser }) => {
    // Another computer, so another secret. The browser only changes the #fragment
    // of the open page (same document), as Safari does for a scanned link.
    const other = await browser.newContext();
    try {
      const otherDesktop = await openDesktop(other);
      const secret = new URL((await otherDesktop.locator("[data-pair-link]").first().textContent())!).hash;
      const marker = await page.evaluate(() => ((window as unknown as { loadMark: number }).loadMark = Math.random()));
      await page.goto(`phone/?relay=${encodeURIComponent(demo.relay.url)}${secret}`);
      expect(await page.evaluate(() => (window as unknown as { loadMark?: number }).loadMark)).toBe(marker);  // not reloaded
      await expect(page).toHaveURL(/\/phone\/\?relay=[^#]*$/);
      await expect(page.locator("#toast")).toContainText("Paired with the computer");
      await expect(page.locator("#link-text")).toHaveText("Computer connected");
      await otherDesktop.close();
      await expect(page.locator("#link-text")).toHaveText("Computer not open");
    } finally {
      await other.close();
    }
  });
});

/** Visits the phone page until its engine is ready: the core's files (Pyodide,
 * wheels) fetched by the page or its workers, from the network and from the
 * service worker, and their bytes. */
test("past Photos kept and with Delete photos, keep curves, photos go and their curves stay; Clear history empties the list", async ({ browser }) => {
  const ctx = await browser.newContext({ ...PHONE, permissions: ["clipboard-read", "clipboard-write"] });
  const p = await ctx.newPage();
  const tap = async (name: string) => {
    const b = p.getByRole("button", { name }).first();
    await b.scrollIntoViewIfNeeded();
    await b.tap();
  };
  const blobKinds = (id: string) => p.evaluate((id) => new Promise<string[]>((resolve, reject) => {
    const req = indexedDB.open("splinewire-phone");
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const keys = req.result.transaction("blobs").objectStore("blobs").getAllKeys();
      keys.onsuccess = () => resolve((keys.result as string[]).filter((k) => k.startsWith(`${id}:`)).map((k) => k.slice(id.length + 1)).sort());
    };
  }), id);
  await p.goto("phone/");
  await p.locator("#settings summary").tap();
  await p.selectOption("#s-keep", "1");
  await p.setInputFiles("#take", pipe.photo);
  const items = p.locator("#photos li");
  await expect(items.first()).toHaveAttribute("data-status", "done", { timeout: 180_000 });
  const first = (await items.first().getAttribute("data-id"))!;
  const want = await p.evaluate((id) => (window as unknown as PhoneWindow).phone.core.call("outputs", { photo: id }), first);
  await expect.poll(() => blobKinds(first)).toEqual(["original", "outputs", "preview", "thumb"]);

  // A second photo: the first, past "the newest only", loses its photo and keeps its curve.
  await p.setInputFiles("#take", pipe.photo);
  await expect(items).toHaveCount(2);
  await expect(items.first()).toHaveAttribute("data-status", "done", { timeout: 180_000 });
  const old = p.locator(`#photos li[data-id="${first}"]`);
  await expect(old.locator(".meta")).toContainText("curve only", { timeout: 15_000 });
  expect(await blobKinds(first)).toEqual(["outputs", "thumb"]);
  await expect(p.locator("#space")).toContainText("1 photo (");
  await expect(p.locator("#space")).toContainText("1 curve without its photo");

  // After a reload, with the core holding nothing, the old curve is copied and downloaded from its kept files.
  await p.reload();
  await old.locator("button.open").tap();
  await expect(p.locator("#r-status")).toContainText("deleted to save space");
  await expect(p.locator("#editor")).toBeHidden();
  await expect(p.getByRole("button", { name: "Outside" })).toBeDisabled();
  await tap("Copy points");
  await expect(p.locator("#toast")).toContainText("Copied");
  expect(await p.evaluate(() => navigator.clipboard.readText())).toBe(want.files.tsv);
  const download = p.waitForEvent("download");
  await tap("Download the DXF file");
  expect(fs.readFileSync((await (await download).path())!, "utf8")).toBe(want.files.dxf);
  await expect(old).toHaveAttribute("data-status", "done");

  // Delete photos, keep curves: the newer one goes too.
  p.once("dialog", (d) => {
    expect(d.message()).toBe("Delete 1 photo and keep its curve? The curves can still be copied, downloaded and sent, but their pins can no longer be fixed.");
    void d.accept();
  });
  await tap("Delete photos, keep curves");
  await expect(p.locator("#toast")).toContainText("Deleted 1 photo, kept the curves.");
  await expect(items.locator(".meta", { hasText: "curve only" })).toHaveCount(2);
  await expect(p.getByRole("button", { name: "Delete photos, keep curves" })).toBeHidden();

  // Clear history: nothing left, after a reload too.
  p.once("dialog", (d) => {
    expect(d.message()).toBe("Delete all 2 photos and curves from this phone?");
    void d.accept();
  });
  await tap("Clear history");
  await expect(items).toHaveCount(0);
  await expect(p.locator("#photos-box")).toBeHidden();
  await expect(p.locator("#result")).toBeHidden();
  await p.reload();
  await expect(p.locator("#engine-stage")).toBeAttached();
  await expect(items).toHaveCount(0);
  expect(await blobKinds(first)).toEqual([]);
  await ctx.close();
});

test("photos shown grey while the live camera is (the default), in colour when it is; measuring is unchanged", async ({ browser }) => {
  const ctx = await browser.newContext({ ...PHONE });
  const p = await ctx.newPage();
  await p.goto("phone/");
  await p.setInputFiles("#take", { name: "colour.jpg", mimeType: "image/jpeg", buffer: await colourJpeg(p) });
  await expect(p.locator("#photo-img")).toHaveAttribute("href", /^blob:/);
  // Grey: each band's three channels equal (to JPEG's rounding), and the thumbnail filtered.
  await expect.poll(async () => (await spreads(p, "#photo-img"))?.most ?? Infinity).toBeLessThanOrEqual(4);
  expect(await p.locator("#r-thumb").evaluate((t) => getComputedStyle(t).filter)).toBe("grayscale(1)");
  // Live camera shows: in colour. The photo is shown again in colour.
  await p.locator("#settings summary").tap();
  await p.selectOption("#s-view", "color");
  await expect.poll(async () => (await spreads(p, "#photo-img"))?.least ?? 0).toBeGreaterThan(100);
  expect(await p.locator("#r-thumb").evaluate((t) => getComputedStyle(t).filter)).toBe("none");
  await ctx.close();
});

const visitPhone = async (ctx: BrowserContext, address = "phone/") => {
    const p = await ctx.newPage();
    let network = 0, fromSw = 0, bytes = 0;
    const sizes: Promise<void>[] = [];
    p.on("requestfinished", (r) => {
      if (!/\/(pyodide|wheels)\//.test(r.url())) return;
      sizes.push(r.response().then(async (res) => {
        if (res?.fromServiceWorker()) {
          fromSw++;
          return;
        }
        network++;
        bytes += (await r.sizes()).responseBodySize;
      }).catch(() => {}));
    });
    const t0 = Date.now();
    await p.goto(address);
    await p.waitForFunction(() => (window as unknown as PhoneWindow).phone.engine !== null, null, { timeout: 120_000 });
    const engine = await p.evaluate(() => (window as unknown as PhoneWindow).phone.engine!);
    await Promise.all(sizes);
    await p.close();
    return { readyAt: engine.readyAt, load: engine.ms, wall: Date.now() - t0, network, fromSw, mb: (bytes / 1e6).toFixed(1) };
};

test("the TypeScript engine downloads nothing of Pyodide, and is ready at once", async ({ browser }) => {
  const ctx = await browser.newContext({ ...PHONE });
  const v = await visitPhone(ctx);
  await ctx.close();
  log(`TypeScript engine: ready ${(v.readyAt / 1000).toFixed(2)} s after the page opened; core files fetched: ${v.network}`);
  expect(v.network + v.fromSw).toBe(0);
});

test("the second visit with the Python engine (?engine=python): with the service worker and without", async ({ browser }) => {
  // Each case in a fresh context: a first visit fills the caches, the second is timed.
  const visit = (ctx: BrowserContext) => visitPhone(ctx, "phone/?engine=python");
  const out: string[] = [];
  for (const sw of ["allow", "block"] as const) {
    const ctx = await browser.newContext({ ...PHONE, serviceWorkers: sw });
    const first = await visit(ctx);
    const second = await visit(ctx);
    const caches = sw === "allow" ? await (async () => {
      const p = await ctx.newPage();
      await p.goto("phone/");
      const names = await p.evaluate(() => caches.keys());
      await p.close();
      return names.join(", ");
    })() : "";
    await ctx.close();
    out.push(`${sw === "allow" ? "with" : "without"} service worker: first visit ready ${(first.readyAt / 1000).toFixed(1)} s ` +
      `(load ${(first.load / 1000).toFixed(1)} s), second ${(second.readyAt / 1000).toFixed(1)} s ` +
      `(load ${(second.load / 1000).toFixed(1)} s; core files: ${second.network} past the service worker, ${second.mb} MB, ` +
      `${second.fromSw} from it)` +
      (caches ? `; caches: ${caches}` : ""));
    if (sw === "allow") expect(second.fromSw).toBeGreaterThan(5);
    else expect(second.fromSw).toBe(0);
  }
  for (const line of out) log(line);
});
