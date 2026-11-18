// The phone page's camera view end to end (web-build.md step 5), at a phone's
// size in Chromium, with its fake camera streaming a rendered chain
// (e2e/camera_scene.py: 3024×4032 portrait, the full sensor an iPhone streams):
//
//   - the overlay says "point at the chain" for a chain it cannot order, names
//     the problem when pins are missing, and with the right pin count holds
//     steady and takes the still itself; the still is measured and its pins
//     match native Python on the same bytes;
//   - the cost per frame and the page's responsiveness while it runs;
//   - Live camera / Upload a photo: upload releases every track, live opens
//     the stream again, the choice is remembered;
//   - a camera that is not allowed falls back to the system camera's file input.
//
//   npm run e2e         (after npm run assets)
//
// Numbers are printed as "[camera] ..." lines; screenshots of the camera view
// go to test-results/camera/ (or CAMERA_SHOTS).

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices, expect, test, type Browser, type BrowserContext, type Page } from "@playwright/test";
import type { Point } from "../src/core/types";
import type { CameraStats, CameraView } from "../src/phone/camera-view";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.resolve(ROOT, "web/test-results/camera-scene");
const SHOTS = process.env.CAMERA_SHOTS ?? path.resolve(ROOT, "web/test-results/camera");
const PHONE = devices["Pixel 7"];
/** As e2e/phone.spec.ts: the browser's OpenCV against native, pins drawn rounded to 0.01 px. */
const PX_TOLERANCE = 0.06;
const BASE = "http://127.0.0.1:4173/";

interface Scene { size: [number, number]; pins_px: Point[]; focal_35mm: number; chain: Record<string, unknown> }
interface PhoneWindow {
  phone: { engine: unknown; camera: CameraStats; cameraView: CameraView; timings: { status: string; ms: number }[] };
  __streams: MediaStream[];
}

const log = (...a: unknown[]) => console.log("[camera]", ...a);
const med = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[v.length >> 1]! : NaN);
const p95 = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[Math.floor(0.95 * (v.length - 1))]! : NaN);

let scene: Scene;
let browser: Browser;
let context: BrowserContext;
let page: Page;

test.describe.configure({ mode: "serial" });
test.use({ actionTimeout: 30_000 });

/** Every stream the page opens, so the test can tell that each track was stopped. */
const recordStreams = () => {
  const w = window as unknown as PhoneWindow;
  w.__streams = [];
  const md = navigator.mediaDevices;
  if (!md?.getUserMedia) return;
  const real = md.getUserMedia.bind(md);
  md.getUserMedia = async (c) => {
    const s = await real(c);
    w.__streams.push(s);
    return s;
  };
};

/** Settings as the phone page keeps them (settings.ts): the scene's focal length saved. */
/** The phone page's saved settings for these tests: the O-ring dot chain, which
 * this file's camera scene draws (the snap-fit chain, the default, has camera-snap.spec.ts). */
function settings(extra: Record<string, unknown> = {}): string {
  return JSON.stringify({ chain: "oring", savedFocal35: 24, ...extra });
}

test.beforeAll(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(SHOTS, { recursive: true });
  execFileSync("uv", ["run", "python", "web/e2e/camera_scene.py", OUT], { cwd: ROOT, stdio: "inherit" });
  scene = JSON.parse(fs.readFileSync(path.join(OUT, "camera.json"), "utf8")) as Scene;
  browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
      `--use-file-for-fake-video-capture=${path.join(OUT, "camera.y4m")}`],
  });
  context = await browser.newContext({ ...PHONE, baseURL: BASE, permissions: ["camera"] });
  await context.addInitScript(recordStreams);
  await context.addInitScript((s) => localStorage.getItem("splinewire-phone:phone-settings") ?? localStorage.setItem("splinewire-phone:phone-settings", s), settings());
  page = await context.newPage();
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[page error]", e.message));
});

test.afterAll(async () => {
  await context?.close();
  await browser?.close();
});

const stats = () => page.evaluate(() => (window as unknown as PhoneWindow).phone.camera);
const liveTracks = () => page.evaluate(() => (window as unknown as PhoneWindow).__streams
  .flatMap((s) => s.getTracks()).filter((t) => t.readyState === "live").length);

async function setChain(key: string, pins: number): Promise<void> {
  await page.evaluate(([key, pins]) => {
    const select = document.getElementById("s-chain") as HTMLSelectElement;
    select.value = key as string;
    select.dispatchEvent(new Event("change"));
    const input = document.querySelector("#s-pins input") as HTMLInputElement;
    input.value = String(pins);
    input.dispatchEvent(new Event("change"));
  }, [key, pins] as const);
}

async function openCamera(): Promise<void> {
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-shutter")).toBeEnabled({ timeout: 20_000 });
}

async function closeCamera(): Promise<void> {
  await page.locator("#cam-close").click();
  await expect(page.locator("#cam")).toBeHidden();
  expect(await liveTracks()).toBe(0);
}

test("the overlay coaches: no chain, pins missing", async () => {
  await page.goto("phone/");
  await expect.poll(() => page.evaluate(() => (window as unknown as PhoneWindow).phone.engine !== null), { timeout: 180_000 }).toBe(true);

  // The snap-fit chain's pitch and fiducials: these dots are not that chain.
  await setChain("snap", 11);
  await openCamera();
  await expect(page.locator("#cam-say-title")).toHaveText("Point the camera at the chain", { timeout: 30_000 });
  await page.screenshot({ path: path.join(SHOTS, "searching.png") });
  const first = await stats();
  log(`opened ${first.camera} at ${first.size}; ImageCapture ${first.imageCapture ? "yes" : "no"}; focal ${first.focal}`);
  expect(first.size).toBe(`${scene.size[0]}×${scene.size[1]}`);
  await closeCamera();

  // Two pins more than the chain has: the coach asks for the rest, and never shoots.
  await setChain("oring", 15);
  await openCamera();
  await expect(page.locator("#cam-say-title")).toHaveText("Part of the chain is out of view", { timeout: 30_000 });
  await expect(page.locator("#cam-count")).toHaveText("13 of 15 pins");
  await page.screenshot({ path: path.join(SHOTS, "adjusting.png") });

  // The performance budget, measured while it runs: the overlay's cost per
  // frame, the analyses, and how late a 16 ms timer fires (the page's responsiveness).
  const before = await stats();
  const lag = await page.evaluate(async () => {
    const late: number[] = [];
    const end = performance.now() + 5000;
    while (performance.now() < end) {
      const t = performance.now();
      await new Promise((r) => setTimeout(r, 16));
      late.push(performance.now() - t - 16);
    }
    return late;
  });
  const after = await stats();
  const frames = after.frames - before.frames, analysed = after.analysed - before.analysed;
  log(`work canvas: ${after.work}`);
  log(`per frame over 5 s: grab+gray ${med(after.grabMs).toFixed(1)} ms (p95 ${p95(after.grabMs).toFixed(1)}),`,
    `blobs ${med(after.blobMs).toFixed(1)} ms (p95 ${p95(after.blobMs).toFixed(1)}); ${(frames / 5).toFixed(1)} frames/s worked,`,
    `${after.skipped - before.skipped} skipped; ${(analysed / 5).toFixed(1)} analyses/s, analysis ${med(after.analyseMs).toFixed(0)} ms`,
    `(p95 ${p95(after.analyseMs).toFixed(0)}); 16 ms timer late by median ${med(lag).toFixed(1)} ms, p95 ${p95(lag).toFixed(1)}, max ${Math.max(...lag).toFixed(1)}`);
  // Where the grab's time goes, and whether a CPU-backed canvas would be cheaper here.
  const parts = await page.evaluate(() => {
    const v = document.getElementById("cam-video") as HTMLVideoElement;
    const w = Math.round(v.videoWidth / 6.3), h = Math.round(v.videoHeight / 6.3);
    const time = (cpu: boolean) => {
      const c = new OffscreenCanvas(w, h);
      const g = c.getContext("2d", { willReadFrequently: cpu })!;
      g.imageSmoothingQuality = "high";
      let draw = 0, read = 0;
      for (let i = 0; i < 15; i++) {
        const t0 = performance.now();
        g.drawImage(v, 0, 0, w, h);
        const t1 = performance.now();
        g.getImageData(0, 0, w, h);
        read += performance.now() - t1;
        draw += t1 - t0;
      }
      return [draw / 15, read / 15];
    };
    return { gpu: time(false), cpu: time(true) };
  });
  log(`grab parts at ${await page.evaluate(() => { const v = document.getElementById("cam-video") as HTMLVideoElement; return `${v.videoWidth}×${v.videoHeight}`; })}:`,
    `GPU canvas drawImage ${parts.gpu[0]!.toFixed(1)} ms + getImageData ${parts.gpu[1]!.toFixed(1)} ms;`,
    `CPU canvas (willReadFrequently) ${parts.cpu[0]!.toFixed(1)} + ${parts.cpu[1]!.toFixed(1)} ms`);
  expect(analysed).toBeGreaterThan(5);
  expect(p95(lag)).toBeLessThan(150);           // a tap is still answered at once; CI has no GPU either
  expect(await page.locator("#photos li").count()).toBe(0);           // pins missing: no shot
  await closeCamera();
});

test("with the right pin count it holds steady, takes the still, and measures it as native Python does", async () => {
  await setChain("oring", 13);
  await openCamera();
  // Catch the hold while the ring fills (700 ms of good readings).
  await expect(page.locator("#cam-shutter")).toHaveAttribute("data-phase", "steady", { timeout: 30_000 });
  await page.waitForFunction(() => {
    const ring = document.getElementById("cam-ring");
    return !!ring && Number(ring.style.strokeDashoffset) < 0.6 * 2 * Math.PI * 36;
  }, null, { polling: "raf", timeout: 5000 }).catch(() => {});
  await page.screenshot({ path: path.join(SHOTS, "steady.png") });
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });     // the still was taken, the camera released
  expect(await liveTracks()).toBe(0);
  const st = await stats();
  log(`still: ${st.still}; takePhoto ${st.takePhotoMs ?? "-"} ms${st.takePhotoError ? `, failed: ${st.takePhotoError}` : ""}`);

  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 120_000 });
  await expect(page.locator("#overlay circle[data-k]")).toHaveCount(13);
  await expect(page.locator("#r-meta")).toContainText("page camera");
  await expect(page.locator("#r-meta")).toContainText("live view scale");
  const meta = await page.locator("#r-meta").textContent();
  log(`photo: ${meta}`);

  // The still's bytes, from the page's photo store, measured natively with the same focal length.
  const id = (await item.getAttribute("data-id"))!;
  const got = await page.evaluate(async (id) => {
    const db = await new Promise<IDBDatabase>((res, rej) => {
      const r = indexedDB.open("splinewire-phone");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
    const get = <T,>(store: string, key: string) => new Promise<T>((res, rej) => {
      const r = db.transaction(store).objectStore(store).get(key);
      r.onsuccess = () => res(r.result as T);
      r.onerror = () => rej(r.error);
    });
    const blob = await get<Blob>("blobs", `${id}:original`);
    const rec = await get<{ name: string; focal?: { focal35: number } | null; doc: { result: { focal_source: string; focal_35mm: number } } }>("photos", id);
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return { b64: btoa(bin), name: rec.name, focal: rec.focal?.focal35 ?? null, source: rec.doc.result.focal_source, f35: rec.doc.result.focal_35mm };
  }, id);
  const still = path.join(OUT, got.name);
  fs.writeFileSync(still, Buffer.from(got.b64, "base64"));
  const native = JSON.parse(execFileSync("uv", ["run", "python", "-c", [
    "import json, sys",
    "from splinewire.session import PhotoSession",
    "s = PhotoSession.open(open(sys.argv[1], 'rb').read(), 'still')",
    "focal = None if sys.argv[3] == 'null' else float(sys.argv[3])",
    "d = s.measure({'chain': json.loads(sys.argv[2]), 'saved_focal_35mm': 24, 'focal_override_35mm': focal})",
    "print(json.dumps({'pins': [[p['x'], p['y']] for p in d['view']['pins']], 'size': [s.size[0], s.size[1]]}))",
  ].join("\n"), still, JSON.stringify(scene.chain), String(got.focal)], { cwd: ROOT, encoding: "utf8" }).trim().split("\n").at(-1)!) as { pins: Point[]; size: [number, number] };
  const px = await page.evaluate(() => [...document.querySelectorAll<SVGCircleElement>("#overlay circle[data-k]")]
    .sort((a, b) => Number(a.dataset.k) - Number(b.dataset.k))
    .map((c): [number, number] => [Number(c.getAttribute("cx")), Number(c.getAttribute("cy"))]));
  const along = (pins: Point[]) => Math.max(...px.map((p, i) => Math.hypot(p[0] - pins[i]![0], p[1] - pins[i]![1])));
  const fwd = along(native.pins), rev = along([...native.pins].reverse());
  if (rev < fwd) log(`the page's chain runs the other way from native Python's (${fwd.toFixed(1)} px forward)`);
  const worst = Math.min(fwd, rev);
  // And against the scene's truth, when the still is the frame's own size.
  const truth = native.size[0] === scene.size[0] && native.size[1] === scene.size[1]
    ? Math.max(...px.map((p) => Math.min(...scene.pins_px.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1]))))) : NaN;
  log(`still ${native.size.join("×")}, focal ${got.f35} mm (${got.source}${got.focal ? ", from the frame" : ""});`,
    `pins ${worst.toFixed(3)} px from native Python on the same bytes, ${truth.toFixed(2)} px from the rendered truth`);
  expect(worst).toBeLessThan(PX_TOLERANCE);
});

test("without ImageCapture (Firefox) the still is the video frame, measured with the saved focal length and marked so", async () => {
  await page.evaluate(() => delete (window as { ImageCapture?: unknown }).ImageCapture);
  const before = await page.locator("#photos li").count();
  await openCamera();
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });
  // The photo joins the list once its preview is encoded. Chromium encodes a
  // canvas's toBlob in idle time: it waits up to 1 s for an idle period, then
  // up to 5 s more before it forces the encode, so in a busy headless run the
  // photo can arrive 7-8 s after the camera closes (measured 2026-11-18).
  await expect(page.locator("#photos li")).toHaveCount(before + 1, { timeout: 30_000 });
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 120_000 });
  await expect(page.locator("#overlay circle[data-k]")).toHaveCount(13, { timeout: 30_000 });
  await expect(page.locator("#r-meta")).toContainText("page camera, video frame");
  await expect(page.locator("#numbers")).toContainText("your saved focal length (a video frame has no camera data)");
  await expect(page.locator("#numbers")).toContainText("24.0 mm");
  log(`frame still: ${(await stats()).still}; ${await page.locator("#r-meta").textContent()}`);
});

test("Upload a photo turns the camera off, Live camera opens it again, and the choice is kept", async () => {
  await expect(page.locator("#mode-live")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#mode-upload").click();
  await expect(page.locator("#upload-box")).toBeVisible();
  await expect(page.locator("#live-box")).toBeHidden();
  await expect(page.locator("#upload-box")).toContainText("Library photos keep the focal length");
  const opened = await page.evaluate(() => (window as unknown as PhoneWindow).__streams.length);
  await page.locator("#mode-live").click();                     // opens the camera at once
  await expect(page.locator("#cam")).toBeVisible();
  await expect.poll(liveTracks).toBe(1);
  expect(await page.evaluate(() => (window as unknown as PhoneWindow).__streams.length)).toBe(opened + 1);
  await page.locator("#cam-upload").click();                    // from inside the camera view
  await expect(page.locator("#cam")).toBeHidden();
  expect(await liveTracks()).toBe(0);
  await expect(page.locator("#mode-upload")).toHaveAttribute("aria-pressed", "true");
  await page.screenshot({ path: path.join(SHOTS, "upload-mode.png") });
  await page.reload();
  await expect(page.locator("#mode-upload")).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#upload-box")).toBeVisible();
  expect(await liveTracks()).toBe(0);
  await page.locator("#mode-live").click();
  await expect.poll(liveTracks).toBe(1);
  await page.locator("#cam-close").click();
  expect(await liveTracks()).toBe(0);
});

test("a camera that is not allowed falls back to the system camera", async () => {
  // Chromium's stand-in for the permission prompt, answering no: getUserMedia fails with NotAllowedError.
  const denied = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream=deny"] });
  try {
    const ctx = await denied.newContext({ ...PHONE, baseURL: BASE });
    await ctx.addInitScript((s) => localStorage.getItem("splinewire-phone:phone-settings") ?? localStorage.setItem("splinewire-phone:phone-settings", s), settings());
    const p = await ctx.newPage();
    await p.goto("phone/");
    await p.locator("#take-camera").click();
    await expect(p.locator("#cam-fail")).toBeVisible({ timeout: 20_000 });
    log(`denied: ${await p.locator("#cam-fail-title").textContent()} ${await p.locator("#cam-fail-text").textContent()}`);
    await expect(p.locator("#cam-fail-title")).toHaveText("The camera wasn't allowed.");
    await expect(p.locator("#cam-fail")).toContainText("lose the focal length");
    await p.screenshot({ path: path.join(SHOTS, "denied.png") });
    const chooser = p.waitForEvent("filechooser");
    await p.locator("#cam-fail-system").click();
    const fc = await chooser;
    expect(await fc.element().getAttribute("capture")).toBe("environment");
    await fc.setFiles(path.join(OUT, "camera.jpg"));
    await expect(p.locator("#cam")).toBeHidden();
    await expect(p.locator("#mode-upload")).toHaveAttribute("aria-pressed", "true");
    const item = p.locator("#photos li").first();
    await expect(item).toHaveAttribute("data-status", "done", { timeout: 180_000 });
    await expect(p.locator("#overlay circle[data-k]")).toHaveCount(13);
    await expect(p.locator("#r-meta")).toContainText("system camera");
    await expect(p.locator("#numbers")).toContainText("from the photo");
    await ctx.close();
  } finally {
    await denied.close();
  }
});
