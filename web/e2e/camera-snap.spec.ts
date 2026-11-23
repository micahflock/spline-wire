// The camera view with the snap-fit chain (data/chain-snap.yaml): rings, not
// dots. Found by the owner on a phone: with the snap preset chosen, the live
// view found nothing, because the overlay's frame detector knew only solid
// dots. Chromium's fake camera streams a rendered snap chain (camera_scene.py
// ... snap SPAN); the preset is chosen in settings, and the overlay must find
// all 13 rims, hold, take the still by itself, and the still must measure.
//
// At three distances (span: the chain's extent plus a link's width over the
// frame's short side; docs/open-items.md, live view resolution):
//   0.55  close, as before;
//   0.30  where the old 800 px work frame found none of the rims: the live
//         view tracks the chain in a region drawn finer (overlay/region.ts),
//         and the coach takes the still (its pins span 0.27 > minSpan 0.26);
//   0.22  every rim still found and tracked, but the coach asks to come closer
//         (its pins span 0.20 < 0.22, minSpan's keep) and takes no still.
// The first grabs frames on the page (?grab=main), the others in the worker
// (camera/grab.ts).
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices, expect, test, type Browser, type Page } from "@playwright/test";
import type { CameraStats } from "../src/phone/camera-view";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const BASE = "http://127.0.0.1:4173/";
const log = (...a: unknown[]) => console.log("[camera-snap]", ...a);
const med = (v: number[]) => (v.length ? [...v].sort((a, b) => a - b)[v.length >> 1]! : NaN);

let browser: Browser | null = null;

test.describe.configure({ mode: "serial" });
test.use({ actionTimeout: 30_000 });

test.afterEach(async () => {
  await browser?.close();
  browser = null;
});

/** A phone page whose fake camera streams the snap chain at `span` (with `dark`,
 * printed the other way round, and Colours set to match; seen `tilt` degrees
 * off straight-on, 12 by default). The live camera at full resolution
 * (Settings): Chromium's fake camera scales its portrait file to the fast
 * stream's landscape 1920×1440, where a phone gives a portrait 1440×1920
 * (camera/stream.ts); `fast` streams a landscape scene and leaves Settings as they are. */
async function phoneAt(span: number, query = "", dark = false, tilt?: number, fast = false, liveView?: string): Promise<Page> {
  const out = sceneDir(span, dark, tilt, fast);
  fs.mkdirSync(out, { recursive: true });
  execFileSync("uv", ["run", "python", "web/e2e/camera_scene.py", out, fast ? "4032x3024" : "3024x4032", "snap", String(span), ...(dark ? ["dark"] : []),
    ...(tilt !== undefined ? [`tilt=${tilt}`] : [])], { cwd: ROOT, stdio: "inherit" });
  browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
      `--use-file-for-fake-video-capture=${path.join(out, "camera.y4m")}`],
  });
  const context = await browser.newContext({ ...devices["Pixel 7"], baseURL: BASE, permissions: ["camera"] });
  // The snap preset chosen, and the scene's focal length saved, as a person would have it.
  await context.addInitScript(([shade, liveSize, liveView]) => localStorage.setItem("splinewire-phone:phone-settings",
    JSON.stringify({ savedFocal35: 24, chain: "snap", shade, liveSize, ...(liveView ? { liveView } : {}) })),
    [dark ? "dark" : "light", fast ? "fast" : "full", liveView ?? ""]);
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log("[page error]", e.message));
  await page.goto(`phone/${query}`);
  await expect(page.locator("#s-chain")).toHaveValue("snap");
  return page;
}

function sceneDir(span: number, dark = false, tilt?: number, fast = false): string {
  return path.resolve(ROOT, `web/test-results/camera-scene-snap-${span}${dark ? "-dark" : ""}${tilt !== undefined ? `-tilt${tilt}` : ""}${fast ? "-fast" : ""}`);
}

const stats = (page: Page) => page.evaluate(() => (window as unknown as { phone: { camera: CameraStats } }).phone.camera);

/** The camera view's per-frame cost and regions so far, as a line for the log. */
function costs(st: CameraStats): string {
  return `grab+gray ${med(st.grabMs).toFixed(1)} ms, blobs ${med(st.blobMs).toFixed(1)} ms, the page's own ${med(st.pageMs).toFixed(1)} ms,`
    + ` over ${st.frames} frames, grabbed: ${st.grab};`
    + ` regions: ${st.regions.full} full, ${st.regions.tile} quarter, ${st.regions.track} tracked (last ${st.region}); ${st.work}`;
}

async function measured(page: Page): Promise<void> {
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 120_000 });
  await expect(page.locator("#overlay circle[data-k]")).toHaveCount(13);
  log(await page.locator("#r-meta").textContent());
}

test("span 0.55: the live view finds the snap chain's rims, takes the still, and it measures (grabbing on the page, the video in colour)", async () => {
  // ?grab=main: the page grabs, as a browser without VideoFrame does (camera/grab.ts).
  const page = await phoneAt(0.55, "?grab=main", false, undefined, false, "color");
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam")).not.toHaveClass(/\bgrey\b|chain-only/);
  expect(await page.locator("#cam-video").evaluate((v) => getComputedStyle(v).filter)).toBe("none");
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  const st = await stats(page);
  log(`0.55: ${costs(st)}`);
  expect(st.grab).toMatch(/^main/);
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });       // the coach took the still
  await measured(page);
});

test("span 0.30: tracked in a finer region, every rim found, the still taken and measured (the video grey, by default)", async () => {
  const page = await phoneAt(0.3);
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  // Settings' default: the video shown grey and dimmed; only what is shown, the frames read keep their colour.
  await expect(page.locator("#cam")).toHaveClass(/\bgrey\b/);
  expect(await page.locator("#cam-video").evaluate((v) => getComputedStyle(v).filter)).toBe("grayscale(1) brightness(0.6)");
  await expect(page.locator("#cam-info")).toContainText("video grey");
  fs.writeFileSync(path.resolve(ROOT, "web/test-results/camera-grey.png"), await page.locator("#cam").screenshot());
  const st = await stats(page);
  expect(st.shows).toBe("grey");
  log(`0.30: ${costs(st)}`);
  expect(st.regions.track).toBeGreaterThan(0);
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });
  await measured(page);
});

test("span 0.22: every rim found and tracked, and the coach asks to come closer", async () => {
  const page = await phoneAt(0.22);
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  await expect(page.locator("#cam-say-title")).toHaveText("Move closer", { timeout: 30_000 });
  // Watch it for a while: tracked throughout, all 13 pins, no still.
  const before = await stats(page);
  await page.waitForTimeout(3000);
  const st = await stats(page);
  log(`0.22: ${costs(st)}`);
  expect(st.grab).toBe("worker");                         // Chromium has VideoFrame: the worker grabs
  expect(st.regions.track - before.regions.track).toBeGreaterThan(10);
  expect(st.regions.full + st.regions.tile).toBe(before.regions.full + before.regions.tile);   // never lost
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins");
  await expect(page.locator("#cam-info")).toContainText("track");
  expect(await page.locator("#photos li").count()).toBe(0);
  await page.screenshot({ path: path.resolve(ROOT, "web/test-results/camera-scene-snap-0.22/too-far.png") });
});

test("dark rims on light links (Colours: dark): tracked in the worker, every rim found, the still taken and measured", async () => {
  const page = await phoneAt(0.3, "", true);
  await expect(page.locator("#s-shade")).toHaveValue("dark");
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  const st = await stats(page);
  log(`dark 0.30: ${costs(st)}`);
  expect(st.regions.track).toBeGreaterThan(0);
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });     // the coach took the still
  await measured(page);
});

test("tilted 32 degrees: no still, and an arrow the way to move the phone", async () => {
  const page = await phoneAt(0.4, "", false, 32);
  const truth = JSON.parse(fs.readFileSync(path.join(sceneDir(0.4, false, 32), "camera.json"), "utf-8")) as { move_dir: [number, number] };
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  await expect(page.locator("#cam-say-title")).toHaveText("Tilt the phone flatter over the chain", { timeout: 30_000 });
  await expect(page.locator("#cam-say-detail")).toHaveText(/arrow/);
  await expect.poll(async () => (await stats(page)).arrow, { timeout: 10_000 }).not.toBeNull();
  await page.waitForTimeout(1500);
  const arrow = (await stats(page)).arrow!;
  const off = (Math.acos(Math.min(1, arrow[0] * truth.move_dir[0] + arrow[1] * truth.move_dir[1])) * 180) / Math.PI;
  log(`tilt 32: arrow (${arrow.map((v) => v.toFixed(3)).join(", ")}), truth (${truth.move_dir.map((v) => v.toFixed(3)).join(", ")}): ${off.toFixed(1)} degrees off`);
  expect(off).toBeLessThan(15);
  expect(await page.locator("#photos li").count()).toBe(0);                // too tilted for the still
  await page.screenshot({ path: path.join(sceneDir(0.4, false, 32), "arrow.png") });
});

test("the fast stream: 1920×1440 once the focal length is known, every rim tracked, the still taken and measured", async () => {
  // A landscape scene (a camera whose frames come across, as a desktop webcam's
  // do): the fast stream's 1920×1440 is then the same picture, scaled.
  const page = await phoneAt(0.4, "", false, undefined, true);
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  const st = await stats(page);
  log(`fast: ${st.size}, ${costs(st)}; ${st.canvas}`);
  expect(st.size).toBe("1920×1440 (fast)");
  expect(st.regions.track).toBeGreaterThan(0);
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });     // the coach took the still
  const after = await stats(page);
  log(`fast: the still ${after.still}, ${after.stillSize}`);
  expect(after.stillSize).toMatch(/^\d+×\d+$/);
  // A still smaller than the full sensor puts this camera back to full size (Settings.fullStreamCameras).
  const [w, h] = after.stillSize!.split("×").map(Number) as [number, number];
  const refused = await page.evaluate(() => JSON.parse(localStorage.getItem("splinewire-phone:phone-settings")!).fullStreamCameras as string[]);
  expect(refused.length).toBe(Math.max(w, h) < 0.9 * 4032 ? 1 : 0);
  await measured(page);
});

test("Live camera shows only the chain: no video seen, every rim still tracked, the still taken and measured", async () => {
  const page = await phoneAt(0.3, "", false, undefined, false, "chain");
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toHaveClass(/chain-only/);
  await expect(page.locator("#cam")).not.toHaveClass(/\bgrey\b/);
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  // The video plays under the overlay, which is opaque: what the screen shows is the overlay alone.
  expect(await page.locator("#cam-overlay").evaluate((c) => getComputedStyle(c).backgroundColor)).toBe("rgb(0, 0, 0)");
  expect(await page.locator("#cam-video").evaluate((v) => (v as HTMLVideoElement).paused)).toBe(false);
  const shot = await page.locator("#cam").screenshot();
  fs.writeFileSync(path.resolve(ROOT, "web/test-results/camera-chain-only.png"), shot);
  const st = await stats(page);
  log(`chain only: ${st.loop}; ${costs(st)}`);
  expect(st.shows).toBe("chain");
  expect(st.loop).toBe("video frames");
  await expect(page.locator("#cam-info")).toContainText("video hidden");
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });     // the coach took the still
  await measured(page);
});
