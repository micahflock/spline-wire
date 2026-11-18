// The camera view with the snap-fit chain (data/chain-snap.yaml): rings, not
// dots. Found by the owner on a phone: with the snap preset chosen, the live
// view found nothing, because the overlay's frame detector knew only solid
// dots. Chromium's fake camera streams a rendered snap chain (camera_scene.py
// ... snap); the preset is chosen in settings, and the overlay must find all
// 13 rims, hold, take the still by itself, and the still must measure.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices, expect, test, type Browser, type Page } from "@playwright/test";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.resolve(ROOT, "web/test-results/camera-scene-snap");
const BASE = "http://127.0.0.1:4173/";
const log = (...a: unknown[]) => console.log("[camera-snap]", ...a);

let browser: Browser;
let page: Page;

test.describe.configure({ mode: "serial" });
test.use({ actionTimeout: 30_000 });

test.beforeAll(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  execFileSync("uv", ["run", "python", "web/e2e/camera_scene.py", OUT, "3024x4032", "snap"], { cwd: ROOT, stdio: "inherit" });
  browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
      `--use-file-for-fake-video-capture=${path.join(OUT, "camera.y4m")}`],
  });
  const context = await browser.newContext({ ...devices["Pixel 7"], baseURL: BASE, permissions: ["camera"] });
  // The snap preset chosen, and the scene's focal length saved, as a person would have it.
  await context.addInitScript(() => localStorage.setItem("splinewire-phone:phone-settings",
    JSON.stringify({ savedFocal35: 24, chain: "snap" })));
  page = await context.newPage();
  page.on("pageerror", (e) => console.log("[page error]", e.message));
});

test.afterAll(async () => {
  await browser?.close();
});

test("the live view finds the snap chain's rims, takes the still, and it measures", async () => {
  await page.goto("phone/");
  await expect(page.locator("#s-chain")).toHaveValue("snap");
  await page.locator("#take-camera").click();
  await expect(page.locator("#cam")).toBeVisible();
  await expect(page.locator("#cam-count")).toHaveText("13 of 13 pins", { timeout: 30_000 });
  await expect(page.locator("#cam")).toBeHidden({ timeout: 30_000 });       // the coach took the still
  const item = page.locator("#photos li").first();
  await expect(item).toHaveAttribute("data-status", "done", { timeout: 120_000 });
  await expect(page.locator("#overlay circle[data-k]")).toHaveCount(13);
  log(await page.locator("#r-meta").textContent());
});
