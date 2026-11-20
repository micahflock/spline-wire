// Pairing from the live camera: the phone page's camera view, pointed at the
// desktop page's pairing code, pairs without leaving the camera.
//
// Chromium's fake camera streams one frame drawn here: the code as the desktop
// page draws it (desktop/qr.ts: level M, 4 modules of quiet zone), a quarter of
// the frame wide and turned 8°, on a dark screen, at the full sensor size an
// iPhone streams (3024×4032). Scaled down to qr-scan.ts's 1024 px that is about
// 4 px a module: a code on a screen at arm's length.
//
//   npm run e2e         (after npm run assets)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices, expect, test, type Browser } from "@playwright/test";
import qrcode from "qrcode-generator";
import type { CameraStats } from "../src/phone/camera-view";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = path.resolve(ROOT, "web/test-results/camera-qr");
const BASE = "http://127.0.0.1:4173/";
const SECRET = "UkypDe-UM7GtIrxg48MQHw";

/** A one-frame YUV4MPEG2 file (4:2:0, gray) of the pairing code `text`. */
function codeVideo(file: string, text: string, w = 3024, h = 4032): void {
  const qr = qrcode(0, "M");
  qr.addData(text, "Byte");
  qr.make();
  const n = qr.getModuleCount(), size = n + 8;
  const side = 0.25 * w, cell = side / size;
  const cx = w / 2, cy = h * 0.45, a = (8 * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
  /** Brightness at a point: the screen, the white card, a dark module. */
  const at = (x: number, y: number) => {
    const u = ((x - cx) * cos + (y - cy) * sin) / cell + size / 2;
    const v = (-(x - cx) * sin + (y - cy) * cos) / cell + size / 2;
    if (u < 0 || v < 0 || u >= size || v >= size) return 45;
    const c = Math.floor(u) - 4, r = Math.floor(v) - 4;
    return c >= 0 && r >= 0 && c < n && r < n && qr.isDark(r, c) ? 35 : 220;
  };
  const y = Buffer.alloc(w * h);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      // 2×2 samples a pixel, so module edges are soft as a camera sees them.
      y[j * w + i] = (at(i + 0.25, j + 0.25) + at(i + 0.75, j + 0.25) + at(i + 0.25, j + 0.75) + at(i + 0.75, j + 0.75)) / 4;
    }
  }
  const chroma = Buffer.alloc((w / 2) * (h / 2) * 2, 128);
  fs.writeFileSync(file, Buffer.concat([Buffer.from(`YUV4MPEG2 W${w} H${h} F30:1 Ip A1:1 C420jpeg\nFRAME\n`), y, chroma]));
}

let browser: Browser | undefined;
test.afterAll(async () => browser?.close());

test("the live camera pairs with the computer whose code it sees, and stays open", async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const video = path.join(OUT, "camera.y4m");
  codeVideo(video, `${BASE}phone/#pair=${SECRET}`);
  browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-video-capture=${video}`],
  });
  const context = await browser.newContext({ ...devices["Pixel 7"], baseURL: BASE, permissions: ["camera"] });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log("[page error]", e.message));
  await page.goto("phone/");
  await expect(page.locator("#pair-text")).toContainText("Not paired");
  await page.locator("#take-camera").click();
  await expect(page.locator("#toast")).toHaveText("Paired with the computer.", { timeout: 20_000 });
  expect(await page.evaluate(() => localStorage.getItem("splinewire-phone:pairing"))).toBe(SECRET);
  await expect(page.locator("#cam")).toBeVisible();                 // still measuring: the camera stays open
  expect(page.url()).toBe(`${BASE}phone/`);                          // and the address is left alone
  const st = await page.evaluate(() => (window as unknown as { phone: { camera: CameraStats } }).phone.camera);
  console.log("[camera-qr]", `qr: ${st.qr}; page ${st.pageMs.length ? Math.max(...st.pageMs).toFixed(1) : "–"} ms a frame at most`);
  await page.locator("#cam-close").click();
  await expect(page.locator("#pair-text")).toContainText("Paired");
});
