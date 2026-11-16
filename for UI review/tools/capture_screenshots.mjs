// Re-takes every screenshot in "for UI review/screenshots".
//
//   uv run python "for UI review/tools/make_sample_photos.py" <photos-dir>
//   (start the app on a scratch workspace; see tools/README.md)
//   SW_PHOTOS=<photos-dir> node "for UI review/tools/capture_screenshots.mjs"
//
// Needs Playwright (require('playwright')) and a running app. Environment:
//   SW_URL     the app as the computer sees it       (default http://localhost:8765)
//   SW_PHONE   the app at a non-loopback address     (default http://192.0.2.2:8765)
//              The server treats any non-loopback client as "the phone", so the
//              phone page is shot through this address.
//   SW_PHOTOS  folder written by make_sample_photos.py
//   SW_OUT     where the PNGs go                      (default ../screenshots)
import { createRequire } from "node:module";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium, devices } = require("playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
const URL_ = process.env.SW_URL || "http://localhost:8765";
const PHONE = process.env.SW_PHONE || "http://192.0.2.2:8765";
const PHOTOS = process.env.SW_PHOTOS;
const OUT = process.env.SW_OUT || path.join(here, "..", "screenshots");
if (!PHOTOS) throw new Error("set SW_PHOTOS to the folder make_sample_photos.py wrote");
fs.mkdirSync(OUT, { recursive: true });

const photo = (n) => path.join(PHOTOS, n);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const token = (await (await fetch(`${URL_}/api/local-token`)).json()).token;

const browser = await chromium.launch();
const shots = [];
async function shot(page, name, opts = {}) {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file, ...opts });
  shots.push(name);
  console.log("  ", name);
}

// Wait until the app has nothing left to measure.
async function settled(page, n) {
  await page.waitForFunction(
    (n) => state && state.photos.length >= n && state.photos.every((p) => p.status === "done" || p.status === "error"),
    n, { timeout: 120000 });
  await sleep(500);
}

// ------------------------------------------------------------------ desktop
console.log("desktop");
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: "light",
  permissions: ["clipboard-read", "clipboard-write"],
});
const page = await ctx.newPage();
await page.goto(URL_);
await page.waitForFunction(() => state && state.photos !== undefined);
await sleep(600);
await shot(page, "d01-empty-no-photos");

// Photos go in oldest first; the newest is selected, like a phone upload would leave it.
const uploads = [
  "pipe-steep.jpg", "s-curve-glare.jpg", "kitchen-table-no-chain.jpg", "cove-wood.jpg",
  "s-curve-bare-chain.jpg", "pipe-clutter.jpg",
];
for (let i = 0; i < uploads.length; i++) {
  await page.setInputFiles("#file", photo(uploads[i]));
  await settled(page, i + 1);
}
await page.waitForSelector(".views svg.editor");
await sleep(800);

const select = async (name) => {
  await page.click(`.item:has(.name:text-is("${name}"))`);
  await page.waitForSelector("#editor-slot svg.editor");
  await sleep(700);
};

await shot(page, "d02-photo-measured-viewport");
await shot(page, "d03-photo-measured-full-page", { fullPage: true });

// Pin editor: hover a pin, remove it, then add one by clicking empty space.
const pinScreen = (k) => page.evaluate((k) => {
  const m = viewer.view.pins[k], pt = viewer.svg.createSVGPoint();
  pt.x = m.x; pt.y = m.y;
  const q = pt.matrixTransform(viewer.svg.getScreenCTM());
  return { x: q.x, y: q.y };
}, k);
let p = await pinScreen(5);
await page.mouse.move(p.x, p.y);
await sleep(200);
await shot(page, "d04-editor-hover-pin", { clip: await clipOf(page, ".views") });
await page.mouse.click(p.x, p.y);
await sleep(1800);
await shot(page, "d05-editor-pin-removed-gap", { clip: await clipOf(page, ".views") });
await page.click("#ed-reset");
await sleep(1800);

// A pin added by hand, away from any dot: it does not fit the chain, and the hint says so.
const empty = await page.evaluate(() => {
  const v = viewer.view, pt = viewer.svg.createSVGPoint();
  const m = v.pins[3]; pt.x = m.x + v.r * 6; pt.y = m.y - v.r * 6;
  const q = pt.matrixTransform(viewer.svg.getScreenCTM());
  return { x: q.x, y: q.y };
});
await page.mouse.click(empty.x, empty.y);
await sleep(1800);
await page.mouse.move(empty.x, empty.y);
await shot(page, "d05b-editor-added-pin-that-does-not-fit", { clip: await clipOf(page, ".views") });
await page.click("#ed-undo");
await sleep(1800);

// Pick the object side on the photo.
await page.click("#opt-pick");
await sleep(300);
p = await pinScreen(6);
await page.mouse.move(p.x + 60, p.y + 40);
await sleep(200);
await shot(page, "d06-pick-side-mode");
await page.keyboard.press("Escape");
await sleep(300);
await page.click("#opt-flip");
await sleep(2200);
await shot(page, "d07-side-flipped", { clip: await clipOf(page, ".title,.views", true) });
await page.click("#opt-flip");
await sleep(2200);

// Copy points -> toast.
await page.click("#copy");
await sleep(300);
await shot(page, "d08-copy-points-toast");
await sleep(3200);       // let the toast fade so it is not in the shots that follow

// Same view on a photo whose background has many look-alike dots: "Zoom to chain" frames the stray
// detections too, so the chain stays small.
await select("s-curve-bare-chain.jpg");
await shot(page, "d08b-zoom-to-chain-with-stray-detections", { clip: await clipOf(page, ".views") });

// Other photos in the list: a lamp reflection that defeats the detector, a photo with no chain, a steep tilt, a concave cove.
await select("s-curve-glare.jpg");
await shot(page, "d09-photo-error-glare-washed-out-dots", { fullPage: true });
await select("kitchen-table-no-chain.jpg");
await shot(page, "d10-photo-error-no-chain-in-photo", { fullPage: true });
await select("pipe-steep.jpg");
await shot(page, "d11-photo-steep-tilt", { fullPage: true });
await select("cove-wood.jpg");
await shot(page, "d12-photo-concave-cove", { fullPage: true });

// Settings dialog.
await page.click("#settings-btn");
await sleep(400);
await shot(page, "d13-settings-dialog");
await page.setViewportSize({ width: 1440, height: 1150 });     // the dialog scrolls in a 900 px window
await sleep(300);
await shot(page, "d13b-settings-dialog-whole");
await page.setViewportSize({ width: 1440, height: 900 });
await page.keyboard.press("Escape");
await sleep(200);

// Drag-and-drop overlay (forced on).
await page.evaluate(() => document.getElementById("drop").classList.add("show"));
await shot(page, "d14-drop-overlay");
await page.evaluate(() => document.getElementById("drop").classList.remove("show"));

// Firewall warning in the phone panel (mocked: the real check only runs on Windows).
await page.route("**/api/state", async (route) => {
  const res = await route.fetch();
  const body = await res.json();
  if (body.network) body.network = { phone_seen: null, firewall: { ok: false, busy: false, blocking_rules: true,
    networks: [{ name: "Home Wi-Fi", category: "Public" }] } };
  await route.fulfill({ response: res, json: body });
});
await page.reload();
await page.waitForFunction(() => state && state.photos.length);
await sleep(1500);
await shot(page, "d15-phone-panel-firewall-warning-mocked", { clip: { x: 0, y: 0, width: 420, height: 900 } });
await page.click(".net summary");
await sleep(200);
await shot(page, "d16-phone-panel-help-expanded-mocked", { clip: { x: 0, y: 0, width: 420, height: 900 } });
await page.unroute("**/api/state");

// "Phone connected" note (mocked).
await page.route("**/api/state", async (route) => {
  const res = await route.fetch();
  const body = await res.json();
  if (body.network) body.network = { phone_seen: Date.now() / 1000, firewall: { ok: true, busy: false } };
  await route.fulfill({ response: res, json: body });
});
await page.reload();
await page.waitForFunction(() => state && state.photos.length);
await sleep(1500);
await shot(page, "d17-phone-panel-connected-mocked", { clip: { x: 0, y: 0, width: 420, height: 420 } });
await page.unroute("**/api/state");
await ctx.close();

// ------------------------------------------------------------- desktop, dark
console.log("desktop, dark mode");
const dark = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "dark" });
const dp = await dark.newPage();
await dp.goto(URL_);
await dp.waitForFunction(() => state && state.photos.length);
await sleep(300);
await dp.click(`.item:has(.name:text-is("pipe-clutter.jpg"))`);
await dp.waitForSelector("#editor-slot svg.editor");
await sleep(900);
await shot(dp, "d18-dark-photo-measured-full-page", { fullPage: true });
await dp.click("#settings-btn");
await sleep(300);
await shot(dp, "d19-dark-settings-dialog");
await dark.close();

// ------------------------------------------------------- desktop, narrow window
console.log("desktop, narrow window");
const narrow = await browser.newContext({ viewport: { width: 820, height: 1000 }, colorScheme: "light" });
const np = await narrow.newPage();
await np.goto(URL_);
await np.waitForFunction(() => state && state.photos.length);
await np.click(`.item:has(.name:text-is("pipe-clutter.jpg"))`);
await np.waitForSelector("#editor-slot svg.editor");
await sleep(900);
await shot(np, "d20-narrow-820px-full-page", { fullPage: true });
await narrow.close();

// -------------------------------------------------------------------- phone
console.log("phone");
const iphone = { ...devices["iPhone 14"], deviceScaleFactor: 2 };
const pctx = await browser.newContext({ ...iphone, colorScheme: "light" });
const ph = await pctx.newPage();
await ph.goto(`${PHONE}/phone#t=${encodeURIComponent(token)}`);
await sleep(700);
await shot(ph, "p01-phone-start", { fullPage: true });

// Without the QR link's token.
const bare = await browser.newContext({ ...iphone });
const bp = await bare.newPage();
await bp.goto(`${PHONE}/phone`);
await sleep(500);
await shot(bp, "p02-phone-no-token", { fullPage: true });
await bare.close();

// Upload from the phone: the photo is measured and the phone shows its own result.
await ph.setInputFiles("#choose", photo("pipe-clutter.jpg"));
await sleep(150);
await shot(ph, "p03-phone-uploading-or-measuring", { fullPage: true });
await ph.waitForSelector(".fix svg", { timeout: 60000 });
await sleep(900);
await shot(ph, "p04-phone-measured-with-fix-view", { fullPage: true });

// Tap the far side of the chain to choose the other object side. The photo is below the fold, so scroll to it.
await ph.evaluate(() => document.querySelector(".fix svg").scrollIntoView({ block: "center" }));
await sleep(300);
const svgBox = await (await ph.$(".fix svg")).boundingBox();
await ph.touchscreen.tap(svgBox.x + svgBox.width * 0.92, svgBox.y + svgBox.height * 0.5);
await sleep(250);
await shot(ph, "p05-phone-side-tapped", { fullPage: true });
await ph.waitForFunction(() => document.querySelector(".status").textContent.startsWith("Measured"), null, { timeout: 60000 });
await sleep(600);

// A photo with no chain.
await ph.setInputFiles("#choose", photo("kitchen-table-no-chain.jpg"));
await ph.waitForFunction(() => document.querySelector(".upload .status.bad"), null, { timeout: 60000 });
await sleep(400);
await shot(ph, "p06-phone-error-no-chain", { fullPage: true });

// The glare photo (it measures with warnings or not, depending on the detector).
await ph.setInputFiles("#choose", photo("s-curve-glare.jpg"));
await ph.waitForFunction(() => document.querySelectorAll(".upload .fix").length >= 2 ||
  document.querySelectorAll(".upload .status.bad").length >= 2, null, { timeout: 60000 });
await sleep(900);
await shot(ph, "p07-phone-several-photos", { fullPage: true });
await pctx.close();

const dphone = await browser.newContext({ ...iphone, colorScheme: "dark" });
const dph = await dphone.newPage();
await dph.goto(`${PHONE}/phone#t=${encodeURIComponent(token)}`);
await sleep(600);
await dph.setInputFiles("#choose", photo("pipe-clutter.jpg"));
await dph.waitForSelector(".fix svg", { timeout: 60000 });
await sleep(900);
await shot(dph, "p08-phone-dark-measured", { fullPage: true });
await dphone.close();

await browser.close();
fs.writeFileSync(path.join(OUT, "_taken.txt"), shots.join("\n") + "\n");
console.log(`${shots.length} screenshots in ${OUT}`);

// The bounding box of the first match for a selector (or the union of several, with union=true).
async function clipOf(pg, selector, union = false) {
  const boxes = [];
  for (const el of await pg.$$(selector)) { const b = await el.boundingBox(); if (b) boxes.push(b); if (!union) break; }
  const x = Math.min(...boxes.map((b) => b.x)), y = Math.min(...boxes.map((b) => b.y));
  const x2 = Math.max(...boxes.map((b) => b.x + b.width)), y2 = Math.max(...boxes.map((b) => b.y + b.height));
  return { x, y, width: x2 - x, height: y2 - y };
}
