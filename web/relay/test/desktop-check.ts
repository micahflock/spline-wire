// The desktop page (web/desktop/) in Chromium, end to end: a dev relay and a
// vite dev server, the page paired with a test phone (desktop-check-phone.html,
// the real backbone with no page around it) by the link the page shows, and a
// Python measurement sent through them. Checks that the curve arrives and is
// shown, that Copy points and Ctrl+C put Python's table on the clipboard, that
// every download is byte for byte Python's file, that a reload keeps the
// pairing and the list, and that Unpair makes a new code.
//
//   npm run check:desktop                      exits 1 on failure
//   SCREENSHOTS=<dir> npm run check:desktop    also saves 1280×800 screenshots, light and dark

import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { createServer as netServer } from "node:net";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { startDevRelay } from "../dev-relay.ts";

const WEB = fileURLToPath(new URL("../../", import.meta.url));
const SHOTS = process.env.SCREENSHOTS;

type Download = { suggestedFilename(): string; path(): Promise<string> };
type Locator = {
  count(): Promise<number>;
  allTextContents(): Promise<string[]>;
  getAttribute(name: string): Promise<string | null>;
  evaluateAll<T>(fn: string): Promise<T>;
  first(): Locator;
  nth(i: number): Locator;
  click(): Promise<void>;
};
type Page = {
  goto(url: string): Promise<unknown>;
  reload(): Promise<unknown>;
  url(): string;
  click(sel: string): Promise<void>;
  focus(sel: string): Promise<void>;
  textContent(sel: string): Promise<string | null>;
  innerHTML(sel: string): Promise<string>;
  getAttribute(sel: string, name: string): Promise<string | null>;
  evaluate<T>(fn: string | ((arg: never) => T), arg?: unknown): Promise<T>;
  waitForFunction(fn: string, arg?: unknown, opts?: { timeout: number }): Promise<unknown>;
  waitForEvent(event: "download"): Promise<Download>;
  locator(sel: string): Locator;
  keyboard: { press(key: string): Promise<void> };
  emulateMedia(o: { colorScheme: "light" | "dark" }): Promise<void>;
  screenshot(o: { path: string }): Promise<unknown>;
  title(): Promise<string>;
  addInitScript(script: string): Promise<void>;
  on(event: "pageerror", fn: (e: Error) => void): void;
  on(event: "console", fn: (m: { type(): string; text(): string }) => void): void;
};
type Context = { newPage(): Promise<Page> };
const { chromium } = createRequire(import.meta.url)("playwright") as {
  chromium: {
    launch(): Promise<{ newContext(o?: object): Promise<Context>; close(): Promise<void> }>;
  };
};

// web/'s own vite, with web/vite.config.ts: the page is served as the site serves it.
const vite = (await import(pathToFileURL(join(WEB, "node_modules/vite/dist/node/index.js")).href)) as {
  createServer(o: object): Promise<{ listen(): Promise<unknown>; close(): Promise<void> }>;
};

const fixture = JSON.parse(await readFile(join(WEB, "src/test-data/measurement.json"), "utf8")) as {
  document: { name: string; result: Record<string, unknown> };
  outputs: Record<string, string>;
  chain: { pitch_mm: number; half_width_mm: number; n_pins: number };
};
const ctx = { chainName: "O-ring chain", pitch_mm: fixture.chain.pitch_mm, half_width_mm: fixture.chain.half_width_mm, n_pins: fixture.chain.n_pins };
const NAMES: Record<string, string> = {
  dxf: "IMG_0001-curve.dxf", svg: "IMG_0001-curve.svg", csv: "IMG_0001-curve.csv", json: "IMG_0001.json",
  fusion_csv: "IMG_0001-fusion-cm.csv", tsv: "IMG_0001-points.tsv",
};

const steps: string[] = [];
const step = (s: string) => {
  steps.push(s);
  console.log("ok  " + s);
};
const errors: string[] = [];

async function freePort(): Promise<number> {
  const s = netServer();
  await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
  const port = (s.address() as { port: number }).port;
  await new Promise((r) => s.close(r));
  return port;
}

/** Wait until the page's expression is true. */
async function until(page: Page, expr: string, timeout = 15_000): Promise<void> {
  await page.waitForFunction(`(() => { try { return !!(${expr}); } catch { return false; } })()`, undefined, { timeout });
}
const q = (sel: string) => `document.querySelector(${JSON.stringify(sel)})`;
const textOf = (sel: string) => `(${q(sel)}?.textContent ?? "")`;

const relay = await startDevRelay({ port: 0 });
const port = await freePort();
const server = await vite.createServer({
  root: WEB,
  configFile: join(WEB, "vite.config.ts"),
  logLevel: "warn",
  server: { host: "127.0.0.1", port, strictPort: true },
});
await server.listen();
const site = `http://127.0.0.1:${port}`;
const browser = await chromium.launch();

try {
  const desktopCtx = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    permissions: ["clipboard-read", "clipboard-write"],
    acceptDownloads: true,
  });
  const desktop = await desktopCtx.newPage();
  const phone = await (await browser.newContext()).newPage();
  for (const [name, p] of [["desktop", desktop], ["phone", phone]] as const) {
    p.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
    p.on("console", (m) => {
      if (m.type() === "error") errors.push(`${name} console: ${m.text()}`);
    });
  }
  /** The clipboard once it holds `want`, or what it holds after 5 s (the copy is asynchronous). */
  const clipboard = async (want: string) => {
    let got = "";
    for (let i = 0; i < 50 && got !== want; i++) {
      got = await desktop.evaluate<string>("navigator.clipboard.readText()");
      if (got !== want) await new Promise((r) => setTimeout(r, 100));
    }
    return got;
  };

  // ---- pairing -----------------------------------------------------------
  await desktop.goto(`${site}/desktop/?relay=${encodeURIComponent(relay.url)}`);
  await until(desktop, `/#pair=[A-Za-z0-9_-]{22}$/.test(${textOf("[data-pair-link]")})`);
  await until(desktop, `${textOf("#link-text")} === "Connected · no phone open"`);
  const pairing = (await desktop.textContent("[data-pair-link]"))!;
  const pairUrl = new URL(pairing);
  if (pairUrl.origin !== site || pairUrl.pathname !== "/phone/") throw new Error(`the pairing link is not the phone page: ${pairing}`);
  if (pairUrl.searchParams.get("relay") !== relay.url) throw new Error(`the pairing link does not carry the relay override: ${pairing}`);
  if (!(await desktop.innerHTML("#main")).includes('class="qr"')) throw new Error("no QR code on a first visit");
  const footer = (await desktop.textContent("#build"))!;
  if (!/^Spline Wire \S+ · build \S+$/.test(footer)) throw new Error(`footer: ${footer}`);
  step(`first visit: QR code and link to ${pairUrl.pathname} with the relay passed on; status "connected, no phone"; footer "${footer}"`);
  if (SHOTS) {
    await desktop.screenshot({ path: join(SHOTS, "desktop-welcome-light.png") });
  }

  await phone.goto(`${site}/relay/test/desktop-check-phone.html${pairUrl.search}${pairUrl.hash}`);
  await until(desktop, `${textOf("#link-text")} === "Phone connected · Test phone"`);
  step("the test phone opened the link: the status line names it");

  // ---- a curve arrives -----------------------------------------------------
  await phone.evaluate(`sendCurve(...${JSON.stringify([fixture.document, fixture.outputs, "IMG_0001.jpg", ctx])})`);
  await until(desktop, `${textOf("#curve-title")} === "IMG_0001.jpg"`);
  await until(desktop, `${textOf('.item[aria-current="true"] .name')} === "IMG_0001.jpg"`);
  await until(phone, "delivered() === 1");
  if ((await desktop.locator(".plot svg .pt").count()) !== 13) throw new Error("the drawing does not show 13 points");
  if (!(await desktop.innerHTML("#pair-side")).includes("Pair another phone")) throw new Error("the QR code did not shrink once paired");
  step("a curve from Python's measurement arrived, was acknowledged, is selected and drawn; the QR code moved to \"Pair another phone\"");

  // ---- Copy points and Ctrl+C ----------------------------------------------
  await desktop.click('[data-act="copy"]');
  if ((await clipboard(fixture.outputs.tsv!)) !== fixture.outputs.tsv) throw new Error("Copy points did not copy Python's points_tsv");
  await desktop.evaluate("navigator.clipboard.writeText('')");
  await desktop.focus('.item[aria-current="true"]');
  await desktop.keyboard.press("Control+c");
  if ((await clipboard(fixture.outputs.tsv!)) !== fixture.outputs.tsv) throw new Error("Ctrl+C did not copy Python's points_tsv");
  step("Copy points and Ctrl+C put Python's points_tsv on the clipboard, byte for byte");

  // ---- downloads -----------------------------------------------------------
  await until(desktop, `document.querySelectorAll("#exp a[data-kind]").length === 6`);
  for (const kind of Object.keys(NAMES)) {
    const [download] = await Promise.all([desktop.waitForEvent("download"), desktop.click(`#exp a[data-kind="${kind}"]`)]);
    if (download.suggestedFilename() !== NAMES[kind]) throw new Error(`${kind} downloads as ${download.suggestedFilename()}, expected ${NAMES[kind]}`);
    const bytes = await readFile(await download.path());
    if (!bytes.equals(Buffer.from(fixture.outputs[kind]!, "utf8"))) throw new Error(`the ${kind} download differs from Python's file`);
  }
  step(`all 6 downloads have Python's names and bytes (${Object.values(NAMES).join(", ")})`);

  // ---- which curve is shown ------------------------------------------------
  await phone.evaluate(`sendCurve(...${JSON.stringify([fixture.document, fixture.outputs, "IMG_0002.jpg", ctx])})`);
  await until(desktop, `${textOf("#curve-title")} === "IMG_0002.jpg"`);
  await desktop.click('.item[data-id] >> nth=1');           // the person picks the older one
  await until(desktop, `${textOf("#curve-title")} === "IMG_0001.jpg"`);
  await phone.evaluate(`sendCurve(...${JSON.stringify([fixture.document, fixture.outputs, "IMG_0003.jpg", ctx])})`);
  await until(desktop, `${textOf(".item .name")} === "IMG_0003.jpg"`);
  await until(desktop, `${textOf(".item.flash .name")} === "IMG_0003.jpg"`, 2000);
  if ((await desktop.textContent("#curve-title")) !== "IMG_0001.jpg") throw new Error("a new curve took the place of the person's pick");
  const names = await desktop.locator(".item .name").allTextContents();
  if (names.join() !== "IMG_0003.jpg,IMG_0002.jpg,IMG_0001.jpg") throw new Error(`list order: ${names.join()}`);
  step("a new curve is shown unless the person just picked another; then it flashes at the top of the list, newest first");

  // ---- reload --------------------------------------------------------------
  await desktop.reload();
  await until(desktop, `${textOf("#link-text")} === "Phone connected · Test phone"`);
  if ((await desktop.textContent("[data-pair-link]")) !== pairing) throw new Error("the desktop made a new pairing on reload");
  if ((await desktop.locator(".item").count()) !== 3) throw new Error("the list did not survive the reload");
  step("reload: same pairing, the three curves still listed, the phone back in the room");

  await desktop.focus('.item[aria-current="true"]');
  await desktop.keyboard.press("ArrowDown");
  await until(desktop, `document.activeElement?.getAttribute("aria-current") === "true" && ${textOf("#curve-title")} === "IMG_0002.jpg"`);
  await desktop.keyboard.press("End");
  await until(desktop, `${textOf("#curve-title")} === "IMG_0001.jpg"`);
  step("arrow keys and End move through the list, focus following the curve on show");

  // ---- warnings and notes ----------------------------------------------------
  // The fixture's measurement had none: give one the warnings Python writes for a
  // photo with two pins hidden and no focal length, and a note.
  const doc = fixture.document;
  const warned = { ...doc, result: { ...doc.result, missing_pins: 2, focal_source: "estimated",
    warnings: ["2 pin(s) not detected; the curve bridges those gaps",
      "no focal length (none in the photo's EXIF, and none set); estimating it from the chain, which is unreliable for nearly straight chains"],
    notes: ["ignored 3 look-alike detection(s) not on the chain"] } };
  await phone.evaluate(`sendCurve(...${JSON.stringify([warned, fixture.outputs, "IMG_0004.jpg", ctx])})`);
  await until(desktop, `${textOf(".item .name")} === "IMG_0004.jpg"`);
  await desktop.click(".item >> nth=0");                     // the keys above were a pick: it holds
  await until(desktop, `${textOf("#curve-title")} === "IMG_0004.jpg"`);
  const warns = await desktop.locator(".note.warn b").allTextContents();
  const notes = await desktop.locator(".note.info b").allTextContents();
  if (warns.join() !== "2 pins not found,Focal length estimated" || notes.join() !== "3 stray dots ignored") {
    throw new Error(`warnings ${warns.join()} / notes ${notes.join()}`);
  }
  if ((await desktop.textContent(".item .warns")) !== "2 warnings") throw new Error("the list does not count the warnings");
  if ((await desktop.locator(".cell.warn").count()) !== 2) throw new Error("the pin count and the estimated focal length are not marked");
  step("warnings in orange (list count, notes, cells) and Python's note in grey");

  if (SHOTS) {
    await desktop.evaluate("document.activeElement.blur()");
    await new Promise((r) => setTimeout(r, 2000));      // past the new curve's flash
    await desktop.screenshot({ path: join(SHOTS, "desktop-curve-light.png") });
    await desktop.emulateMedia({ colorScheme: "dark" });
    await desktop.screenshot({ path: join(SHOTS, "desktop-curve-dark.png") });
    await desktop.emulateMedia({ colorScheme: "light" });
    step(`screenshots in ${SHOTS}`);
  }

  // ---- delete --------------------------------------------------------------
  await desktop.click('[data-act="delete"]');
  await desktop.click("#confirm-yes");
  await until(desktop, `document.querySelectorAll(".item").length === 3`);
  if ((await desktop.textContent("#curve-title")) !== "IMG_0003.jpg") throw new Error("after a delete the next curve is not shown");
  step("Delete asks first, then takes the curve off the list and shows the next");

  // ---- unpair --------------------------------------------------------------
  const qrBefore = await desktop.innerHTML("#pair-side svg.qr");
  await desktop.click("#pair-side summary");                 // "Pair another phone"
  await desktop.click('[data-act="unpair"]');
  await desktop.click("#confirm-yes");
  await until(desktop, `${textOf("[data-pair-link]")} !== ${JSON.stringify(pairing)}`);
  const qrAfter = await desktop.innerHTML("#pair-side svg.qr");
  if (qrAfter === qrBefore) throw new Error("Unpair kept the QR code");
  await until(desktop, `${textOf("#link-text")} === "Connected · no phone open"`);
  if ((await desktop.locator(".item").count()) !== 3) throw new Error("Unpair lost the curves");
  step("Unpair: a new link and QR code, the old phone cut off, the curves kept");
  if (SHOTS) await desktop.screenshot({ path: join(SHOTS, "desktop-unpaired-light.png") });

  // ---- the dull cases ----------------------------------------------------------
  // A private window where localStorage throws, and a relay that is not there.
  const bare = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
  bare.on("pageerror", (e) => errors.push(`no-storage page: ${e.message}`));
  await bare.addInitScript(
    `Object.defineProperty(window, "localStorage", { get() { throw new DOMException("denied", "SecurityError"); } });`);
  await bare.goto(`${site}/desktop/?relay=${encodeURIComponent("ws://127.0.0.1:" + (await freePort()))}`);
  await until(bare, `!${q("#storage-note")}.hidden && ${q("#main svg.qr")}`);
  await until(bare, `${textOf("#link-text")} === "Can't reach the relay" && !${q("#retry")}.hidden`);
  await until(bare, `/^Trying again in \\d+ s$/.test(${textOf("#link-wait")})`);
  step("no localStorage: the page works and says the pairing will not be remembered; no relay: it says so, counts down and offers Retry now");
  if (SHOTS) await bare.screenshot({ path: join(SHOTS, "desktop-no-storage-no-relay-light.png") });

  if (errors.length) throw new Error("page errors:\n  " + errors.join("\n  "));
  console.log(`\ndesktop check passed (${steps.length} steps)`);
} catch (err) {
  console.error("desktop check FAILED after:\n  " + steps.join("\n  ") + "\n" + String(err) + (errors.length ? "\n" + errors.join("\n") : ""));
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
  await relay.close();
}
