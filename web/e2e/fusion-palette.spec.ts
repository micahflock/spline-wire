// The Fusion palette spike page (spikes/fusion-palette/), outside Fusion: in a
// plain browser it checks the crypto, a dev relay and localStorage and says it
// is not in a palette; with adsk.fusionSendData stubbed as Fusion's Qt browser
// has it (a Promise, appearing a little after the page starts) it does the round
// trips, the insert and its push back, and saving the summary.

import { expect, test, type Page } from "@playwright/test";
import { startDevRelay, type DevRelay } from "../relay/dev-relay";

let relay: DevRelay;
test.beforeAll(async () => { relay = await startDevRelay({ port: 0 }); });
test.afterAll(async () => { await relay.close(); });

const page_ = () => `spikes/fusion-palette/?relay=${encodeURIComponent(relay.url)}`;
const summary = (page: Page) => page.locator("#summary");

/** Fusion's side as the spike add-in answers, in the page: `promise` makes
 * fusionSendData return a Promise (Qt) rather than the text (the old CEF). */
function stubFusion(opts: { promise: boolean; refuseInsert?: boolean }) {
  const w = window as unknown as Record<string, unknown> & { fusionJavaScriptHandler: { handle(a: string, d: string): string } };
  const calls: Array<{ action: string; data: string }> = [];
  w.__calls = calls;
  const answer = (action: string, data: string): string => {
    calls.push({ action, data });
    const msg = JSON.parse(data) as { nonce?: string };
    if (action === "spike-echo") {
      return JSON.stringify({ ok: true, nonce: msg.nonce, fusion: "2.0.21000", os: "win32", started: "2026-11-19T10:00:00",
        sketch: "Sketch1", palettes_made: 1, switches: 0, palette_lost: 0 });
    }
    if (action === "spike-insert") {
      if (opts.refuseInsert) return JSON.stringify({ ok: false, error: "The points must be 2 to 2000 [x, y] pairs" });
      setTimeout(() => w.fusionJavaScriptHandler.handle("spike-inserted",
        JSON.stringify({ ok: true, sketch: "Sketch1", points: 3 })), 30);
      return JSON.stringify({ ok: true });
    }
    if (action === "spike-report") return JSON.stringify({ ok: true, path: "C:\\Users\\me\\spline-wire-palette-spike.txt" });
    return JSON.stringify({ ok: false, error: `Unknown request '${action}'` });
  };
  setTimeout(() => {
    w.adsk = { fusionSendData: (a: string, d: string) => (opts.promise ? Promise.resolve(answer(a, d)) : answer(a, d)) };
  }, 150);
}

test("outside Fusion: crypto, the relay and localStorage, and no adsk", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto(page_());
  const s = summary(page);
  await expect(s).toHaveValue(/page script: ran/);
  await expect(s).toHaveValue(/HKDF: ok/);
  await expect(s).toHaveValue(/AES-GCM: ok \(sealed, opened, a tampered frame refused\)/);
  await expect(s).toHaveValue(/relay: ok: ws:\/\/127\.0\.0\.1:\d+, connected in \d+ ms, ping\/pong \d+ ms, peers 2, a sealed frame crossed and opened/, { timeout: 20_000 });
  await expect(s).toHaveValue(/chromium: \d+/);
  await expect(s).toHaveValue(/secure context: yes/);                // 127.0.0.1 counts as secure
  await expect(s).toHaveValue(/localStorage: empty: first load here/);
  await expect(s).toHaveValue(/adsk: not seen in 10 s/, { timeout: 20_000 });
  await expect(s).toHaveValue(/fusionSendData: no adsk.fusionSendData: not in a Fusion palette/);
  await expect(page.locator("#insert")).toBeDisabled();
  console.log(`[fusion-palette] ${(await s.inputValue()).replace(/\n/g, " | ")}`);

  await page.reload();
  await expect(s).toHaveValue(/localStorage: kept from \S+ \(load 2\)/);
});

test("in a stubbed palette: round trip, insert and push back, undo answer, save", async ({ page, context }) => {
  test.setTimeout(60_000);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.addInitScript(stubFusion, { promise: true });
  await page.goto(page_());
  const s = summary(page);
  await expect(s).toHaveValue(/adsk: appeared \d+ ms after navigation/);
  await expect(s).toHaveValue(/fusionSendData: ok: round trip \d+ ms, it returns a Promise/);
  await expect(s).toHaveValue(/Fusion: 2\.0\.21000 on win32; sketch being edited: Sketch1/);
  await expect(s).toHaveValue(/palette: made 1 time\(s\); workspace switches 0/);
  await expect(s).toHaveValue(/localStorage across restarts: not tested yet/);

  await page.getByRole("button", { name: "Insert a test spline" }).click();
  await expect(s).toHaveValue(/insert: 3 points and a spline into Sketch1, as a command; pushed back after \d+ ms/);
  const calls = await page.evaluate(() => (window as unknown as { __calls: Array<{ action: string; data: string }> }).__calls);
  const insert = calls.find((c) => c.action === "spike-insert")!;
  expect(JSON.parse(insert.data)).toEqual({ v: 1, points_mm: [[0, 0], [20, 10], [40, 0]] });
  expect(calls.every((c) => JSON.parse(c.data).v === 1)).toBe(true);

  await page.getByLabel("What one undo did").selectOption("removed the spline");
  await expect(s).toHaveValue(/one Ctrl\+Z: removed the spline/);
  await page.getByRole("button", { name: "Copy the summary" }).click();
  await expect(page.locator("#copied")).toHaveText("copied (clipboard API)");
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("one Ctrl+Z: removed the spline");

  await page.getByRole("button", { name: /Save it to a file/ }).click();
  await expect(page.locator("#copied")).toContainText("saved to C:\\Users\\me\\spline-wire-palette-spike.txt");
  const report = (await page.evaluate(() => (window as unknown as { __calls: Array<{ action: string; data: string }> }).__calls))
    .find((c) => c.action === "spike-report")!;
  expect(JSON.parse(report.data).text).toContain("fusionSendData: ok");
});

test("in a palette whose fusionSendData returns text, and an add-in that refuses", async ({ page }) => {
  test.setTimeout(60_000);
  await page.addInitScript(stubFusion, { promise: false, refuseInsert: true });
  await page.goto(page_());
  const s = summary(page);
  await expect(s).toHaveValue(/fusionSendData: ok: round trip \d+ ms, it returns string/);
  await page.getByRole("button", { name: "Insert a test spline" }).click();
  await expect(s).toHaveValue(/insert: FAILED: the add-in refused: The points must be 2 to 2000/);
});

test("if the module never runs, the watchdog says so and the summary still copies", async ({ page, context }) => {
  test.setTimeout(60_000);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.route(/\/assets\/fusion-[^/]*\.js$/, (route) => route.abort());     // as an old Chromium failing to parse it
  await page.goto(page_());
  const s = summary(page);
  await expect(s).toHaveValue(/page script: did NOT run/, { timeout: 15_000 });
  await expect(s).toHaveValue(/chromium: \d+/);
  await page.getByRole("button", { name: "Copy the summary" }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("page script: did NOT run");
});
