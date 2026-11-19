// The desktop page's ?fusion mode (src/desktop/fusion.ts), as it runs in the
// Spline Wire add-in's palette: at palette width, with adsk.fusionSendData
// stubbed as fusion/SplineWire/palette.py answers it (a Promise, appearing a
// little after the page starts, S6), paired through a dev relay with a test
// phone that is the site's own relay client running in Node. Curves that arrive
// go into the open sketch or wait; Insert puts one into the open sketch or a new
// one; marks survive a reload; nothing of the clipboard, downloads or measuring.

import { expect, test, type Page } from "@playwright/test";
import { startDevRelay, type DevRelay } from "../relay/dev-relay";
import { deriveKeys } from "../src/relay/crypto";
import { RelayLink, type LinkStatus } from "../src/relay/link";
import { newCurveId, type Curve, type Message } from "../src/relay/messages";
import { secretFromHash } from "../src/relay/pairing";

let relay: DevRelay;
test.beforeAll(async () => { relay = await startDevRelay({ port: 0 }); });
test.afterAll(async () => { await relay.close(); });
test.use({ viewport: { width: 380, height: 760 } });

function curve(photo: string): Curve {
  return {
    v: 1, id: newCurveId(), measured_at: new Date().toISOString(), photo,
    chain: { name: "Snap-fit chain", pitch_mm: 13, half_width_mm: 2.2, n_pins: 13 },
    points_mm: [[0, 0], [12.5, 2.25], [25, 8]], pins_mm: [[0, 2], [13, 4], [26, 10]],
    object_side: "outside", warnings: [], notes: [],
    diagnostics: { focal_source: "exif", focal_35mm: 24, tilt_deg: 3.5, missing_pins: 0,
      link_residual_rms_mm: 0.02, link_residual_max_mm: 0.04 },
  };
}

/** The add-in as palette.py answers. window.__fusion.sketch is the sketch being edited. */
function stubAddin() {
  type W = { __fusion: { sketch: string | null; calls: Array<{ action: string; msg: Record<string, unknown> }> };
    fusionJavaScriptHandler: { handle(a: string, d: string): string }; adsk?: unknown };
  const w = window as unknown as W;
  w.__fusion = { sketch: null, calls: [] };
  const answer = (action: string, data: string) => {
    const msg = JSON.parse(data) as Record<string, unknown>;
    w.__fusion.calls.push({ action, msg });
    if (action === "status") return { ok: true, v: 1, addin: "0.2.0", sketch: w.__fusion.sketch };
    if (action !== "insert") return { ok: false, error: "unknown" };
    const sketch = w.__fusion.sketch;
    if (msg.how === "auto" && !sketch) return { ok: true, queued: false, sketch: null };
    const into = sketch ?? `Spline Wire · ${String(msg.photo).replace(/\.[^.]*$/, "")}`;
    setTimeout(() => w.fusionJavaScriptHandler.handle("inserted",
      JSON.stringify({ v: 1, id: msg.id, ok: true, sketch: into, points: (msg.points_mm as unknown[]).length })), 30);
    return { ok: true, queued: true };
  };
  setTimeout(() => { w.adsk = { fusionSendData: (a: string, d: string) => Promise.resolve(JSON.stringify(answer(a, d))) }; }, 150);
}

const calls = (page: Page) => page.evaluate(() =>
  (window as unknown as { __fusion: { calls: Array<{ action: string; msg: Record<string, unknown> }> } }).__fusion.calls
    .filter((c) => c.action === "insert"));
const setSketch = (page: Page, name: string | null) =>
  page.evaluate((n) => { (window as unknown as { __fusion: { sketch: string | null } }).__fusion.sketch = n; }, name);

/** A phone paired by the page's link, sending curves and waiting for each ack. */
async function phoneFrom(link: string) {
  const secret = secretFromHash(new URL(link).hash)!;
  expect(secret).toBeTruthy();
  const acked = new Set<string>();
  let status: LinkStatus = { state: "off", peers: 0 };
  const phone = new RelayLink({
    relayUrl: relay.url, keys: await deriveKeys(secret), role: "phone",
    events: {
      status: (s) => { status = s; },
      message: (m: Message) => { if (m.type === "ack") m.ids.forEach((id) => acked.add(id)); },
    },
  });
  phone.start();
  await expect.poll(() => status.state === "open" && status.peers > 0, { timeout: 10_000 }).toBe(true);
  return {
    async send(c: Curve) {
      await phone.send({ type: "curve", curve: c });
      await expect.poll(() => acked.has(c.id), { timeout: 10_000 }).toBe(true);
    },
    stop: () => phone.stop(),
  };
}

test("in Fusion's palette: pairing, curves into the open sketch or waiting, Insert, marks after a reload", async ({ page }) => {
  test.setTimeout(90_000);
  await page.addInitScript(stubAddin);
  await page.goto(`./?fusion&relay=${encodeURIComponent(relay.url)}`);
  await expect(page.locator("body")).toHaveClass(/fusion/);
  await expect(page.getByRole("heading", { name: "Pair your phone" })).toBeVisible();
  await expect(page.locator("#fusion-sketch")).toHaveText("No sketch open: new curves wait here");
  await expect(page.locator("#addin-name")).toHaveText("add-in 0.2.0");
  await expect(page.locator("#pick-label")).toBeHidden();
  await expect(page.getByRole("button", { name: "Copy link" })).toHaveCount(0);
  const phone = await phoneFrom((await page.locator("[data-pair-link]").first().textContent())!);

  // No sketch open: the curve arrives and waits.
  const first = curve("IMG_0001.jpg");
  await phone.send(first);
  await expect(page.locator(`[data-id="${first.id}"] .tag-wait`)).toHaveText("waiting");
  expect((await calls(page))[0]!.msg).toEqual({ v: 1, id: first.id, photo: "IMG_0001.jpg", points_mm: first.points_mm, how: "auto" });
  await expect(page.getByRole("button", { name: "Insert into a new sketch…" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Copy points/ })).toHaveCount(0);
  await expect(page.locator("#exp")).toHaveCount(0);

  // A sketch is opened in Fusion: the page sees it within a poll, and the next curve goes straight in.
  await setSketch(page, "Sketch3");
  await expect(page.locator("#fusion-sketch")).toHaveText("Editing Sketch3", { timeout: 5000 });
  const second = curve("IMG_0002.jpg");
  await phone.send(second);
  await expect(page.locator(`[data-id="${second.id}"] .tag-in`)).toHaveText("Sketch3");
  await expect(page.locator("#toast")).toContainText("3 points into Sketch3");

  // The waiting curve, inserted by hand into the open sketch.
  await page.locator(`[data-id="${first.id}"]`).click();
  await page.getByRole("button", { name: "Insert into Sketch3" }).click();
  await expect(page.locator(`[data-id="${first.id}"] .tag-in`)).toHaveText("Sketch3");
  await expect(page.locator("#insert-hint")).toHaveText("Inserted into Sketch3");
  expect((await calls(page)).at(-1)!.msg.how).toBe("ask");

  // No sketch: Insert makes a new one (Fusion asks for the plane).
  await setSketch(page, null);
  await expect(page.getByRole("button", { name: "Insert into a new sketch…" })).toBeVisible({ timeout: 5000 });
  await page.getByRole("button", { name: "Insert into a new sketch…" }).click();
  await expect(page.locator(`[data-id="${first.id}"] .tag-in`)).toHaveText("Spline Wire · IMG_0001");

  // Turned off: arriving curves are only listed.
  await setSketch(page, "Sketch3");
  await page.getByLabel("Put new curves into the open sketch").uncheck();
  const before = (await calls(page)).length;
  const third = curve("IMG_0003.jpg");
  await phone.send(third);
  await expect(page.locator(`[data-id="${third.id}"]`)).toBeVisible();
  expect((await calls(page)).length).toBe(before);
  await expect(page.locator(`[data-id="${third.id}"] .tag-in, [data-id="${third.id}"] .tag-wait`)).toHaveCount(0);

  // A reload keeps the pairing, the curves, their marks and the setting.
  phone.stop();
  await page.reload();
  await expect(page.locator(`[data-id="${second.id}"] .tag-in`)).toHaveText("Sketch3");
  await expect(page.locator(`[data-id="${first.id}"] .tag-in`)).toHaveText("Spline Wire · IMG_0001");
  await expect(page.getByLabel("Put new curves into the open sketch")).not.toBeChecked();
  await expect(page.getByRole("heading", { name: "Pair your phone" })).toHaveCount(0);
});

test("opened outside Fusion, the page says where it belongs", async ({ page }) => {
  test.setTimeout(30_000);
  await page.goto(`./?fusion&relay=${encodeURIComponent(relay.url)}`);
  await expect(page.locator("#fusion-note")).toContainText("Open it there: Add from phone", { timeout: 10_000 });
  await expect(page.locator("#fusion-bar")).toBeHidden();
});

test("the desktop page's own pairing is not the palette's", async ({ page }) => {
  await page.addInitScript(stubAddin);
  await page.goto(`./?fusion&relay=${encodeURIComponent(relay.url)}`);
  const inFusion = await page.locator("[data-pair-link]").first().textContent();
  await page.goto(`./?relay=${encodeURIComponent(relay.url)}`);
  const onDesktop = await page.locator("[data-pair-link]").first().textContent();
  expect(secretFromHash(new URL(onDesktop!).hash)).not.toBe(secretFromHash(new URL(inFusion!).hash));
});
