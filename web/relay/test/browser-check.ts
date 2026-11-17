// S4 in a real browser: two pages paired by the link the desktop shows, a
// curve passed, the pairing remembered over a reload, and delivery after the
// relay drops every socket. The phone and the desktop each get their own
// browser context, so they share no storage, as two devices would not.
//
//   npm run check:browser        (Chromium through Playwright; exits 1 on failure)

import { createRequire } from "node:module";
import { startDemo } from "../demo/serve.ts";

type Page = {
  goto(url: string): Promise<unknown>;
  reload(): Promise<unknown>;
  url(): string;
  click(sel: string): Promise<void>;
  textContent(sel: string): Promise<string | null>;
  getAttribute(sel: string, name: string): Promise<string | null>;
  waitForFunction(fn: string, arg?: unknown, opts?: { timeout: number }): Promise<unknown>;
  locator(sel: string): { count(): Promise<number>; allTextContents(): Promise<string[]> };
  on(event: "pageerror" | "console", fn: (e: { message?: string; text?(): string; type?(): string }) => void): void;
};

const { chromium } = createRequire(import.meta.url)("playwright") as {
  chromium: { launch(): Promise<{ newContext(): Promise<{ newPage(): Promise<Page> }>; close(): Promise<void> }> };
};

const demo = await startDemo();
const browser = await chromium.launch();
const steps: string[] = [];
const step = (s: string) => {
  steps.push(s);
  console.log("ok  " + s);
};

/** Wait until the element's text matches. */
async function text(page: Page, sel: string, re: RegExp, timeout = 15_000): Promise<void> {
  await page.waitForFunction(
    `(() => { const e = document.querySelector(${JSON.stringify(sel)}); return !!e && ${re}.test(e.textContent); })()`,
    undefined,
    { timeout },
  );
}

try {
  const desktop = await (await browser.newContext()).newPage();
  const phone = await (await browser.newContext()).newPage();
  for (const p of [desktop, phone]) p.on("pageerror", (e) => console.log("page error: " + e.message));

  await desktop.goto(demo.desktop);
  await text(desktop, "#pair", /#pair=[A-Za-z0-9_-]{22}$/);
  const pairing = (await desktop.getAttribute("#pair", "href"))!;
  step(`desktop shows the pairing link ${pairing.replace(/#pair=.*/, "#pair=…")}`);

  await phone.goto(pairing);
  await text(phone, "#pairing", /^Paired/);
  if (phone.url().includes("#")) throw new Error("the phone page left the secret in its address: " + phone.url());
  await text(phone, "#status", /open, 1 peer/);
  await text(desktop, "#status", /open, 1 peer/);
  step("phone opened the link, removed the secret from its address, both pages are in the room");

  await phone.click("#send");
  await text(desktop, "#curves", /demo-1\.jpg/);
  await text(phone, "#delivered", /^1$/);
  step("a curve went from the phone to the desktop and was acknowledged");

  const before = await desktop.getAttribute("#pair", "href");
  await desktop.reload();
  await text(desktop, "#status", /open, 1 peer/);
  if ((await desktop.getAttribute("#pair", "href")) !== before) throw new Error("the desktop made a new pairing on reload");
  await text(desktop, "#curves", /demo-1\.jpg/);
  step("desktop reloaded: same pairing (localStorage), the received curve still listed, back in the room");

  await phone.click("#send");
  await text(desktop, "#curves", /demo-2\.jpg/);
  step("a second curve arrived after the reload");

  await demo.stopRelay();
  await text(phone, "#status", /^(waiting|connecting)/);
  await phone.click("#send");
  await text(phone, "#outbox", /demo-3\.jpg: queued/);
  await demo.restartRelay();
  await text(desktop, "#curves", /demo-3\.jpg/, 40_000);
  await text(phone, "#delivered", /^3$/);
  const listed = await desktop.locator("#curves li").count();
  if (listed !== 3) throw new Error(`the desktop lists ${listed} curves, expected 3`);
  step("relay dropped every socket: the phone queued a curve, both reconnected, it arrived once");

  console.log(`\nbrowser check passed (${steps.length} steps)`);
} catch (err) {
  console.error("browser check FAILED after:\n  " + steps.join("\n  ") + "\n" + String(err));
  process.exitCode = 1;
} finally {
  await browser.close();
  await demo.close();
}
