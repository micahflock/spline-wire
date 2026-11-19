// The accuracy test page (spikes/accuracy/), as the owner will use it: a photo
// of the S plaque picked from the "library" is measured and scored at once,
// kept in the list, and the downloaded report is what splinewire score reads.

import { readFileSync } from "node:fs";
import { chromium, expect, test } from "@playwright/test";

test("the accuracy page scores a plaque photo and writes the report", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("spikes/accuracy/");
  await page.locator("#phone").fill("Test Phone");
  await page.locator("#target").selectOption("plaque-s");
  await page.locator("#surface").selectOption("white paper");
  await page.locator("#pitch").fill("12.98");          // the chain's: a plaque keeps its printed 13.00
  await page.locator("#pick").setInputFiles("spikes/accuracy/test-photo.jpg");
  await expect(page.locator("#now")).toContainText("PASS", { timeout: 120_000 });
  const now = (await page.locator("#now").textContent())!;
  console.log(`[accuracy] ${now}`);
  expect(now).toContain("testphone-001");
  await expect(page.locator("#tests")).toContainText("exif focal 26 mm: worst pin");
  await expect(page.locator("#tests")).toContainText("estimated focal");
  await expect(page.locator("#tests")).toContainText("best-fit focal 2");       // the render's 26 mm
  await expect(page.locator("#fit-text")).toContainText("over 1 plaque photo(s)");
  await page.getByRole("button", { name: "Use it as the override" }).click();
  const fitted = Number(await page.locator("#override").inputValue());
  expect(fitted).toBeGreaterThan(25.5);
  expect(fitted).toBeLessThan(26.5);

  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download the report (JSON)" }).click()]);
  const report = JSON.parse(readFileSync((await download.path())!, "utf-8"));
  expect(report.schema).toBe("spline-wire/test-report@1");
  const t = report.tests[0];
  expect(t).toMatchObject({ id: "testphone-001", photo: "testphone-001-plaque-s.jpg", target: "plaque-s", pins: 13, pitch_mm: 13,
    shade: "light", chain: "snap", route: "library", conditions: { surface: "white paper" } });
  expect(t.exif.focal_35mm).toBe(26);
  expect(t.results.focal.main.source).toBe("exif");
  expect(t.results.focal.main.score.max_error_mm).toBeLessThan(0.05);
  expect(t.results.focal.fitted.source).toBe("fitted");
  expect(t.focal_override_35mm).toBeNull();

  // A reload keeps the test (IndexedDB) and the settings.
  await page.reload();
  await expect(page.locator("#tests")).toContainText("testphone-001");
  await expect(page.locator("#phone")).toHaveValue("Test Phone");
  await expect(page.locator("#pitch")).toHaveValue("12.98");
});

test("the camera opens again after a photo, and the shutter waits for it", async ({ baseURL }) => {
  test.setTimeout(3 * 60_000);
  const browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
  try {
    const ctx = await browser.newContext({ baseURL, permissions: ["camera"] });
    const page = await ctx.newPage();
    await page.goto("spikes/accuracy/");
    for (const round of [1, 2]) {
      await page.getByRole("button", { name: "Open the camera" }).click();
      await expect(page.locator("#video")).toBeVisible();
      await expect(page.locator("#now")).toContainText("Frame the whole chain", { timeout: 15_000 });
      await expect(page.locator("#shoot")).toBeEnabled();
      await page.getByRole("button", { name: "Take the photo" }).click();
      // The fake camera shows no chain: the photo is taken and measured, and says so.
      await expect(page.locator("#now")).toContainText(`-00${round}:`, { timeout: 60_000 });
      await expect(page.locator("#now")).not.toContainText("measuring", { timeout: 60_000 });
      await expect(page.locator("#video")).toBeHidden();
    }
  } finally {
    await browser.close();
  }
});

