// The accuracy test page (spikes/accuracy/), as the owner will use it: a photo
// of the S plaque picked from the "library" is measured and scored at once,
// kept in the list, and the downloaded report is what splinewire score reads.

import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

test("the accuracy page scores a plaque photo and writes the report", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("spikes/accuracy/");
  await page.locator("#phone").fill("Test Phone");
  await page.locator("#target").selectOption("plaque-s");
  await page.locator("#surface").selectOption("white paper");
  await page.locator("#pick").setInputFiles("spikes/accuracy/test-photo.jpg");
  await expect(page.locator("#now")).toContainText("PASS", { timeout: 120_000 });
  const now = (await page.locator("#now").textContent())!;
  console.log(`[accuracy] ${now}`);
  expect(now).toContain("testphone-001");
  await expect(page.locator("#tests")).toContainText("exif focal 26 mm: worst pin");
  await expect(page.locator("#tests")).toContainText("estimated focal");

  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download the report (JSON)" }).click()]);
  const report = JSON.parse(readFileSync((await download.path())!, "utf-8"));
  expect(report.schema).toBe("spline-wire/test-report@1");
  const t = report.tests[0];
  expect(t).toMatchObject({ id: "testphone-001", photo: "testphone-001-plaque-s.jpg", target: "plaque-s", pins: 13,
    shade: "light", chain: "snap", route: "library", conditions: { surface: "white paper" } });
  expect(t.exif.focal_35mm).toBe(26);
  expect(t.results.focal.main.source).toBe("exif");
  expect(t.results.focal.main.score.max_error_mm).toBeLessThan(0.05);

  // A reload keeps the test (IndexedDB) and the settings.
  await page.reload();
  await expect(page.locator("#tests")).toContainText("testphone-001");
  await expect(page.locator("#phone")).toHaveValue("Test Phone");
});
