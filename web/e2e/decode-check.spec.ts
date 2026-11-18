// The decoding check page (spikes/decode-check/), as the owner will open it on
// a phone: in Chromium the test photo decodes to Pillow's pixels exactly and
// measures Python's pins, and a photo picked from the "library" is measured.

import { expect, test } from "@playwright/test";

test("the decoding check page finds Pillow's pixels and Python's pins, and measures a picked photo", async ({ page }) => {
  test.setTimeout(3 * 60_000);
  await page.goto("spikes/decode-check/");
  await page.getByRole("button", { name: "Run the check" }).click();
  const out = page.locator("#test-out");
  await expect(out).toContainText("pins against Python's", { timeout: 120_000 });
  const text = await out.textContent();
  console.log(`[decode-check] ${text!.trim().replace(/\n/g, " | ")}`);
  expect(text).toContain("upright, as Pillow");
  expect(text).toContain("gray pixels differing from Pillow's: 0 of");
  expect(text).toContain("worst 0.0000 px");

  await page.locator("#pick").setInputFiles("spikes/decode-check/photo.jpg");
  await expect(page.locator("#own-out")).toContainText("measured 13 pins", { timeout: 120_000 });
  console.log(`[decode-check] own: ${(await page.locator("#own-out").textContent())!.trim().replace(/\n/g, " | ")}`);
});
