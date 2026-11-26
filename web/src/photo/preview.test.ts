import { describe, expect, it } from "vitest";
import { greyPixels } from "./preview";

describe("greyPixels", () => {
  it("turns each pixel to the luma detection reads (Pillow's L), keeping alpha", () => {
    const px = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 200, 100, 50, 255]);
    greyPixels(px);
    // Pillow's Image.convert("L") of these four pixels gives 76, 150, 29 and 124.
    expect([...px]).toEqual([76, 76, 76, 255, 150, 150, 150, 128, 29, 29, 29, 0, 124, 124, 124, 255]);
  });
});

describe("the previews' sizes, shared by the pages and the engine", () => {
  it("fits the long side in PREVIEW_PX, never enlarging; the thumbnail's short side is THUMB_PX", async () => {
    const { previewSizeOf, thumbSizeOf, PREVIEW_PX, THUMB_PX } = await import("./preview");
    expect(previewSizeOf({ width: 4032, height: 3024 })).toEqual({ width: PREVIEW_PX, height: 1920 });
    expect(previewSizeOf({ width: 3024, height: 4032 })).toEqual({ width: 1920, height: PREVIEW_PX });
    expect(previewSizeOf({ width: 1200, height: 900 })).toEqual({ width: 1200, height: 900 });
    expect(thumbSizeOf({ width: 4032, height: 3024 })).toEqual({ width: 171, height: THUMB_PX });
  });
});
