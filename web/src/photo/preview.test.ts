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
