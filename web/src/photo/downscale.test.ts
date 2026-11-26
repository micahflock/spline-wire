import { describe, expect, it } from "vitest";
import { boxDownscale } from "./downscale";

const rgb = (...px: number[][]) => new Uint8Array(px.flat());

describe("boxDownscale", () => {
  it("averages the source pixels under each output pixel", () => {
    // 4×2 RGB to 2×1: each output pixel the mean of a 2×2 block.
    const src = rgb([0, 0, 0], [40, 80, 120], [255, 255, 255], [255, 255, 255],
                    [10, 20, 30], [30, 60, 90], [0, 0, 0], [255, 255, 255]);
    const out = boxDownscale(src, 4, 2, 12, 3, 2, 1);
    expect([...out.data]).toEqual([20, 40, 60, 255, 191, 191, 191, 255]);
  });

  it("skips each row's padding, reads RGBA, and keeps the size when asked to", () => {
    // 2×2 RGBA rows padded to 12 bytes; alpha is ignored and comes out opaque.
    const src = new Uint8Array([1, 2, 3, 9, 4, 5, 6, 9, 7, 7, 7, 7, 10, 20, 30, 0, 40, 50, 60, 0, 7, 7, 7, 7]);
    expect([...boxDownscale(src, 2, 2, 12, 4, 2, 2).data]).toEqual([1, 2, 3, 255, 4, 5, 6, 255, 10, 20, 30, 255, 40, 50, 60, 255]);
  });

  it("shares the columns out when the sizes do not divide (5 to 2: 2 and 3 columns)", () => {
    const src = rgb([0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]);
    expect([...boxDownscale(src, 5, 1, 15, 3, 2, 1).data]).toEqual([5, 5, 5, 255, 30, 30, 30, 255]);
  });

  it("refuses to enlarge or to make nothing", () => {
    expect(() => boxDownscale(rgb([0, 0, 0]), 1, 1, 3, 3, 2, 1)).toThrow(RangeError);
    expect(() => boxDownscale(rgb([0, 0, 0]), 1, 1, 3, 3, 0, 1)).toThrow(RangeError);
  });
});
