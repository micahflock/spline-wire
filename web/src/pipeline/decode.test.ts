import { expect, it } from "vitest";
import { grayFromRgba } from "./decode";

it("turns RGB gray as Pillow's convert('L') does", () => {
  // Values from Pillow 12: Image.fromarray(rgb).convert("L").
  const rgba = Uint8Array.from([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 12, 200, 99, 255, 128, 128, 128, 255]);
  expect(Array.from(grayFromRgba(rgba, 5, 1).data)).toEqual([76, 150, 29, 132, 128]);
});
