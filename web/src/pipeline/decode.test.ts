import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { shapeOf, toGray } from "../overlay/blobs";
import { grayFromRgba } from "./decode";
import { readRgbaPng } from "./golden/png";
import { grayFromRgbRows } from "./heif";
import type { GrayChannel } from "./types";

const CHANNELS: GrayChannel[] = ["luma", "r", "g", "b"];
// Values from Pillow 12: Image.fromarray(rgb).convert("L"), and getchannel("R"), ("G"), ("B").
const RGB = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [12, 200, 99], [128, 128, 128], [240, 122, 35], [235, 230, 214]];
const PILLOW: Record<GrayChannel, number[]> = {
  luma: [76, 150, 29, 132, 128, 147, 230],
  r: [255, 0, 0, 12, 128, 240, 235],
  g: [0, 255, 0, 200, 128, 122, 230],
  b: [0, 0, 255, 99, 128, 35, 214],
};
const rgba = Uint8Array.from(RGB.flatMap((p) => [...p, 255]));

describe("a photo's colour to the gray detection reads (ChainSpec gray_channel), as Pillow does it", () => {
  it("turns RGB gray as Pillow's convert('L') does, and takes one channel as its getchannel does", () => {
    for (const ch of CHANNELS) expect(Array.from(grayFromRgba(rgba, RGB.length, 1, ch).data), ch).toEqual(PILLOW[ch]);
    expect(Array.from(grayFromRgba(rgba, RGB.length, 1).data)).toEqual(PILLOW.luma);        // the default
  });

  it("gives Pillow's gray of every channel of a colour photo, bit for bit (golden/channels.json)", () => {
    // photo-snap-orange.png and the SHA-256 of pipeline.gray_from_image's output: web/scripts/golden_session.py.
    const want = JSON.parse(new TextDecoder().decode(readFileSync(new URL("./golden/channels.json", import.meta.url))))["photo-snap-orange.png"];
    const photo = readRgbaPng(new URL("./golden/photo-snap-orange.png", import.meta.url));
    for (const ch of CHANNELS) {
      const gray = grayFromRgba(photo.data, photo.width, photo.height, ch);
      expect(createHash("sha256").update(gray.data).digest("hex"), ch).toBe(want[ch]);
    }
  });

  it("does the same from the HEIC decoder's RGB rows, padding and all (heif.ts)", () => {
    const width = RGB.length, stride = 3 * width + 5;
    const rows = new Uint8Array(2 * stride).fill(77);                       // two rows, the padding left as junk
    for (let y = 0; y < 2; y++) RGB.forEach((p, x) => rows.set(p, y * stride + 3 * x));
    for (const ch of CHANNELS) {
      expect(Array.from(grayFromRgbRows(rows, width, 2, stride, ch).data), ch).toEqual([...PILLOW[ch], ...PILLOW[ch]]);
    }
  });

  it("the live view's toGray: its own luma as before, and a channel exactly as the still reads it", () => {
    expect(Array.from(toGray(rgba, RGB.length, 1).data)).toEqual(RGB.map(([r, g, b]) => (r! * 77 + g! * 150 + b! * 29) >> 8));
    for (const ch of ["r", "g", "b"] as const) {
      expect(Array.from(toGray(rgba, RGB.length, 1, undefined, ch).data), ch).toEqual(PILLOW[ch]);
      const into = new Uint8Array(20).fill(9);                                // a buffer kept between frames
      expect(Array.from(toGray(rgba, RGB.length, 1, into, ch).data), ch).toEqual(PILLOW[ch]);
    }
    // The grab (page or worker) reads the channel from the chain's shape.
    const snap = { fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07 };
    expect(shapeOf({ ...snap, fiducial_shade: "dark", gray_channel: "b" })).toEqual({ inner: 7.07 / 8.79, dark: true, channel: "b" });
    expect(shapeOf({ ...snap, gray_channel: "luma" })).toEqual({ inner: 7.07 / 8.79 });
  });
});
