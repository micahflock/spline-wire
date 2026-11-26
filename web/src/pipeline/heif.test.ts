// The HEIF decoder (heif.ts) against pillow-heif: golden/photo-snap.heic, an
// iPhone-style HEIC (pixels stored a quarter turn round, an irot box turning
// them back, EXIF orientation 6 that must not turn them again), with Python's
// decode, photo_info and measurement from scripts/golden_heif.py.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { beforeAll, describe, expect, it } from "vitest";
import createLibheif from "libheif-js/libheif-wasm/libheif.js";
import type { MeasureOptions, MeasurementDocument, PhotoInfo } from "../core/types";
import type { Gray } from "./cv";
import { readGrayPng } from "./golden/png";
import { decodeHeif, grayFromRgbRows, type LibheifFactory } from "./heif";
import type { Rgba } from "../photo/downscale";
import { greyPixels, previewSizeOf } from "../photo/preview";
import { isHeif, photoInfo } from "./photo-info";
import { PhotoSession } from "./session";

const golden = JSON.parse(gunzipSync(readFileSync(new URL("./golden/heif.json.gz", import.meta.url))).toString("utf-8")) as {
  info: PhotoInfo; options: MeasureOptions; document: MeasurementDocument;
};
const heic = readFileSync(new URL("./golden/photo-snap.heic", import.meta.url));
const wasm = readFileSync(new URL("../../node_modules/libheif-js/libheif-wasm/libheif.wasm", import.meta.url));
const factory = createLibheif as unknown as LibheifFactory;

describe("HEIC decoded with libheif, as pillow-heif decodes it", () => {
  let gray: Gray;
  let preview: Rgba | null;
  beforeAll(async () => {
    ({ gray, preview } = await decodeHeif(heic, "photo-snap.heic", factory, wasm, previewSizeOf));
  }, 60000);

  it("is a HEIF file, turned upright by its container and not again by its EXIF", () => {
    expect(isHeif(heic)).toBe(true);
    expect([gray.width, gray.height]).toEqual([1200, 900]);
  });

  // The wasm build's HEVC decoder is libde265 1.0.15, pillow-heif's 1.1.3, and they
  // differ on this deliberately coarse photo (quality 50): worst 7 levels, 5% of
  // the pixels, mean 0.054; pins 0.010 px and 0.0015 mm apart. On a 12 MP photo at
  // quality 90, 15 values of 12 million differed, by 1, and the pins by 0.00002 mm.
  it("has Python's pixels, to within a few gray levels", () => {
    const want = readGrayPng(new URL("./golden/photo-snap-heic.png", import.meta.url));
    expect([want.width, want.height]).toEqual([gray.width, gray.height]);
    let worst = 0, differ = 0, sum = 0;
    for (let i = 0; i < want.data.length; i++) {
      const d = Math.abs(want.data[i]! - gray.data[i]!);
      if (d) differ++;
      sum += d;
      worst = Math.max(worst, d);
    }
    console.log(`[heif] gray against pillow-heif: worst ${worst}, ${differ} of ${want.data.length} differ, mean ${(sum / want.data.length).toFixed(4)}`);
    expect(worst).toBeLessThanOrEqual(10);
    expect(sum / want.data.length).toBeLessThan(0.1);
  });

  it("reports what photo_info reports, and the EXIF focal length", () => {
    const { info, focalPx } = photoInfo(heic, gray);
    expect(info).toEqual(golden.info);
    expect(focalPx).not.toBeNull();
  });

  it("measures as Python measures it", () => {
    const { info, focalPx } = photoInfo(heic, gray);
    const doc = new PhotoSession("photo-snap.heic", gray, focalPx, info).measure(golden.options);
    const want = golden.document;
    expect(doc.ok).toBe(true);
    expect(doc.result!.pins).toBe(want.result!.pins);
    expect(doc.result!.focal_source).toBe("exif");
    expect(doc.result!.focal_35mm).toBe(want.result!.focal_35mm);
    const px = doc.view!.pins, wpx = want.view!.pins;
    console.log(`[heif] pins against Python: ${Math.max(...wpx.map((p, k) => Math.hypot(px[k]!.x - p.x, px[k]!.y - p.y))).toFixed(3)} px, `
      + `${Math.max(...doc.curve!.pins.map((p, k) => Math.hypot(p[0] - want.curve!.pins[k]![0], p[1] - want.curve!.pins[k]![1]))).toFixed(4)} mm`);
    for (const [k, p] of wpx.entries()) expect(Math.hypot(px[k]!.x - p.x, px[k]!.y - p.y)).toBeLessThanOrEqual(0.02);
    const mm = doc.curve!.pins, wmm = want.curve!.pins;
    for (const [k, p] of wmm.entries()) expect(Math.hypot(mm[k]![0] - p[0], mm[k]![1] - p[1])).toBeLessThanOrEqual(0.003);
  });
});

describe("the pages' copy of a HEIC photo, which their browser cannot decode", () => {
  it("is upright, the size the pages' own preview would be, and the pixels measured", async () => {
    // 1200×900 is under the preview's 2560 px: the copy is the decoded photo itself.
    const { gray, preview } = await decodeHeif(heic, "photo-snap.heic", factory, wasm, previewSizeOf);
    expect(preview).not.toBeNull();
    expect([preview!.width, preview!.height]).toEqual([1200, 900]);
    const grey = new Uint8ClampedArray(preview!.data);
    greyPixels(grey);
    let differ = 0;
    for (let i = 0; i < gray.data.length; i++) if (grey[4 * i] !== gray.data[i]) differ++;
    expect(differ).toBe(0);                          // the same pixels, the same way up
  }, 60000);

  it("is made only when asked for, and shrunk when the photo is larger", async () => {
    expect((await decodeHeif(heic, "photo-snap.heic", factory, wasm)).preview).toBeNull();
    const { gray, preview } = await decodeHeif(heic, "photo-snap.heic", factory, wasm, () => ({ width: 300, height: 225 }));
    expect([preview!.width, preview!.height]).toEqual([300, 225]);
    // Each pixel the mean of a 4×4 block: its grey within a level or two of the block's mean grey.
    const grey = new Uint8ClampedArray(preview!.data);
    greyPixels(grey);
    let worst = 0;
    for (let y = 0; y < 225; y++) for (let x = 0; x < 300; x++) {
      let s = 0;
      for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++) s += gray.data[(4 * y + dy) * gray.width + 4 * x + dx]!;
      worst = Math.max(worst, Math.abs(grey[4 * (y * 300 + x)]! - s / 16));
    }
    expect(worst).toBeLessThanOrEqual(2);
  }, 60000);
});

describe("HEIF decoding failures are for the person", () => {
  it("a HEIC file cut short", async () => {
    await expect(decodeHeif(heic.subarray(0, 2000), "cut.heic", factory, wasm)).rejects.toThrow(/could not read the photo cut\.heic/);
  }, 60000);
});

describe("grayFromRgbRows", () => {
  it("skips each row's padding and converts as Pillow does", () => {
    // 2x2, rows padded to 8 bytes.
    const rgb = new Uint8Array([255, 0, 0, 0, 255, 0, 9, 9, 0, 0, 255, 10, 20, 30, 9, 9]);
    expect([...grayFromRgbRows(rgb, 2, 2, 8).data]).toEqual([76, 150, 29, 18]);
  });
});
