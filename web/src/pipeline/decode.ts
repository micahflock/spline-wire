// A photo's bytes to the gray image the pipeline measures, in a page or a
// worker, as splinewire.pipeline.load_photo gets it from Pillow: upright by
// its EXIF orientation, then gray by Pillow's integer formula
// L = (19595 R + 38470 G + 7471 B + 0x8000) >> 16 (checked equal to
// Image.convert("L")), or one channel alone (ChainSpec gray_channel, as
// pipeline.gray_from_image). The browser's own JPEG decoder does the decoding, with
// no colour management: Pillow ignores a photo's ICC profile, so a Display P3
// phone photo must not be converted to sRGB here either. e2e/decode.spec.ts
// measures what the decoder's differences from Pillow's do to the pins.

import type { Gray } from "./cv";
import type { GrayChannel } from "./types";

/** Where a channel lies in an RGB or RGBA pixel. */
export const CHANNEL_OFFSET = { r: 0, g: 1, b: 2 } as const;

/** Gray from RGBA bytes, as Pillow converts RGB to L; or, with `channel` r, g
 * or b (ChainSpec gray_channel), that channel alone, as Pillow's getchannel. */
export function grayFromRgba(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number,
  channel: GrayChannel = "luma"): Gray {
  const data = new Uint8Array(width * height);
  if (channel === "luma") {
    for (let i = 0, j = 0; i < data.length; i++, j += 4) {
      data[i] = (rgba[j]! * 19595 + rgba[j + 1]! * 38470 + rgba[j + 2]! * 7471 + 0x8000) >>> 16;
    }
  } else {
    for (let i = 0, j = CHANNEL_OFFSET[channel]; i < data.length; i++, j += 4) data[i] = rgba[j]!;
  }
  return { width, height, data };
}

/** Decode a photo (JPEG, PNG, WebP; HEIC where the browser can, Safari) to
 * gray (its `channel`), upright. Throws if the browser cannot decode it. */
export async function decodeToGray(blob: Blob, channel: GrayChannel = "luma"): Promise<Gray> {
  const bitmap = await createImageBitmap(blob, {
    imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none",
  });
  try {
    const { width, height } = bitmap;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    if (!ctx) throw new Error("no 2D canvas to read the photo's pixels");
    ctx.drawImage(bitmap, 0, 0);
    return grayFromRgba(ctx.getImageData(0, 0, width, height).data, width, height, channel);
  } finally {
    bitmap.close();
  }
}
