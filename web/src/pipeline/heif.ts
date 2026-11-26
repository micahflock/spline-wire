// HEIC/HEIF photos (the iPhone's own format) to gray, for the browsers that
// cannot decode them (all but Safari). libheif in WebAssembly (the npm
// libheif-js build, libheif 1.23), as splinewire.pipeline.load_photo gets
// them through pillow-heif (libheif 1.23.4): the primary image, decoded to
// RGB with the container's own rotation, mirroring and crop applied
// (irot/imir/clap), then gray by Pillow's formula. pillow-heif sets the
// EXIF orientation to 1 after applying those, so the EXIF orientation of a
// HEIF photo turns nothing here either.
//
// Not bit for bit: the wasm build's HEVC decoder is libde265 1.0.15 and
// pillow-heif's 1.1.3. On a 12 MP photo at quality 90, 15 of 12 million gray
// values differ, by 1, and the pins by 0.00002 mm; on heif.test.ts's coarse
// quality-50 photo, pins 0.010 px and 0.0015 mm. (Pyodide's pillow-heif,
// which decoded HEIC in the browser before, has libheif 1.12.) Decoding a
// 12 MP photo takes about 1.3 s in Node.
//
// The decoder (1.4 MB of wasm) is loaded only for a HEIC photo, and a new
// instance is made for each photo, so the memory a decode needs is given
// back with it; compiling the module again is cheap next to the decode.

import type { Gray } from "./cv";
import { PipelineError } from "./errors";
import { boxDownscale, type Rgba } from "../photo/downscale";

// The parts of libheif-js used here (its own types leave out the JS wrapper).
interface HeifChannel { id: number; width: number; height: number; stride: number; data: Uint8Array }
interface HeifImage {
  handle: unknown;
  is_primary(): boolean;
  free(): void;
}
interface Libheif {
  HeifDecoder: new () => { decoder: number; decode(bytes: Uint8Array): HeifImage[] };
  heif_js_decode_image2(handle: unknown, colorspace: number, chroma: number): Promise<{ code?: number; message?: string; image: number; channels: HeifChannel[] }>;
  heif_image_release(image: number): void;
  heif_context_free(context: number): void;
  heif_colorspace: { heif_colorspace_RGB: number };
  heif_chroma: { heif_chroma_interleaved_RGB: number };
  heif_channel: { heif_channel_interleaved: number };
}
/** libheif-js's module factory. Given the wasm's bytes it is ready on return
 * (compiled synchronously, which a worker allows); onRuntimeInitialized says so either way. */
export type LibheifFactory = (options: { wasmBinary: ArrayBuffer | Uint8Array; onRuntimeInitialized?: () => void }) => Libheif;

/** Gray from interleaved RGB rows (with padding), as Pillow converts RGB to L. */
export function grayFromRgbRows(rgb: Uint8Array, width: number, height: number, stride: number): Gray {
  const data = new Uint8Array(width * height);
  for (let y = 0, i = 0; y < height; y++) {
    for (let j = y * stride, end = j + 3 * width; j < end; j += 3, i++) {
      data[i] = (rgb[j]! * 19595 + rgb[j + 1]! * 38470 + rgb[j + 2]! * 7471 + 0x8000) >>> 16;
    }
  }
  return { width, height, data };
}

/** Decode a HEIF photo's primary image to gray, upright. `name` is for messages.
 * With `previewSize` (from photo/preview.ts's previewSizeOf, given the upright
 * size), also the colour copy the pages draw, which their browser cannot make. */
export async function decodeHeif(bytes: Uint8Array, name: string, factory: LibheifFactory, wasm: ArrayBuffer | Uint8Array,
  previewSize?: (pixels: { width: number; height: number }) => { width: number; height: number }): Promise<{ gray: Gray; preview: Rgba | null }> {
  let ready!: () => void;
  const initialized = new Promise<void>((resolve) => (ready = resolve));
  const lib = factory({ wasmBinary: wasm, onRuntimeInitialized: () => ready() });
  await initialized;
  const decoder = new lib.HeifDecoder();
  const images = decoder.decode(bytes);
  try {
    const image = images.find((im) => im.is_primary()) ?? images[0];
    if (!image) throw new PipelineError(`could not read the photo ${name}: no image in the HEIF file`);
    const out = await lib.heif_js_decode_image2(image.handle, lib.heif_colorspace.heif_colorspace_RGB, lib.heif_chroma.heif_chroma_interleaved_RGB);
    if (!out || out.code) throw new PipelineError(`could not read the photo ${name}: ${out?.message || "the HEIF image could not be decoded"}`);
    try {
      const ch = out.channels.find((c) => c.id === lib.heif_channel.heif_channel_interleaved);
      if (!ch) throw new PipelineError(`could not read the photo ${name}: the HEIF image has no RGB pixels`);
      const gray = grayFromRgbRows(ch.data, ch.width, ch.height, ch.stride);
      let preview: Rgba | null = null;
      if (previewSize) {
        const size = previewSize({ width: ch.width, height: ch.height });
        preview = boxDownscale(ch.data, ch.width, ch.height, ch.stride, 3, size.width, size.height);
      }
      return { gray, preview };
    } finally {
      lib.heif_image_release(out.image);
    }
  } finally {
    for (const im of images) im.free();
    if (decoder.decoder) lib.heif_context_free(decoder.decoder);
  }
}
