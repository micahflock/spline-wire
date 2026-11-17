// What the pages draw of a photo: a downscaled copy for the editor and a
// thumbnail, so a page never holds the decoded original (S1: memory; a
// 12-megapixel photo decoded is 48 MB). The core measures the original.

import type { Size } from "../editor/viewport";

/** The editor's copy: long side in px. */
export const PREVIEW_PX = 2560;
export const THUMB_PX = 128;

export interface Previews {
  /** The original's size, upright, as the core reads it. */
  pixels: Size;
  previewSize: Size;
  preview: Blob;
  /** Only when asked for. */
  thumb?: Blob;
}

/** null if this browser cannot decode the file (HEIC outside Safari): the core may still read it. */
export async function previews(file: Blob, opts: { thumb?: boolean } = {}): Promise<Previews | null> {
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file);       // upright per EXIF, as the core reads it
  } catch {
    return null;
  }
  try {
    const pixels = { width: bmp.width, height: bmp.height };
    const k = Math.min(1, PREVIEW_PX / Math.max(bmp.width, bmp.height));
    const previewSize = { width: Math.round(bmp.width * k), height: Math.round(bmp.height * k) };
    const preview = await jpeg(bmp, previewSize, 0.85);
    const out: Previews = { pixels, previewSize, preview };
    if (opts.thumb) {
      const t = THUMB_PX / Math.min(bmp.width, bmp.height);
      out.thumb = await jpeg(bmp, { width: Math.round(bmp.width * t), height: Math.round(bmp.height * t) }, 0.8);
    }
    return out;
  } catch {
    return null;
  } finally {
    bmp.close();
  }
}

function jpeg(src: CanvasImageSource, size: Size, quality: number): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = size.width;
  c.height = size.height;
  const g = c.getContext("2d")!;
  g.imageSmoothingQuality = "high";
  g.drawImage(src, 0, 0, size.width, size.height);
  return new Promise<Blob>((resolve, reject) => c.toBlob((b) => {
    c.width = c.height = 0;                      // give the canvas memory back now (iOS counts it)
    if (b) resolve(b);
    else reject(new Error("could not encode the preview"));
  }, "image/jpeg", quality));
}
