// A minimal PNG reader for the tests: 8-bit grayscale, non-interlaced, as
// cv2.imwrite writes a gray image, or 8-bit RGB, as Pillow writes a colour
// one. Not used by the pages.

import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import type { Gray } from "../cv";

export function readGrayPng(path: string | URL): Gray {
  const png = readPng(path);
  if (png.channels !== 1) throw new Error("only 8-bit gray, non-interlaced PNGs");
  return { width: png.width, height: png.height, data: png.data };
}

/** An 8-bit RGB PNG (as Pillow writes an RGB image) as RGBA bytes, as a canvas's getImageData gives them. */
export function readRgbaPng(path: string | URL): { width: number; height: number; data: Uint8Array } {
  const png = readPng(path);
  if (png.channels !== 3) throw new Error("only 8-bit RGB, non-interlaced PNGs");
  const n = png.width * png.height, data = new Uint8Array(4 * n);
  for (let i = 0; i < n; i++) {
    data[4 * i] = png.data[3 * i]!;
    data[4 * i + 1] = png.data[3 * i + 1]!;
    data[4 * i + 2] = png.data[3 * i + 2]!;
    data[4 * i + 3] = 255;
  }
  return { width: png.width, height: png.height, data };
}

/** An 8-bit gray (1 channel) or RGB (3) PNG, non-interlaced, its samples row by row. */
function readPng(path: string | URL): { width: number; height: number; channels: number; data: Uint8Array } {
  const buf = readFileSync(path as URL);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 8, width = 0, height = 0, channels = 0;
  const idat: Uint8Array[] = [];
  while (pos < buf.length) {
    const len = view.getUint32(pos);
    const type = String.fromCharCode(...buf.subarray(pos + 4, pos + 8));
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      channels = body[9] === 0 ? 1 : body[9] === 2 ? 3 : 0;
      if (body[8] !== 8 || !channels || body[12] !== 0) throw new Error("only 8-bit gray or RGB, non-interlaced PNGs");
    } else if (type === "IDAT") {
      idat.push(body);
    }
    pos += 12 + len;
  }
  let total = 0;
  for (const b of idat) total += b.length;
  const z = new Uint8Array(total);
  let o = 0;
  for (const b of idat) {
    z.set(b, o);
    o += b.length;
  }
  const raw = inflateSync(z);
  const row = width * channels, bpp = channels;
  const data = new Uint8Array(row * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (row + 1)]!;
    const src = y * (row + 1) + 1;
    for (let x = 0; x < row; x++) {
      const a = x >= bpp ? data[y * row + x - bpp]! : 0;
      const b = y > 0 ? data[(y - 1) * row + x]! : 0;
      const c = x >= bpp && y > 0 ? data[(y - 1) * row + x - bpp]! : 0;
      let v = raw[src + x]!;
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      data[y * row + x] = v & 255;
    }
  }
  return { width, height, channels, data };
}
