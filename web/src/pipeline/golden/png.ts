// A minimal PNG reader for the tests: 8-bit grayscale, non-interlaced, as
// cv2.imwrite writes a gray image. Not used by the pages.

import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import type { Gray } from "../cv";

export function readGrayPng(path: string | URL): Gray {
  const buf = readFileSync(path as URL);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 8, width = 0, height = 0;
  const idat: Uint8Array[] = [];
  while (pos < buf.length) {
    const len = view.getUint32(pos);
    const type = String.fromCharCode(...buf.subarray(pos + 4, pos + 8));
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      if (body[8] !== 8 || body[9] !== 0 || body[12] !== 0) throw new Error("only 8-bit gray, non-interlaced PNGs");
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
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (width + 1)]!;
    const src = y * (width + 1) + 1;
    for (let x = 0; x < width; x++) {
      const a = x > 0 ? data[y * width + x - 1]! : 0;
      const b = y > 0 ? data[(y - 1) * width + x]! : 0;
      const c = x > 0 && y > 0 ? data[(y - 1) * width + x - 1]! : 0;
      let v = raw[src + x]!;
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      data[y * width + x] = v & 255;
    }
  }
  return { width, height, data };
}
