// Downscaling RGB(A) pixels by averaging the source pixels under each
// output pixel (a box filter), for the previews the engine makes of photos
// the browser cannot decode itself (HEIC outside Safari: pipeline/heif.ts).
// Pure, so it is tested in Node; the pages' own previews use a canvas.

/** RGBA pixels, row after row, no padding. */
export interface Rgba {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Shrink `src` (`channels` bytes a pixel, R G B first; rows `stride` bytes
 * apart) to `outW` × `outH`, no larger than it. Each output pixel is the mean
 * of the source pixels whose index falls in its share of the rows and columns. */
export function boxDownscale(src: Uint8Array | Uint8ClampedArray, width: number, height: number, stride: number,
  channels: number, outW: number, outH: number): Rgba {
  if (!(outW >= 1 && outH >= 1 && outW <= width && outH <= height)) {
    throw new RangeError(`cannot shrink ${width}×${height} to ${outW}×${outH}`);
  }
  const out = new Uint8ClampedArray(outW * outH * 4);
  const xs = new Int32Array(outW + 1);
  for (let i = 0; i <= outW; i++) xs[i] = Math.floor((i * width) / outW);
  const sum = new Uint32Array(outW * 3);
  for (let oy = 0; oy < outH; oy++) {
    const y0 = Math.floor((oy * height) / outH), y1 = Math.floor(((oy + 1) * height) / outH);
    sum.fill(0);
    for (let y = y0; y < y1; y++) {
      const row = y * stride;
      for (let ox = 0; ox < outW; ox++) {
        let r = 0, g = 0, b = 0;
        for (let x = xs[ox]!, end = xs[ox + 1]!; x < end; x++) {
          const j = row + x * channels;
          r += src[j]!;
          g += src[j + 1]!;
          b += src[j + 2]!;
        }
        sum[3 * ox] = sum[3 * ox]! + r;
        sum[3 * ox + 1] = sum[3 * ox + 1]! + g;
        sum[3 * ox + 2] = sum[3 * ox + 2]! + b;
      }
    }
    for (let ox = 0, o = oy * outW * 4; ox < outW; ox++, o += 4) {
      const n = (xs[ox + 1]! - xs[ox]!) * (y1 - y0);
      out[o] = Math.round(sum[3 * ox]! / n);
      out[o + 1] = Math.round(sum[3 * ox + 1]! / n);
      out[o + 2] = Math.round(sum[3 * ox + 2]! / n);
      out[o + 3] = 255;
    }
  }
  return { width: outW, height: outH, data: out };
}
