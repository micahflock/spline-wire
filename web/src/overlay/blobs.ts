// The overlay's frame detector: light dots on dark links, found fast.
//
// This is not the measuring detector (splinewire/detect.py), which fits
// ellipses to sub-pixel edges at full resolution and takes about a second. It
// follows the same idea at one coarse scale, cheaply enough for every video
// frame, and only has to be good enough to coach the shot:
//
//   1. Work on a small gray frame (toWorkFrame: long side WORK_LONG_SIDE).
//   2. Threshold against the local mean, as detect._masks does (a 25 px
//      window, the pixel at least 4 gray levels above it).
//   3. Connected components; keep those of a plausible size and shape
//      (detect._dot_candidates' limits). A dot wider than the window comes out
//      hollow, so only the outline's bounding box is used, never its area.
//   4. The test that makes a blob a dot (detect._refine_dot): along rays from
//      the centre, the margin just outside is darker than the inside by a
//      steady fraction. That is the dark link around the dot; chips, glints
//      and paper do not have it.
//   5. Centre: the brightness-weighted centroid. Sharpness: how wide the edge is.
//
// Look-alikes that pass are left to the ordering (splinewire/live.py), which
// knows the pitch.

import type { Blob, GrayFrame } from "./types";

export const WORK_LONG_SIDE = 640;

export const BLOBS = {
  /** Local threshold window (px, odd) and offset, as in detect.py at its coarsest level. */
  block: 25,
  offset: 4,
  /** Bounding box limits: the smaller side at least this, the larger at most
   * a quarter of the frame and at most 4× the smaller. */
  minSide: 4,
  maxAspect: 4,
  /** Rays per blob, and where along a ray (in radii) the inside and the margin are read. */
  rays: 16,
  inside: [0.1, 0.7] as const,
  margin: [1.15, 1.45] as const,
  /** Median inside-minus-margin contrast, gray levels; the share of rays that must show any. */
  minContrast: 15,
  minRayContrast: 6,
  minRaysWithContrast: 0.8,
  /** The margin is at most this fraction of the inside, on the median ray. */
  maxMarginRatio: 0.92,
  /** Stop after this many candidates: a busy frame is not worth more time. */
  maxCandidates: 400,
};

/** RGBA pixels (ImageData.data) to gray, by the usual luma weights. */
export function toGray(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): GrayFrame {
  const data = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i++, j += 4) {
    data[i] = (rgba[j]! * 77 + rgba[j + 1]! * 150 + rgba[j + 2]! * 29) >> 8;
  }
  return { width, height, data };
}

/** The size to draw a frame at for detectBlobs, and the factor back to frame pixels. */
export function workSize(width: number, height: number): { width: number; height: number; scale: number } {
  const scale = Math.max(1, Math.max(width, height) / WORK_LONG_SIDE);
  return { width: Math.round(width / scale), height: Math.round(height / scale), scale };
}

/** Dots in a gray work frame. `scale` multiplies positions and sizes back
 * into the pixels of the frame the work frame was drawn from. */
export function detectBlobs(frame: GrayFrame, scale = 1): Blob[] {
  const { width: w, height: h, data } = frame;
  if (w < BLOBS.block || h < BLOBS.block) return [];
  const mask = localThreshold(frame);
  const out: Blob[] = [];
  const stack = new Int32Array(w * h);
  const maxSide = Math.max(w, h) / 4;
  let candidates = 0;

  for (let start = 0; start < mask.length && candidates < BLOBS.maxCandidates; start++) {
    if (mask[start] !== 1) continue;
    // Flood fill (8-connected), marking visited pixels 2.
    let top = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
    stack[top++] = start;
    mask[start] = 2;
    while (top > 0) {
      const p = stack[--top]!;
      const px = p % w, py = (p - px) / w;
      if (px < x0) x0 = px;
      if (px > x1) x1 = px;
      if (py < y0) y0 = py;
      if (py > y1) y1 = py;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if (nx < 0 || nx >= w) continue;
          const q = ny * w + nx;
          if (mask[q] === 1) {
            mask[q] = 2;
            stack[top++] = q;
          }
        }
      }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    const lo = Math.min(bw, bh), hi = Math.max(bw, bh);
    if (lo < BLOBS.minSide || hi > maxSide || hi > BLOBS.maxAspect * lo) continue;
    candidates++;
    const blob = dotAt(frame, (x0 + x1) / 2, (y0 + y1) / 2, bw / 2, bh / 2);
    if (blob) {
      out.push({ x: (blob.x + 0.5) * scale - 0.5, y: (blob.y + 0.5) * scale - 0.5, d: blob.d * scale,
        contrast: blob.contrast, sharpness: blob.sharpness });
    }
  }
  return out;
}

/** 1 where a pixel is above its neighbourhood's mean by BLOBS.offset, else 0. */
export function localThreshold(frame: GrayFrame): Uint8Array {
  const { width: w, height: h, data } = frame;
  // Summed-area table with a zero row and column in front.
  const sat = new Uint32Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += data[y * w + x]!;
      sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1]! + row;
    }
  }
  const r = (BLOBS.block - 1) / 2;
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const ya = Math.max(0, y - r), yb = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const xa = Math.max(0, x - r), xb = Math.min(w, x + r + 1);
      const sum = sat[yb * (w + 1) + xb]! - sat[ya * (w + 1) + xb]! - sat[yb * (w + 1) + xa]! + sat[ya * (w + 1) + xa]!;
      // pixel > mean + offset, without dividing
      if ((data[y * w + x]! - BLOBS.offset) * (xb - xa) * (yb - ya) > sum) mask[y * w + x] = 1;
    }
  }
  return mask;
}

/** Is the bright thing in this box a dot on a link? If so, where exactly and how sharp.
 * (cx, cy) is the box's centre and (rx, ry) its half-sizes, in work pixels. */
export function dotAt(frame: GrayFrame, cx: number, cy: number, rx: number, ry: number): Omit<Blob, never> | null {
  const n = BLOBS.rays;
  const contrasts: number[] = [], ratios: number[] = [], widths: number[] = [];
  let insideSum = 0, outsideSum = 0;
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n;
    const ex = rx * Math.cos(a), ey = ry * Math.sin(a);      // the box's ellipse, axis-aligned
    const len = Math.hypot(ex, ey);
    const inside = meanAlong(frame, cx, cy, ex, ey, BLOBS.inside[0], BLOBS.inside[1], len);
    const outside = meanAlong(frame, cx, cy, ex, ey, BLOBS.margin[0], BLOBS.margin[1], len);
    insideSum += inside;
    outsideSum += outside;
    contrasts.push(inside - outside);
    ratios.push((outside + 4) / (inside + 4));
    const width = edgeWidth(frame, cx, cy, ex, ey, len, inside, outside);
    if (width !== null) widths.push(width);
  }
  const contrast = median(contrasts);
  if (contrast < BLOBS.minContrast) return null;
  if (contrasts.filter((c) => c > BLOBS.minRayContrast).length < BLOBS.minRaysWithContrast * n) return null;
  if (median(ratios) > BLOBS.maxMarginRatio) return null;

  // Brightness-weighted centroid over the dot and its margin: weight 0 at the
  // margin's level, 1 at the inside's. Works for a mask that came out hollow.
  const inLevel = insideSum / n, outLevel = outsideSum / n;
  const span = inLevel - outLevel;
  const { width: w, height: h, data } = frame;
  const xa = Math.max(0, Math.floor(cx - 1.3 * rx)), xb = Math.min(w - 1, Math.ceil(cx + 1.3 * rx));
  const ya = Math.max(0, Math.floor(cy - 1.3 * ry)), yb = Math.min(h - 1, Math.ceil(cy + 1.3 * ry));
  let sw = 0, sx = 0, sy = 0;
  for (let y = ya; y <= yb; y++) {
    for (let x = xa; x <= xb; x++) {
      const u = (x - cx) / (1.3 * rx), v = (y - cy) / (1.3 * ry);
      if (u * u + v * v > 1) continue;                       // stay inside the margin: beyond it may be bright table
      const wgt = Math.min(1, Math.max(0, (data[y * w + x]! - outLevel) / span));
      sw += wgt;
      sx += wgt * x;
      sy += wgt * y;
    }
  }
  if (sw <= 0) return null;
  // The weights sum to the dot's area in pixels.
  return { x: sx / sw, y: sy / sw, d: 2 * Math.sqrt(sw / Math.PI), contrast, sharpness: sharpnessOf(widths) };
}

/** Edge sharpness, 0-1, from the widths (px) over which rays cross from 75% to
 * 25% of the dot's brightness. A step edge, sampled between pixels, measures
 * about 0.7 px: that is as sharp as a work frame gets (1). A blur that spreads
 * the edge over w work pixels gives 0.7 / w. */
export function sharpnessOf(widths: number[]): number {
  if (widths.length === 0) return 0;
  return Math.min(1, 0.7 / Math.max(median(widths), 1e-6));
}

function meanAlong(f: GrayFrame, cx: number, cy: number, ex: number, ey: number, from: number, to: number, len: number): number {
  const steps = Math.max(2, Math.ceil((to - from) * len));
  let sum = 0;
  for (let i = 0; i <= steps; i++) {
    const s = from + ((to - from) * i) / steps;
    sum += sample(f, cx + s * ex, cy + s * ey);
  }
  return sum / (steps + 1);
}

/** Distance in px along a ray between the 75% and 25% levels of the edge, or null if it has no clean edge. */
function edgeWidth(f: GrayFrame, cx: number, cy: number, ex: number, ey: number, len: number, inside: number, outside: number): number | null {
  if (inside - outside < BLOBS.minRayContrast) return null;
  const hiLevel = outside + 0.75 * (inside - outside), loLevel = outside + 0.25 * (inside - outside);
  const step = 0.25 / len;                                   // a quarter pixel, in radii
  let at75: number | null = null, prev = sample(f, cx + 0.5 * ex, cy + 0.5 * ey), prevS = 0.5;
  for (let s = 0.5 + step; s <= 1.5; s += step) {
    const v = sample(f, cx + s * ex, cy + s * ey);
    if (at75 === null && prev >= hiLevel && v < hiLevel) at75 = prevS + ((prev - hiLevel) / (prev - v)) * step;
    if (at75 !== null && prev >= loLevel && v < loLevel) {
      const at25 = prevS + ((prev - loLevel) / (prev - v)) * step;
      return (at25 - at75) * len;
    }
    prev = v;
    prevS = s;
  }
  return null;
}

/** Bilinear sample, the edge pixels repeated outside the frame. */
function sample(f: GrayFrame, x: number, y: number): number {
  const { width: w, height: h, data } = f;
  const xc = Math.min(w - 1, Math.max(0, x)), yc = Math.min(h - 1, Math.max(0, y));
  const x0 = Math.floor(xc), y0 = Math.floor(yc);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = xc - x0, fy = yc - y0;
  const a = data[y0 * w + x0]!, b = data[y0 * w + x1]!, c = data[y1 * w + x0]!, d = data[y1 * w + x1]!;
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}
