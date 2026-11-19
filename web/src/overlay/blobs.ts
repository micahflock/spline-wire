// The overlay's frame detector: light dots on dark links, found fast.
//
// This is not the measuring detector (splinewire/detect.py), which fits
// ellipses to sub-pixel edges at full resolution and takes about a second. It
// follows the same idea at one coarse scale, cheaply enough for every video
// frame, and only has to be good enough to coach the shot:
//
//   1. Work on a small gray frame: the region of the video round the chain
//      (region.ts), drawn so that its fiducials are a set size, or the whole
//      frame or a quarter of it while there is no chain.
//   2. Threshold against the local mean, as detect._masks does (a 25 px
//      window, the pixel at least 4 gray levels above it): both masks (light
//      and dark) in one pass, into buffers kept between frames.
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

/** The whole frame's work size until 2026-11-19, and still the scale
 * sharpness is read at (Window.refScale), so that the coach's limits keep
 * their meaning: the camera view now draws regions (region.ts). */
export const WORK_LONG_SIDE = 640;
/** Thin rims need more pixels: at 640 the snap-fit chain's rim is about one
 * pixel wide and breaks, so an end pin's hole leaks into the link; at 800 all
 * 13 rims of a synthetic snap-chain frame are found (17 ms, against 12 of 13). */
export const WORK_LONG_SIDE_RINGS = 800;

/** A ring whose hole is at least this share of its outer diameter is a thin rim,
 * found by its hole alone (detect.py's THIN_RING). */
export const THIN_RING = 0.6;

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
  /** A ring's margin, in its outer radii: just outside the rim (see ringAt). */
  ringMargin: [1.08, 1.2] as const,
  /** The share of rays on which a ring's margin must be darker than its rim. */
  ringMarginRays: 0.6,
  /** A thin rim's hole edge is this close to the hole's radius (a share of it) on
   * a ray: a 40 degree tilt, seen through an axis-aligned box, varies it by ~13%. */
  holeRoundness: 0.25,
  /** A thin rim's least minor/major ratio (a 60 degree tilt). */
  minRimAspect: 0.5,
  /** Rays on which a thin rim must be brighter than what lies just outside it. The
   * snap-fit chain's arms are 0.57 ring radii wide: about 2.7 of 16 rays each, head on;
   * a tilt shows the dark side of the head too. */
  minRimBandRays: 2,
  /** Median inside-minus-margin contrast, gray levels; the share of rays that must show any. */
  minContrast: 15,
  minRayContrast: 6,
  minRaysWithContrast: 0.8,
  /** The margin is at most this fraction of the inside, on the median ray. */
  maxMarginRatio: 0.92,
  /** Stop after this many candidates: a busy frame is not worth more time. */
  maxCandidates: 400,
};

/** RGBA pixels (ImageData.data) to gray, by the usual luma weights; into
 * `into` (at least width × height long) if given, which is then reused. */
export function toGray(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number, into?: Uint8Array): GrayFrame {
  const data = into ? into.subarray(0, width * height) : new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i++, j += 4) {
    data[i] = (rgba[j]! * 77 + rgba[j + 1]! * 150 + rgba[j + 2]! * 29) >> 8;
  }
  return { width, height, data };
}

/** The size to draw a frame at for detectBlobs, and the factor back to frame pixels. */
export function workSize(width: number, height: number, shape: Shape = { inner: 0 }): { width: number; height: number; scale: number } {
  const scale = Math.max(1, Math.max(width, height) / (shape.inner > 0 ? WORK_LONG_SIDE_RINGS : WORK_LONG_SIDE));
  return { width: Math.round(width / scale), height: Math.round(height / scale), scale };
}

/** What the fiducials look like: `inner` is a ring's hole diameter over its
 * outer diameter (ChainSpec ring_inner_mm / fiducial_mm), 0 for a solid dot;
 * `dark`, that they are darker than the links (ChainSpec fiducial_shade
 * "dark"): the frame is then read inverted, as detect.py reads such a photo. */
export interface Shape {
  inner: number;
  dark?: boolean;
}

/** The fiducial shape of a chain preset. */
export function shapeOf(chain: { fiducial?: string; fiducial_mm: number; ring_inner_mm?: number; fiducial_shade?: string }): Shape {
  const shape: Shape = chain.fiducial === "ring" && chain.ring_inner_mm ? { inner: chain.ring_inner_mm / chain.fiducial_mm } : { inner: 0 };
  if (chain.fiducial_shade === "dark") shape.dark = true;
  return shape;
}

/** The frame with its gray levels turned round, into a buffer kept between frames. */
function inverted(frame: GrayFrame): GrayFrame {
  const n = frame.data.length;
  if (scratch.inv.length < n) scratch.inv = new Uint8Array(n);
  const data = scratch.inv.subarray(0, n);
  for (let i = 0; i < n; i++) data[i] = 255 - frame.data[i]!;
  return { width: frame.width, height: frame.height, data };
}

/** Where a work frame was drawn from: the top-left corner of its source
 * rectangle in the video frame's pixels, and video pixels per work pixel
 * across and down. `refScale`, video pixels per work pixel of the frame
 * sharpness is read at (default: the mean of scaleX and scaleY): edge widths
 * are converted to that frame's pixels, so that a region drawn finer than the
 * full-frame work frame (overlay/region.ts) reads as sharp as the coach's
 * limits expect (they were calibrated on full frames at WORK_LONG_SIDE). */
export interface Window {
  x: number;
  y: number;
  scaleX: number;
  scaleY: number;
  refScale?: number;
}

/** Dots in a gray work frame. `scale` multiplies positions and sizes back
 * into the pixels of the frame the work frame was drawn from; a Window says
 * where in that frame a region was drawn from as well. */
export function detectBlobs(frame: GrayFrame, scale: number | Window = 1, shape: Shape = { inner: 0 }): Blob[] {
  const win: Window = typeof scale === "number" ? { x: 0, y: 0, scaleX: scale, scaleY: scale } : scale;
  const mean = (win.scaleX + win.scaleY) / 2;
  const widthScale = mean / (win.refScale ?? mean);
  if (shape.dark) frame = inverted(frame);
  const { width: w, height: h } = frame;
  if (w < BLOBS.block || h < BLOBS.block) return [];
  const found: Blob[] = [];
  const keep = (blob: Blob | null) => {
    // A ring can be found twice (by its hole and by its rim): keep the first.
    if (blob && !found.some((f) => Math.hypot(f.x - blob.x, f.y - blob.y) < 0.4 * Math.max(f.d, blob.d))) found.push(blob);
  };
  const masks = thresholds(frame, shape.inner > 0);
  if (shape.inner > 0) {
    // Rings are found by their holes first, as detect.py finds rims: a hole is
    // a clean dark disc, while a thin rim often merges with a light table in
    // the mask wherever the link around it is narrow or missing (the snap-fit
    // chain's rim is as wide as its head: only the arms show outside it).
    // Closed rims are a second source.
    for (const c of components(masks.dark!, w, h)) {
      keep(ringAt(frame, c.cx, c.cy, c.rx / shape.inner, c.ry / shape.inner, shape.inner));
    }
    for (const c of components(masks.bright, w, h)) keep(ringAt(frame, c.cx, c.cy, c.rx, c.ry, shape.inner));
  } else {
    for (const c of components(masks.bright, w, h)) keep(dotAt(frame, c.cx, c.cy, c.rx, c.ry));
  }
  return found.map((b) => ({ x: win.x + (b.x + 0.5) * win.scaleX - 0.5, y: win.y + (b.y + 0.5) * win.scaleY - 0.5, d: b.d * mean,
    contrast: b.contrast, sharpness: widthScale === 1 ? b.sharpness : rescaleSharpness(b.sharpness, widthScale) }));
}

interface Box {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

/** Buffers kept from frame to frame (detectBlobs runs on one thread, one
 * frame at a time): the threshold's column sums, the two masks, the components'
 * stack and an inverted frame. A frame of the same size or smaller allocates
 * nothing. */
const scratch = {
  cols: new Int32Array(0),
  bright: new Uint8Array(0),
  dark: new Uint8Array(0),
  stack: new Int32Array(0),
  inv: new Uint8Array(0),
};

/** Both of localThreshold's masks in one pass: `bright` (dark = false) and,
 * with `withDark`, `dark` (dark = true), each equal to localThreshold's pixel
 * for pixel. The window sums are kept running instead of read from a
 * summed-area table: a column sum per x, moved down a row at a time, and a
 * sum of those moved along each row (integers, so the sums are the same).
 * The masks are laid out with a border of one zero pixel all round (row
 * stride width + 2; pixel (x, y) at (y + 1) * (width + 2) + x + 1) so that
 * components needs no bounds checks. The arrays are scratch buffers: the next
 * call overwrites them. */
export function thresholds(frame: GrayFrame, withDark: boolean): { bright: Uint8Array; dark: Uint8Array | null; stride: number } {
  const { width: w, height: h, data } = frame;
  const W = w + 2, padded = W * (h + 2);
  if (scratch.cols.length < w) scratch.cols = new Int32Array(w);
  if (scratch.bright.length < padded) {
    scratch.bright = new Uint8Array(padded);
    scratch.dark = new Uint8Array(padded);
  }
  const cols = scratch.cols, mb = scratch.bright, md = scratch.dark;
  const r = (BLOBS.block - 1) / 2, off = BLOBS.offset;
  // Column sums over rows 0..r: the window of row 0.
  cols.fill(0, 0, w);
  for (let y = 0; y <= r && y < h; y++) for (let x = 0, d = y * w; x < w; x++) cols[x] = cols[x]! + data[d + x]!;
  for (let y = 0; y < h; y++) {
    if (y > 0) {
      // Move the column sums down a row: row y + r comes in, row y - r - 1 goes out.
      const enter = y + r, leave = y - r - 1;
      if (enter < h) for (let x = 0, d = enter * w; x < w; x++) cols[x] = cols[x]! + data[d + x]!;
      if (leave >= 0) for (let x = 0, d = leave * w; x < w; x++) cols[x] = cols[x]! - data[d + x]!;
    }
    const rows = (y + r + 1 > h ? h : y + r + 1) - (y - r < 0 ? 0 : y - r);
    const m = (y + 1) * W + 1, d = y * w;
    let sum = 0;
    for (let x = 0; x < r && x < w; x++) sum += cols[x]!;
    // Three stretches of the row: the window cut by the left edge, whole, cut
    // by the right edge (a frame narrower than the window has no middle).
    const midA = r + 1, midB = Math.max(midA, w - r);
    for (let x = 0; x < w; x++) {
      if (x === midA && midA < midB) {
        // The whole window: no edges to test, one area.
        const area = BLOBS.block * rows;
        if (withDark) {
          for (; x < midB; x++) {
            sum += cols[x + r]! - cols[x - r - 1]!;
            const v = data[d + x]!;
            mb[m + x] = (v - off) * area > sum ? 1 : 0;
            md[m + x] = (v + off) * area < sum ? 1 : 0;
          }
        } else {
          for (; x < midB; x++) {
            sum += cols[x + r]! - cols[x - r - 1]!;
            mb[m + x] = (data[d + x]! - off) * area > sum ? 1 : 0;
          }
        }
        if (x >= w) break;
      }
      if (x + r < w) sum += cols[x + r]!;
      if (x - r - 1 >= 0) sum -= cols[x - r - 1]!;
      const area = ((x + r + 1 > w ? w : x + r + 1) - (x - r < 0 ? 0 : x - r)) * rows, v = data[d + x]!;
      // pixel > mean + offset (dark: pixel < mean - offset), without dividing
      mb[m + x] = (v - off) * area > sum ? 1 : 0;
      if (withDark) md[m + x] = (v + off) * area < sum ? 1 : 0;
    }
  }
  // The zero border (the buffers may hold an earlier frame of another size).
  for (const mask of withDark ? [mb, md] : [mb]) {
    mask.fill(0, 0, W);
    mask.fill(0, (h + 1) * W, padded);
    for (let y = 1; y <= h; y++) mask[y * W] = mask[y * W + W - 1] = 0;
  }
  return { bright: mb.subarray(0, padded), dark: withDark ? md.subarray(0, padded) : null, stride: W };
}

/** The mask's connected components (8-connected) of a plausible size and
 * shape, as boxes: at least BLOBS.minSide across, at most a quarter of the
 * frame and BLOBS.maxAspect times longer than wide. Stops after
 * BLOBS.maxCandidates. The mask is laid out as thresholds() makes it (a zero
 * border round a w × h frame) and is used up (visited pixels are set to 2). */
function components(mask: Uint8Array, w: number, h: number): Box[] {
  const out: Box[] = [];
  const W = w + 2;
  if (scratch.stack.length < w * h) scratch.stack = new Int32Array(w * h);
  const stack = scratch.stack;
  const maxSide = Math.max(w, h) / 4;
  // Read in row order, four pixels at a time: most of a mask is 0, and the
  // border is, so the padded buffer is read straight through.
  const words = new Uint32Array(mask.buffer, mask.byteOffset, mask.byteOffset % 4 ? 0 : mask.length >> 2);
  for (let i = 0, start = 0; start < mask.length; i++) {
    if (i < words.length && words[i] === 0) {
      start += 4;
      continue;
    }
    for (const end = i < words.length ? start + 4 : start + 1; start < end; start++) {
      if (mask[start] !== 1) continue;
      let top = 0, x0 = W, y0 = h + 2, x1 = -1, y1 = -1;
      stack[top++] = start;
      mask[start] = 2;
      while (top > 0) {
        const p = stack[--top]!;
        const py = (p / W) | 0, px = p - py * W;
        if (px < x0) x0 = px;
        if (px > x1) x1 = px;
        if (py < y0) y0 = py;
        if (py > y1) y1 = py;
        // The eight neighbours; the zero border stops the fill at the frame's edge.
        let q = p - W - 1;
        if (mask[q] === 1) { mask[q] = 2; stack[top++] = q; }
        if (mask[++q] === 1) { mask[q] = 2; stack[top++] = q; }
        if (mask[++q] === 1) { mask[q] = 2; stack[top++] = q; }
        q = p - 1;
        if (mask[q] === 1) { mask[q] = 2; stack[top++] = q; }
        q = p + 1;
        if (mask[q] === 1) { mask[q] = 2; stack[top++] = q; }
        q = p + W - 1;
        if (mask[q] === 1) { mask[q] = 2; stack[top++] = q; }
        if (mask[++q] === 1) { mask[q] = 2; stack[top++] = q; }
        if (mask[++q] === 1) { mask[q] = 2; stack[top++] = q; }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      const lo = Math.min(bw, bh), hi = Math.max(bw, bh);
      if (lo < BLOBS.minSide || hi > maxSide || hi > BLOBS.maxAspect * lo) continue;
      // Padded to frame pixels: one less in each direction.
      out.push({ cx: (x0 + x1) / 2 - 1, cy: (y0 + y1) / 2 - 1, rx: bw / 2, ry: bh / 2 });
      if (out.length >= BLOBS.maxCandidates) return out;
    }
  }
  return out;
}

/** 1 where a pixel is above its neighbourhood's mean by BLOBS.offset (with
 * `dark`, below it by as much), else 0. The reference that thresholds() must
 * equal; detectBlobs uses thresholds(). */
export function localThreshold(frame: GrayFrame, dark = false): Uint8Array {
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
      // pixel > mean + offset (dark: pixel < mean - offset), without dividing
      const area = (xb - xa) * (yb - ya), v = data[y * w + x]!;
      if (dark ? (v + BLOBS.offset) * area < sum : (v - BLOBS.offset) * area > sum) mask[y * w + x] = 1;
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

/** Is the bright thing in this box a ring (a light rim around a darker hole)
 * on a link? The ring's own test, as detect.py's rings and rims are found by
 * their hole: along rays from the centre, the rim is brighter than the hole
 * inside it, and a wide-banded ring's also than the link's margin outside it.
 * A solid dot fails (no hole), and so does a bright spot of glare. A thin rim
 * (hole at least THIN_RING of the outer diameter, as detect.py's detect_rims:
 * the snap-fit chain's, 0.80) is a few work pixels wide, so the rim is read as
 * the brightest sample across its band, not a mean; and nothing dark need lie
 * outside it. The snap-fit chain's rim is as wide as the head it sits on, and
 * the next link's sleeve is hidden under that head, so outside the rim the
 * camera sees the table wherever the arms are not: on white paper the rim has
 * no outer edge. Such a rim is taken by its hole alone and centred on the
 * hole, as detect_rims does. The box is the ring's outer edge. */
export function ringAt(frame: GrayFrame, cx: number, cy: number, rx: number, ry: number, inner: number): Blob | null {
  const n = BLOBS.rays;
  const thin = inner >= THIN_RING;
  const rimHole: number[] = [], rimMargin: number[] = [], ratios: number[] = [], widths: number[] = [];
  let rimSum = 0, marginSum = 0, holeSum = 0, round = 0;
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n;
    const ex = rx * Math.cos(a), ey = ry * Math.sin(a);
    const len = Math.hypot(ex, ey);
    const hole = meanAlong(frame, cx, cy, ex, ey, 0.1, Math.max(0.15, 0.7 * inner), len);
    const rim = maxAlong(frame, cx, cy, ex, ey, 0.85 * inner, 1.05, len);
    // Read the link just outside the rim: on a ring's chain the link may be
    // barely wider than the ring, so the dot's margin band (1.15-1.45) would
    // reach the table.
    const margin = meanAlong(frame, cx, cy, ex, ey, BLOBS.ringMargin[0], BLOBS.ringMargin[1], len);
    rimSum += rim;
    marginSum += margin;
    holeSum += hole;
    rimHole.push(rim - hole);
    rimMargin.push(rim - margin);
    ratios.push((margin + 4) / (rim + 4));
    // A thin rim's sharp edge is its hole's, which every ray crosses where the
    // hole's outline is: about `inner` out, the box being the ring's.
    const width = thin ? edgeWidth(frame, cx, cy, ex, ey, len, rim, hole, (1 + inner) / 2, 0.4 * inner)
      : edgeWidth(frame, cx, cy, ex, ey, len, rim, margin);
    if (width !== null) widths.push(width);
    if (thin) {
      const edge = crossing(frame, cx, cy, ex, ey, len, (rim + hole) / 2, 0.3 * inner, 1.05);
      if (edge !== null && Math.abs(edge / inner - 1) < BLOBS.holeRoundness) round++;
    }
  }
  // The hole is what makes a ring (detect.py finds rims by it): it must show on
  // nearly every ray. A wide-banded ring's dark link outside the rim must show
  // on most, as glare or a tilt can wash out one side.
  const share = (v: number[]) => v.filter((c) => c > BLOBS.minRayContrast).length / n;
  if (median(rimHole) < BLOBS.minContrast || share(rimHole) < BLOBS.minRaysWithContrast) return null;
  const contrast = thin ? median(rimHole) : median(rimMargin);
  if (thin) {
    // A rim's hole is round, or an ellipse no narrower than a 60 degree tilt
    // makes it: not a dark arm between two rims.
    if (Math.min(rx, ry) < BLOBS.minRimAspect * Math.max(rx, ry) || round < BLOBS.minRaysWithContrast * n) return null;
    // Some dark link lies just outside the rim somewhere (an arm, at least):
    // a dark speck of the hole's size on a plain table has no rim round it.
    if (share(rimMargin) * n < BLOBS.minRimBandRays) return null;
  }
  if (!thin) {
    if (contrast < BLOBS.minContrast || share(rimMargin) < BLOBS.ringMarginRays) return null;
    if (median(ratios) > BLOBS.maxMarginRatio) return null;
  }

  // Centre: a thin rim's, darkness-weighted over its hole out to the middle of
  // the rim; a wide ring's, brightness-weighted over its rim's band. Either is
  // there all round, so the weights balance about the centre as a dot's do.
  const rimLevel = rimSum / n;
  const outLevel = thin ? holeSum / n : marginSum / n, span = rimLevel - outLevel;
  const { width: w, height: h, data } = frame;
  const xa = Math.max(0, Math.floor(cx - 1.2 * rx)), xb = Math.min(w - 1, Math.ceil(cx + 1.2 * rx));
  const ya = Math.max(0, Math.floor(cy - 1.2 * ry)), yb = Math.min(h - 1, Math.ceil(cy + 1.2 * ry));
  let sw = 0, sx = 0, sy = 0;
  for (let y = ya; y <= yb; y++) {
    for (let x = xa; x <= xb; x++) {
      const u = (x - cx) / rx, v = (y - cy) / ry, r = Math.sqrt(u * u + v * v);
      let wgt: number;
      if (thin) {
        if (r > (1 + inner) / 2) continue;                 // beyond the middle of the rim
        wgt = Math.min(1, Math.max(0, (rimLevel - data[y * w + x]!) / span));
      } else {
        if (r < 0.6 * inner || r > 1.2) continue;          // the hole and beyond the margin are not the rim
        wgt = Math.min(1, Math.max(0, (data[y * w + x]! - outLevel) / span));
      }
      sw += wgt;
      sx += wgt * x;
      sy += wgt * y;
    }
  }
  if (sw <= 0) return null;
  // The rim's area says little about the ring's size; its outer edge, the box, does.
  return { x: sx / sw, y: sy / sw, d: rx + ry, contrast, sharpness: sharpnessOf(widths) };
}

/** Where along a ray (in radii) the gray level first reaches `level`, reading outwards from `from` to `to`. */
function crossing(f: GrayFrame, cx: number, cy: number, ex: number, ey: number, len: number, level: number,
  from: number, to: number): number | null {
  const step = 0.25 / len;                                   // a quarter pixel, in radii
  let prev = sample(f, cx + from * ex, cy + from * ey);
  for (let s = from + step; s <= to; s += step) {
    const v = sample(f, cx + s * ex, cy + s * ey);
    if (prev < level && v >= level) return s - step + ((level - prev) / (v - prev)) * step;
    prev = v;
  }
  return null;
}

function maxAlong(f: GrayFrame, cx: number, cy: number, ex: number, ey: number, from: number, to: number, len: number): number {
  const steps = Math.max(3, Math.ceil((to - from) * len * 2));
  let best = -Infinity;
  for (let i = 0; i <= steps; i++) {
    const s = from + ((to - from) * i) / steps;
    best = Math.max(best, sample(f, cx + s * ex, cy + s * ey));
  }
  return best;
}

/** Edge sharpness, 0-1, from the widths (px) over which rays cross from 75% to
 * 25% of the dot's brightness. A step edge, sampled between pixels, measures
 * about 0.7 px: that is as sharp as a work frame gets (1). A blur that spreads
 * the edge over w work pixels gives 0.7 / w. */
export function sharpnessOf(widths: number[]): number {
  if (widths.length === 0) return 0;
  return Math.min(1, 0.7 / Math.max(median(widths), 1e-6));
}

/** A sharpness read in one frame's pixels, read in pixels `factor` times as
 * large (edges `factor` times narrower). sharpnessOf is 0.7 / width, capped at 1:
 * a capped value stays 1 (its edge was at most a step), and is not made sharper. */
export function rescaleSharpness(sharpness: number, factor: number): number {
  if (sharpness >= 1 || sharpness <= 0) return sharpness;
  return Math.min(1, sharpness / factor);
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

/** Distance in px along a ray between the 75% and 25% levels of the edge, or
 * null if it has no clean edge. The ray is read from `from` to `to` radii (outwards
 * by default; inwards for a hole), going from the bright `inside` level to `outside`. */
function edgeWidth(f: GrayFrame, cx: number, cy: number, ex: number, ey: number, len: number, inside: number, outside: number,
  from = 0.5, to = 1.5): number | null {
  if (inside - outside < BLOBS.minRayContrast) return null;
  const hiLevel = outside + 0.75 * (inside - outside), loLevel = outside + 0.25 * (inside - outside);
  const step = (to > from ? 0.25 : -0.25) / len;             // a quarter pixel, in radii
  let at75: number | null = null, prev = sample(f, cx + from * ex, cy + from * ey), prevS = from;
  for (let s = from + step; to > from ? s <= to : s >= to; s += step) {
    const v = sample(f, cx + s * ex, cy + s * ey);
    if (at75 === null && prev >= hiLevel && v < hiLevel) at75 = prevS + ((prev - hiLevel) / (prev - v)) * step;
    if (at75 !== null && prev >= loLevel && v < loLevel) {
      const at25 = prevS + ((prev - loLevel) / (prev - v)) * step;
      return Math.abs(at25 - at75) * len;
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
