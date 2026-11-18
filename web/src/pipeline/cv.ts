// The OpenCV operations splinewire/detect.py uses, rebuilt so they give what
// OpenCV 5.0 (the version uv.lock pins) gives: bit for bit where the
// arithmetic is integer or float32 (blur, area resize, thresholds, box
// filters, remap, labels, contours), to rounding where OpenCV solves a linear
// system (the ellipse fits). cv.test.ts checks each against cv2's own output.
// Read from OpenCV's sources (modules/imgproc/src: smooth, resize, thresh,
// box_filter, contours_new, shapedescr) and confirmed on cv2's results.

export interface Gray {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface GrayF {
  width: number;
  height: number;
  data: Float32Array;
}

// ---------------------------------------------------------------------------
// GaussianBlur, 8-bit (OpenCV's bit-exact fixed-point path)

/** The kernel OpenCV uses for an 8-bit image, in 1/256ths: the Gaussian's
 * taps rounded from the ends inward with the error carried along, the centre
 * taking what is left so the taps sum to 256 (getGaussianKernelFixedPoint_ED). */
export function gaussianKernel8(sigma: number): number[] {
  const n = Math.round(sigma * 3 * 2 + 1) | 1;          // ksize from sigma, 8-bit images
  const half = (n - 1) / 2;
  const k: number[] = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - half;
    k.push(Math.exp(-(x * x) / (2 * sigma * sigma)));
    sum += k[i]!;
  }
  const out = new Array<number>(n).fill(0);
  let err = 0, total = 0;
  for (let i = 0; i < n >> 1; i++) {
    const adj = (k[i]! / sum) * 256 + err;
    const v = roundEven(adj);
    err = adj - v;
    out[i] = out[n - 1 - i] = v;
    total += v;
  }
  out[n >> 1] = 256 - 2 * total;
  return out;
}

/** cv2.GaussianBlur(img, (0, 0), sigma) on an 8-bit image: rows then columns
 * in integers, rounded once, borders reflected about the edge pixel. */
export function gaussianBlur8(img: Gray, sigma: number): Gray {
  const k = gaussianKernel8(sigma);
  const r = k.length >> 1;
  const { width: w, height: h, data } = img;
  const rows = new Uint32Array(w * h);
  const xs = reflectIndex(w, r);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += k[i]! * data[row + xs[x + i]!]!;
      rows[row + x] = s;
    }
  }
  const ys = reflectIndex(h, r);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += k[i]! * rows[ys[y + i]! * w + x]!;
      out[y * w + x] = (s + 32768) >>> 16;
    }
  }
  return { width: w, height: h, data: out };
}

/** Index of each padded position (-r .. n-1+r) under BORDER_REFLECT_101. */
function reflectIndex(n: number, r: number): Int32Array {
  const out = new Int32Array(n + 2 * r);
  for (let i = -r; i < n + r; i++) {
    let j = i;
    if (n === 1) j = 0;
    else {
      while (j < 0 || j >= n) j = j < 0 ? -j : 2 * (n - 1) - j;
    }
    out[i + r] = j;
  }
  return out;
}

// ---------------------------------------------------------------------------
// resize to half, INTER_AREA

/** cv2.resize(img, (w // 2, h // 2), interpolation=INTER_AREA). Even sizes:
 * OpenCV averages 2x2 blocks in integers with rounding. Odd sizes: its
 * general area path, each output pixel a float32 weighted sum of the source
 * pixels its cell covers, rows first (resizeArea_<uchar, float>). */
export function halveArea8(img: Gray): Gray {
  const { width: w, height: h, data } = img;
  const W = w >> 1, H = h >> 1;
  const out = new Uint8Array(W * H);
  if (w % 2 === 0 && h % 2 === 0) {
    for (let y = 0; y < H; y++) {
      const a = 2 * y * w, b = a + w;
      for (let x = 0; x < W; x++) {
        out[y * W + x] = (data[a + 2 * x]! + data[a + 2 * x + 1]! + data[b + 2 * x]! + data[b + 2 * x + 1]! + 2) >> 2;
      }
    }
    return { width: W, height: H, data: out };
  }
  const f = Math.fround;
  const xtab = areaTab(w, W, 1 / (W / w));
  const ytab = areaTab(h, H, 1 / (H / h));
  const buf = new Float32Array(W);
  const sum = new Float32Array(W);
  let prev = ytab[0]!.di;
  const store = (dy: number) => {
    for (let x = 0; x < W; x++) out[dy * W + x] = Math.min(255, Math.max(0, roundEven(sum[x]!)));
  };
  for (const { di: dy, si: sy, alpha: beta } of ytab) {
    buf.fill(0);
    const row = sy * w;
    for (const { di, si, alpha } of xtab) buf[di] = f(buf[di]! + f(data[row + si]! * alpha));
    if (dy !== prev) {
      store(prev);
      for (let x = 0; x < W; x++) sum[x] = f(beta * buf[x]!);
      prev = dy;
    } else {
      for (let x = 0; x < W; x++) sum[x] = f(sum[x]! + f(beta * buf[x]!));
    }
  }
  store(prev);
  return { width: W, height: H, data: out };
}

/** OpenCV's computeResizeAreaTab: which source pixels each output pixel's
 * cell covers, and by how much (float32 weights). */
function areaTab(ssize: number, dsize: number, scale: number): { di: number; si: number; alpha: number }[] {
  const tab: { di: number; si: number; alpha: number }[] = [];
  const f = Math.fround;
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cell = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.push({ di: dx, si: sx1 - 1, alpha: f((sx1 - fsx1) / cell) });
    for (let sx = sx1; sx < sx2; sx++) tab.push({ di: dx, si: sx, alpha: f(1.0 / cell) });
    if (fsx2 - sx2 > 1e-3) tab.push({ di: dx, si: sx2, alpha: f(Math.min(Math.min(fsx2 - sx2, 1), cell) / cell) });
  }
  return tab;
}

// ---------------------------------------------------------------------------
// Thresholds (detect._masks)

/** The sum over the (2r+1)^2 box about every pixel, the image padded with
 * the given border rule. Integer images only (8-bit values or their squares),
 * so every sum is exact and the order of adding does not matter: rows first,
 * then a running sum down each column. */
function boxSums(values: ArrayLike<number>, w: number, h: number, r: number, border: "replicate" | "reflect101"): Int32Array {
  const xs = border === "replicate" ? clampIndex(w, r) : reflectIndex(w, r);
  const ys = border === "replicate" ? clampIndex(h, r) : reflectIndex(h, r);
  const k = 2 * r + 1;
  const rows = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    const src = y * w;
    let s = 0;
    for (let i = 0; i < k; i++) s += values[src + xs[i]!]!;
    rows[src] = s;
    for (let x = 1; x < w; x++) {
      s += values[src + xs[x + k - 1]!]! - values[src + xs[x - 1]!]!;
      rows[src + x] = s;
    }
  }
  const out = new Int32Array(w * h);
  const acc = new Int32Array(w);
  for (let i = 0; i < k; i++) {
    const row = ys[i]! * w;
    for (let x = 0; x < w; x++) acc[x] = acc[x]! + rows[row + x]!;
  }
  out.set(acc, 0);
  for (let y = 1; y < h; y++) {
    const add = ys[y + k - 1]! * w, sub = ys[y - 1]! * w;
    for (let x = 0; x < w; x++) {
      acc[x] = acc[x]! + rows[add + x]! - rows[sub + x]!;
      out[y * w + x] = acc[x]!;
    }
  }
  return out;
}

function clampIndex(n: number, r: number): Int32Array {
  const out = new Int32Array(n + 2 * r);
  for (let i = -r; i < n + r; i++) out[i + r] = Math.min(n - 1, Math.max(0, i));
  return out;
}

/** cv2.adaptiveThreshold(img, 255, MEAN_C, THRESH_BINARY, block, c): 255
 * where the pixel exceeds its block's mean (8-bit, rounded) by more than -c. */
export function adaptiveMeanThreshold(img: Gray, block: number, c: number): Uint8Array {
  const { width: w, height: h, data } = img;
  const sums = boxSums(data, w, h, block >> 1, "replicate");
  const inv = 1 / (block * block);
  const idelta = Math.ceil(c);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const mean = roundEven(sums[i]! * inv);
    out[i] = data[i]! - mean > -idelta ? 255 : 0;
  }
  return out;
}

/** detect._masks' third mask: well above the local mean (by 0.8 local
 * standard deviations, float32 as NumPy computes it) where the window has
 * contrast at all (std over 6). */
export function brightMask(img: Gray, block: number): Uint8Array {
  const { width: w, height: h, data } = img;
  const sq = new Int32Array(w * h);
  for (let i = 0; i < w * h; i++) sq[i] = data[i]! * data[i]!;
  const r = block >> 1;
  const s1 = boxSums(data, w, h, r, "reflect101");
  const s2 = boxSums(sq, w, h, r, "reflect101");
  const inv = 1 / (block * block);
  const f = Math.fround;
  const k = f(0.8);
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const mean = f(s1[i]! * inv);
    const var_ = f(f(s2[i]! * inv) - f(mean * mean));
    const std = f(Math.sqrt(Math.max(var_, 0)));
    out[i] = data[i]! > f(mean + f(k * std)) && std > 6 ? 255 : 0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Connected components (8-connected), with OpenCV's label order and stats

export interface Components {
  count: number;            // labels 1..count-1 are components; 0 is the background
  labels: Int32Array;
  x: Int32Array; y: Int32Array; w: Int32Array; h: Int32Array; area: Int32Array;
}

/** cv2.connectedComponentsWithStats(mask, connectivity=8). Labels are numbered
 * as OpenCV's block-based labelling numbers them: by the first 2x2 block (in
 * raster order of blocks) that holds a pixel of the component. */
export function connectedComponents(mask: Uint8Array, w: number, h: number): Components {
  const labels = new Int32Array(w * h);
  const parent: number[] = [0];
  const find = (a: number): number => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]!]!;
      a = parent[a]!;
    }
    return a;
  };
  const union = (a: number, b: number): number => {
    a = find(a);
    b = find(b);
    if (a === b) return a;
    if (a < b) {
      parent[b] = a;
      return a;
    }
    parent[a] = b;
    return b;
  };
  // Provisional labels in raster order; merged with neighbours above and to the left.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      let l = 0;
      const look = (j: number) => {
        const m = labels[j]!;
        if (m) l = l ? union(l, m) : find(m);
      };
      if (x > 0) look(i - 1);
      if (y > 0) {
        if (x > 0) look(i - w - 1);
        look(i - w);
        if (x + 1 < w) look(i - w + 1);
      }
      if (!l) {
        l = parent.length;
        parent.push(l);
      }
      labels[i] = l;
    }
  }
  // Number the components by their first 2x2 block.
  const first = new Map<number, number>();         // root -> block order key
  for (let by = 0; by < h; by += 2) {
    for (let bx = 0; bx < w; bx += 2) {
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
        const x = bx + dx, y = by + dy;
        if (x >= w || y >= h) continue;
        const l = labels[y * w + x]!;
        if (!l) continue;
        const root = find(l);
        if (!first.has(root)) first.set(root, first.size + 1);
      }
    }
  }
  const count = first.size + 1;
  const x0 = new Int32Array(count).fill(w), y0 = new Int32Array(count).fill(h);
  const x1 = new Int32Array(count).fill(-1), y1 = new Int32Array(count).fill(-1);
  const area = new Int32Array(count);
  const rootLabel = new Int32Array(parent.length);
  for (const [root, n] of first) rootLabel[root] = n;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let n = 0;
      if (labels[i]) n = rootLabel[find(labels[i]!)]!;
      labels[i] = n;
      if (x < x0[n]!) x0[n] = x;
      if (x > x1[n]!) x1[n] = x;
      if (y < y0[n]!) y0[n] = y;
      if (y > y1[n]!) y1[n] = y;
      area[n]!++;
    }
  }
  const ww = new Int32Array(count), hh = new Int32Array(count);
  for (let n = 0; n < count; n++) {
    ww[n] = x1[n]! - x0[n]! + 1;
    hh[n] = y1[n]! - y0[n]! + 1;
  }
  return { count, labels, x: x0, y: y0, w: ww, h: hh, area };
}

// ---------------------------------------------------------------------------
// findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE) for one component

const DX = [1, 1, 0, -1, -1, -1, 0, 1];
const DY = [0, -1, -1, -1, 0, 1, 1, 1];

/** The outer border of component `label` inside its box, as OpenCV's
 * findContours traces it (Suzuki's border following, every pixel kept): from
 * the component's first pixel in raster order, points [x0, y0, x1, y1, ...]. */
export function outerContour(c: Components, width: number, height: number, label: number): Int32Array {
  const bx = c.x[label]!, by = c.y[label]!, bw = c.w[label]!, bh = c.h[label]!;
  // The component alone in its box, with a 1-pixel border of zeros.
  const W = bw + 2, H = bh + 2;
  const img = new Uint8Array(W * H);
  let start = -1;
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      if (c.labels[(by + y) * width + bx + x] === label) {
        img[(y + 1) * W + x + 1] = 1;
        if (start < 0) start = (y + 1) * W + x + 1;
      }
    }
  }
  void height;
  if (start < 0) return new Int32Array(0);
  const pts: number[] = [];
  const delta = (s: number) => DX[s & 7]! + DY[s & 7]! * W;
  const i0 = start;
  let px = (start % W) - 1 + bx, py = Math.floor(start / W) - 1 + by;
  // First neighbour, looking from direction 3 down to 4 (s_end = 4 for an outer border).
  let s = 4;
  let i1 = 0;
  do {
    s = (s - 1) & 7;
    i1 = i0 + delta(s);
  } while (!img[i1] && s !== 4);
  if (s === 4) return Int32Array.of(px, py);       // a single pixel
  let i3 = i0;
  for (;;) {
    const sEnd = s;
    s = Math.min(s, 15);
    let i4 = 0;
    while (s < 15) {
      s++;
      i4 = i3 + delta(s);
      if (img[i4]) break;
    }
    s &= 7;
    void sEnd;
    pts.push(px, py);
    px += DX[s]!;
    py += DY[s]!;
    if (i4 === i0 && i3 === i1) break;
    i3 = i4;
    s = (s + 4) & 7;
  }
  return Int32Array.from(pts);
}

// ---------------------------------------------------------------------------
// remap, INTER_LINEAR, BORDER_REPLICATE, float32 image

/** cv2.remap at one point (x, y float32) of a float32 image: linear along x,
 * then along y, with fused multiply-adds in float32 as OpenCV 5.0 does it;
 * pixels outside repeat the edge. */
export function sampleLinear(img: GrayF, x: number, y: number): number {
  const { width: w, height: h, data } = img;
  const fx = Math.floor(x), fy = Math.floor(y);
  const ax = Math.fround(x - fx), ay = Math.fround(y - fy);
  const x0 = fx < 0 ? 0 : fx >= w ? w - 1 : fx;
  const x1 = fx + 1 < 0 ? 0 : fx + 1 >= w ? w - 1 : fx + 1;
  const y0 = fy < 0 ? 0 : fy >= h ? h - 1 : fy;
  const y1 = fy + 1 < 0 ? 0 : fy + 1 >= h ? h - 1 : fy + 1;
  const v0 = data[y0 * w + x0]!, v1 = data[y0 * w + x1]!, v2 = data[y1 * w + x0]!, v3 = data[y1 * w + x1]!;
  const top = fmaf(ax, Math.fround(v1 - v0), v0);
  const bottom = fmaf(ax, Math.fround(v3 - v2), v2);
  return fmaf(ay, Math.fround(bottom - top), top);
}

const F32_BUF = new Float32Array(1);
const I32_BUF = new Int32Array(F32_BUF.buffer);

/** fmaf(a, b, c) for float32 values: a*b + c rounded once to float32. The
 * product is exact in a double; the sum is rounded to a double first, which
 * is corrected where that lands exactly half way between two float32s. */
export function fmaf(a: number, b: number, c: number): number {
  const p = a * b;
  const s = p + c;
  const r = Math.fround(s);
  if (s === r) return r;
  // The double sum's own rounding error.
  const bb = s - p;
  const e = (p - (s - bb)) + (c - bb);
  if (e === 0) return r;
  // Is s exactly half way between r and its float32 neighbour?
  F32_BUF[0] = r;
  I32_BUF[0] = I32_BUF[0]! + (s > r === r >= 0 ? 1 : -1);
  const other = F32_BUF[0]!;
  if (s - r !== other - s) return r;
  // Then the true value lies on e's side of the midpoint.
  return (e > 0) === (other > r) ? other : r;
}

// ---------------------------------------------------------------------------
// Ellipse fits

export interface Ellipse {
  cx: number;
  cy: number;
  width: number;     // full axis lengths, width <= height
  height: number;
  angle: number;     // degrees, OpenCV's convention
}

/** cv2.fitEllipse (fitEllipseNoDirect: the conic through the points by
 * linear least squares, then the centre, then the axes), on integer or
 * float32 points [x0, y0, x1, y1, ...]. */
export function fitEllipse(pts: ArrayLike<number>): Ellipse {
  const n = pts.length >> 1;
  if (n < 5) throw new Error("fitEllipse needs at least 5 points");
  if (n === 5) return fitEllipseDirect(pts);
  const f = Math.fround;
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    cx = f(cx + pts[2 * i]!);
    cy = f(cy + pts[2 * i + 1]!);
  }
  cx = f(cx / n);
  cy = f(cy / n);
  let s = 0;
  for (let i = 0; i < n; i++) s += f(Math.abs(f(pts[2 * i]! - cx)) + Math.abs(f(pts[2 * i + 1]! - cy)));
  const scale = 100 / (s > FLT_EPSILON ? s : FLT_EPSILON);
  const A: number[][] = [];
  const b: number[] = [];
  const P: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const px = f(pts[2 * i]! - cx) * scale, py = f(pts[2 * i + 1]! - cy) * scale;
    P.push([px, py]);
    A.push([-px * px, -py * py, -px * py, px, py]);
    b.push(10000);
  }
  const gfp = leastSquares(A, b);
  // The centre, where the conic's gradient vanishes.
  const rp = solve2([[2 * gfp[0]!, gfp[2]!], [gfp[2]!, 2 * gfp[1]!]], [gfp[3]!, gfp[4]!]);
  const A3 = P.map(([px, py]) => [(px - rp[0]) * (px - rp[0]), (py - rp[1]) * (py - rp[1]), (px - rp[0]) * (py - rp[1])]);
  const g = leastSquares(A3, P.map(() => 1));
  const minEps = 1e-8;
  const rot = -0.5 * Math.atan2(g[2]!, g[1]! - g[0]!);
  const t = Math.abs(g[2]!) > minEps ? g[2]! / Math.sin(-2.0 * rot) : g[1]! - g[0]!;
  let r2 = Math.abs(g[0]! + g[1]! - t);
  if (r2 > minEps) r2 = Math.sqrt(2.0 / r2);
  let r3 = Math.abs(g[0]! + g[1]! + t);
  if (r3 > minEps) r3 = Math.sqrt(2.0 / r3);
  let width = f((r2 * 2) / scale), height = f((r3 * 2) / scale);
  let angle = 0;
  if (width > height) {
    [width, height] = [height, width];
    angle = f(90 + (rot * 180) / Math.PI);
  }
  if (angle < -180) angle += 360;
  if (angle > 360) angle -= 360;
  return { cx: f(f(rp[0] / scale) + cx), cy: f(f(rp[1] / scale) + cy), width, height, angle };
}

const FLT_EPSILON = 1.1920928955078125e-7;

/** cv2.fitEllipseDirect (Fitzgibbon's direct least-squares ellipse, through
 * OpenCV's reduced 3x3 eigenproblem), on points [x0, y0, ...] (float32 values). */
export function fitEllipseDirect(pts: ArrayLike<number>): Ellipse {
  const n = pts.length >> 1;
  if (n < 5) throw new Error("fitEllipseDirect needs at least 5 points");
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    cx += pts[2 * i]!;
    cy += pts[2 * i + 1]!;
  }
  cx /= n;
  cy /= n;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.abs(pts[2 * i]! - cx) + Math.abs(pts[2 * i + 1]! - cy);
  const scale = 100 / (s > FLT_EPSILON ? s : FLT_EPSILON);
  // DM = A^T A / n, A's rows [x^2, xy, y^2, x, y, 1].
  const DM = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
  for (let i = 0; i < n; i++) {
    const px = (pts[2 * i]! - cx) * scale, py = (pts[2 * i + 1]! - cy) * scale;
    const row = [px * px, px * py, py * py, px, py, 1];
    for (let a = 0; a < 6; a++) for (let c = 0; c < 6; c++) DM[a]![c] = DM[a]![c]! + row[a]! * row[c]!;
  }
  for (const row of DM) for (let c = 0; c < 6; c++) row[c] = row[c]! / n;
  const D = (a: number, c: number) => DM[a]![c]!;
  // TM / Ts = -S3^-1 S2^T (the linear terms for given quadratic ones), as OpenCV writes it out.
  const TM = [
    [D(0,5)*D(3,5)*D(4,4) - D(0,5)*D(3,4)*D(4,5) - D(0,4)*D(3,5)*D(5,4) + D(0,3)*D(4,5)*D(5,4) + D(0,4)*D(3,4)*D(5,5) - D(0,3)*D(4,4)*D(5,5),
     D(1,5)*D(3,5)*D(4,4) - D(1,5)*D(3,4)*D(4,5) - D(1,4)*D(3,5)*D(5,4) + D(1,3)*D(4,5)*D(5,4) + D(1,4)*D(3,4)*D(5,5) - D(1,3)*D(4,4)*D(5,5),
     D(2,5)*D(3,5)*D(4,4) - D(2,5)*D(3,4)*D(4,5) - D(2,4)*D(3,5)*D(5,4) + D(2,3)*D(4,5)*D(5,4) + D(2,4)*D(3,4)*D(5,5) - D(2,3)*D(4,4)*D(5,5)],
    [D(0,5)*D(3,3)*D(4,5) - D(0,5)*D(3,5)*D(4,3) + D(0,4)*D(3,5)*D(5,3) - D(0,3)*D(4,5)*D(5,3) - D(0,4)*D(3,3)*D(5,5) + D(0,3)*D(4,3)*D(5,5),
     D(1,5)*D(3,3)*D(4,5) - D(1,5)*D(3,5)*D(4,3) + D(1,4)*D(3,5)*D(5,3) - D(1,3)*D(4,5)*D(5,3) - D(1,4)*D(3,3)*D(5,5) + D(1,3)*D(4,3)*D(5,5),
     D(2,5)*D(3,3)*D(4,5) - D(2,5)*D(3,5)*D(4,3) + D(2,4)*D(3,5)*D(5,3) - D(2,3)*D(4,5)*D(5,3) - D(2,4)*D(3,3)*D(5,5) + D(2,3)*D(4,3)*D(5,5)],
    [D(0,5)*D(3,4)*D(4,3) - D(0,5)*D(3,3)*D(4,4) - D(0,4)*D(3,4)*D(5,3) + D(0,3)*D(4,4)*D(5,3) + D(0,4)*D(3,3)*D(5,4) - D(0,3)*D(4,3)*D(5,4),
     D(1,5)*D(3,4)*D(4,3) - D(1,5)*D(3,3)*D(4,4) - D(1,4)*D(3,4)*D(5,3) + D(1,3)*D(4,4)*D(5,3) + D(1,4)*D(3,3)*D(5,4) - D(1,3)*D(4,3)*D(5,4),
     D(2,5)*D(3,4)*D(4,3) - D(2,5)*D(3,3)*D(4,4) - D(2,4)*D(3,4)*D(5,3) + D(2,3)*D(4,4)*D(5,3) + D(2,4)*D(3,3)*D(5,4) - D(2,3)*D(4,3)*D(5,4)],
  ];
  const Ts = -(D(3,5)*D(4,4)*D(5,3)) + D(3,4)*D(4,5)*D(5,3) + D(3,5)*D(4,3)*D(5,4)
    - D(3,3)*D(4,5)*D(5,4) - D(3,4)*D(4,3)*D(5,5) + D(3,3)*D(4,4)*D(5,5);
  if (Math.abs(Ts) < 2.220446049250313e-16) return fitEllipse(pts);   // OpenCV jitters the points; rare
  const T = (a: number, c: number) => TM[a]![c]!;
  const M = [
    [0, 1, 2].map((c) => (D(2, c) + (D(2,3)*T(0,c) + D(2,4)*T(1,c) + D(2,5)*T(2,c)) / Ts) / 2),
    [0, 1, 2].map((c) => -D(1, c) - (D(1,3)*T(0,c) + D(1,4)*T(1,c) + D(1,5)*T(2,c)) / Ts),
    [0, 1, 2].map((c) => (D(0, c) + (D(0,3)*T(0,c) + D(0,4)*T(1,c) + D(0,5)*T(2,c)) / Ts) / 2),
  ];
  // The eigenvector with 4ac - b^2 largest is the ellipse.
  let best: number[] | null = null, bestCond = -Infinity;
  for (const v of eigenvectors3(M)) {
    const cond = 4 * v[0]! * v[2]! - v[1]! * v[1]!;
    if (cond > bestCond) {
      bestCond = cond;
      best = v;
    }
  }
  if (best === null || !(bestCond > 1e-6 * (best[0]! ** 2 + best[1]! ** 2 + best[2]! ** 2))) return fitEllipse(pts);
  let norm = Math.sqrt(best[0]! ** 2 + best[1]! ** 2 + best[2]! ** 2);
  const sg = (v: number) => (v < 0 ? -1 : 1);
  if (sg(best[0]!) * sg(best[1]!) * sg(best[2]!) <= 0) norm = -norm;
  const p = best.map((v) => v / norm);
  const Q = [0, 1, 2].map((r) => (T(r, 0) * p[0]! + T(r, 1) * p[1]! + T(r, 2) * p[2]!) / Ts);
  const [p0, p1, p2] = p as [number, number, number];
  const [q0, q1, q2] = Q as [number, number, number];
  const u1 = p2 * q0 * q0 - p1 * q0 * q1 + p0 * q1 * q1 + p1 * p1 * q2;
  const u2 = p0 * p2 * q2;
  const l1 = Math.sqrt(p1 * p1 + (p0 - p2) * (p0 - p2));
  const l2 = p0 + p2;
  const l3 = p1 * p1 - 4 * p0 * p2;
  const x0 = (2 * p2 * q0 - p1 * q1) / l3 / scale + cx;
  const y0 = (2 * p0 * q1 - p1 * q0) / l3 / scale + cy;
  const a = (Math.SQRT2 * Math.sqrt((u1 - 4.0 * u2) / ((l1 - l2) * l3))) / scale;
  const b = (Math.SQRT2 * Math.sqrt(-1.0 * ((u1 - 4.0 * u2) / ((l1 + l2) * l3)))) / scale;
  const theta = p1 === 0 ? (p0 < p2 ? 0 : Math.PI / 2) : Math.PI / 2 + 0.5 * Math.atan2(p1, p0 - p2);
  const f = Math.fround;
  let width = f(2 * a), height = f(2 * b), angle: number;
  if (width > height) {
    [width, height] = [height, width];
    angle = f((90 + (theta * 180) / Math.PI) % 180);
  } else {
    angle = f(((theta * 180) / Math.PI) % 180);
  }
  return { cx: f(x0), cy: f(y0), width, height, angle };
}

// ---------------------------------------------------------------------------
// Small linear algebra

/** Least-squares solution of A x = b (Householder QR, columns well scaled). */
function leastSquares(A: number[][], b: number[]): number[] {
  const m = A.length, n = A[0]!.length;
  const R = A.map((r) => r.slice());
  const y = b.slice();
  for (let k = 0; k < n; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i]![k]! ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[k]![k]! > 0 ? -norm : norm;
    const v = new Array<number>(m).fill(0);
    v[k] = R[k]![k]! - alpha;
    for (let i = k + 1; i < m; i++) v[i] = R[i]![k]!;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i]! ** 2;
    if (vv === 0) continue;
    for (let j = k; j < n; j++) {
      let d = 0;
      for (let i = k; i < m; i++) d += v[i]! * R[i]![j]!;
      const t = (2 * d) / vv;
      for (let i = k; i < m; i++) R[i]![j] = R[i]![j]! - t * v[i]!;
    }
    let d = 0;
    for (let i = k; i < m; i++) d += v[i]! * y[i]!;
    const t = (2 * d) / vv;
    for (let i = k; i < m; i++) y[i] = y[i]! - t * v[i]!;
  }
  const x = new Array<number>(n).fill(0);
  for (let k = n - 1; k >= 0; k--) {
    let s = y[k]!;
    for (let j = k + 1; j < n; j++) s -= R[k]![j]! * x[j]!;
    x[k] = R[k]![k]! === 0 ? 0 : s / R[k]![k]!;
  }
  return x;
}

function solve2(A: number[][], b: number[]): [number, number] {
  const det = A[0]![0]! * A[1]![1]! - A[0]![1]! * A[1]![0]!;
  if (det === 0) return [0, 0];
  return [(b[0]! * A[1]![1]! - A[0]![1]! * b[1]!) / det, (A[0]![0]! * b[1]! - b[0]! * A[1]![0]!) / det];
}

/** Eigenvectors of a 3x3 matrix for its real eigenvalues. */
function eigenvectors3(M: number[][]): number[][] {
  const a = M;
  // Characteristic polynomial l^3 + c2 l^2 + c1 l + c0.
  const tr = a[0]![0]! + a[1]![1]! + a[2]![2]!;
  const minors = a[0]![0]! * a[1]![1]! - a[0]![1]! * a[1]![0]! + a[0]![0]! * a[2]![2]! - a[0]![2]! * a[2]![0]!
    + a[1]![1]! * a[2]![2]! - a[1]![2]! * a[2]![1]!;
  const det = a[0]![0]! * (a[1]![1]! * a[2]![2]! - a[1]![2]! * a[2]![1]!)
    - a[0]![1]! * (a[1]![0]! * a[2]![2]! - a[1]![2]! * a[2]![0]!)
    + a[0]![2]! * (a[1]![0]! * a[2]![1]! - a[1]![1]! * a[2]![0]!);
  const roots = cubicRoots(-tr, minors, -det);
  const out: number[][] = [];
  for (let l of roots) {
    // Polish the root with Newton's method on the polynomial.
    for (let k = 0; k < 3; k++) {
      const pv = ((l - tr) * l + minors) * l - det;
      const dv = (3 * l - 2 * tr) * l + minors;
      if (dv === 0) break;
      l -= pv / dv;
    }
    const B = a.map((r, i) => r.map((v, j) => v - (i === j ? l : 0)));
    // The eigenvector is orthogonal to B's rows: the largest cross product of two of them.
    let v: number[] = [0, 0, 0], vn = 0;
    for (const [i, j] of [[0, 1], [0, 2], [1, 2]] as const) {
      const c = cross3(B[i]!, B[j]!);
      const cn = c[0]! ** 2 + c[1]! ** 2 + c[2]! ** 2;
      if (cn > vn) {
        v = c;
        vn = cn;
      }
    }
    if (vn > 0) out.push(v.map((x) => x / Math.sqrt(vn)));
  }
  return out;
}

function cross3(a: number[], b: number[]): number[] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
}

/** Real roots of x^3 + a x^2 + b x + c. */
function cubicRoots(a: number, b: number, c: number): number[] {
  const q = (a * a - 3 * b) / 9;
  const r = (2 * a * a * a - 9 * a * b + 27 * c) / 54;
  const q3 = q * q * q;
  if (r * r < q3) {
    const t = Math.acos(Math.max(-1, Math.min(1, r / Math.sqrt(q3))));
    const sq = -2 * Math.sqrt(q);
    return [sq * Math.cos(t / 3) - a / 3, sq * Math.cos((t + 2 * Math.PI) / 3) - a / 3, sq * Math.cos((t - 2 * Math.PI) / 3) - a / 3];
  }
  const A = -Math.sign(r) * Math.cbrt(Math.abs(r) + Math.sqrt(r * r - q3));
  const B = A === 0 ? 0 : q / A;
  return [A + B - a / 3];
}

/** cvRound: to the nearest integer, halves to even. */
export function roundEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ---------------------------------------------------------------------------
// Otsu's threshold, 8-bit (thresh.cpp getThreshVal_Otsu_8u)

/** The threshold Otsu's method picks: the gray level that best splits the
 * histogram into two classes, computed in doubles as OpenCV does. */
export function otsuThreshold(data: Uint8Array): number {
  const h = new Float64Array(256);
  for (let i = 0; i < data.length; i++) h[data[i]!]!++;
  const scale = 1 / data.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * h[i]!;
  mu *= scale;
  let mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0;
  for (let i = 0; i < 256; i++) {
    const p = h[i]! * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) {
      maxSigma = sigma;
      maxVal = i;
    }
  }
  return maxVal;
}

/** Each component's centroid (connectedComponentsWithStats' centroids). */
export function centroids(c: Components, width: number): [number, number][] {
  const sx = new Float64Array(c.count), sy = new Float64Array(c.count);
  for (let i = 0; i < c.labels.length; i++) {
    const l = c.labels[i]!;
    sx[l] = sx[l]! + (i % width);
    sy[l] = sy[l]! + Math.floor(i / width);
  }
  return Array.from({ length: c.count }, (_, l) => [sx[l]! / c.area[l]!, sy[l]! / c.area[l]!] as [number, number]);
}
