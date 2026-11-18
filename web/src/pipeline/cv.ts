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
  const k = Int32Array.from(gaussianKernel8(sigma));
  const n = k.length, r = n >> 1;
  const { width: w, height: h, data } = img;
  // The taps are whole, at least 0 and sum to 256, so a row sum fits 16 bits
  // and a column sum of them 31: every sum is exact, in any order. The
  // symmetric taps are paired, and the usual 7-tap kernel is written out.
  const rows = new Uint16Array(w * h);
  const xs = reflectIndex(w, r);
  const seven = n === 7;
  const k0 = k[0]!, k1 = k[1]!, k2 = k[2]!, kc = k[r]!;
  const lo = Math.min(r, w), hi = Math.max(lo, w - r);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const edge = (x: number) => {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i]! * data[row + xs[x + i]!]!;
      rows[row + x] = s;
    };
    for (let x = 0; x < lo; x++) edge(x);
    if (seven) {
      for (let x = lo, i = row + lo; x < hi; x++, i++) {
        rows[i] = kc * data[i]! + k2 * (data[i - 1]! + data[i + 1]!) + k1 * (data[i - 2]! + data[i + 2]!)
          + k0 * (data[i - 3]! + data[i + 3]!);
      }
    } else {
      for (let x = lo, i = row + lo; x < hi; x++, i++) {
        let s = kc * data[i]!;
        for (let j = 1; j <= r; j++) s += k[r - j]! * (data[i - j]! + data[i + j]!);
        rows[i] = s;
      }
    }
    for (let x = hi; x < w; x++) edge(x);
  }
  const ys = reflectIndex(h, r);
  const out = new Uint8Array(w * h);
  const at = new Int32Array(n);
  for (let y = 0; y < h; y++) {
    for (let i = 0; i < n; i++) at[i] = ys[y + i]! * w;
    const o = y * w;
    if (seven) {
      const a0 = at[0]!, a1 = at[1]!, a2 = at[2]!, a3 = at[3]!, a4 = at[4]!, a5 = at[5]!, a6 = at[6]!;
      for (let x = 0; x < w; x++) {
        const s = kc * rows[a3 + x]! + k2 * (rows[a2 + x]! + rows[a4 + x]!) + k1 * (rows[a1 + x]! + rows[a5 + x]!)
          + k0 * (rows[a0 + x]! + rows[a6 + x]!);
        out[o + x] = (s + 32768) >>> 16;
      }
    } else {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += k[i]! * rows[at[i]! + x]!;
        out[o + x] = (s + 32768) >>> 16;
      }
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

/** The sums over the (2r+1)^2 box about every pixel of an 8-bit image (and,
 * with `squares`, of its squares), the image padded with the given border
 * rule, handed to `row(y, sums, sumsSq)` one image row at a time. Every sum
 * is an exact integer, so the order of adding does not matter: a running sum
 * along each row, then a running sum of those down each column. Only the
 * last 2r+2 rows' sums are kept, not the image's. */
function boxSumRows(data: Uint8Array, w: number, h: number, r: number, border: "replicate" | "reflect101",
  squares: boolean, row: (y: number, sums: Int32Array, sumsSq: Int32Array) => void): void {
  const xs = border === "replicate" ? clampIndex(w, r) : reflectIndex(w, r);
  const ys = border === "replicate" ? clampIndex(h, r) : reflectIndex(h, r);
  const k = 2 * r + 1;
  const slots = k + 1;                    // padded row p is in slot p % slots until p + k is added
  const ring = new Int32Array(slots * w), ringSq = new Int32Array(squares ? slots * w : 0);
  const lo = Math.min(r + 1, w), hi = Math.max(lo, w - r);
  const rowSums = (src: number, slot: number) => {
    const s0 = src * w, o = slot * w;
    let s = 0, q = 0;
    for (let i = 0; i < k; i++) {
      const v = data[s0 + xs[i]!]!;
      s += v;
      q += v * v;
    }
    ring[o] = s;
    if (squares) ringSq[o] = q;
    // Interior: x - r - 1 >= 0 and x + r <= w - 1, so no index table.
    const step = (x: number, a: number, b: number) => {
      s += a - b;
      ring[o + x] = s;
      if (squares) {
        q += a * a - b * b;
        ringSq[o + x] = q;
      }
    };
    for (let x = 1; x < lo; x++) step(x, data[s0 + xs[x + k - 1]!]!, data[s0 + xs[x - 1]!]!);
    if (squares) {
      for (let x = lo, i = s0 + lo; x < hi; x++, i++) {
        const a = data[i + r]!, b = data[i - r - 1]!;
        s += a - b;
        q += a * a - b * b;
        ring[o + x] = s;
        ringSq[o + x] = q;
      }
    } else {
      for (let x = lo, i = s0 + lo; x < hi; x++, i++) {
        s += data[i + r]! - data[i - r - 1]!;
        ring[o + x] = s;
      }
    }
    for (let x = Math.max(hi, 1); x < w; x++) step(x, data[s0 + xs[x + k - 1]!]!, data[s0 + xs[x - 1]!]!);
  };
  const acc = new Int32Array(w), accSq = new Int32Array(squares ? w : 0);
  for (let p = 0; p < k; p++) {
    rowSums(ys[p]!, p);
    const o = p * w;
    for (let x = 0; x < w; x++) acc[x] = acc[x]! + ring[o + x]!;
    if (squares) for (let x = 0; x < w; x++) accSq[x] = accSq[x]! + ringSq[o + x]!;
  }
  row(0, acc, accSq);
  for (let y = 1; y < h; y++) {
    const p = y + k - 1;                  // padded row coming in; p - k = y - 1 going out
    const add = (p % slots) * w, sub = ((y - 1) % slots) * w;
    rowSums(ys[p]!, p % slots);
    for (let x = 0; x < w; x++) acc[x] = acc[x]! + ring[add + x]! - ring[sub + x]!;
    if (squares) for (let x = 0; x < w; x++) accSq[x] = accSq[x]! + ringSq[add + x]! - ringSq[sub + x]!;
    row(y, acc, accSq);
  }
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
  const inv = 1 / (block * block);
  const idelta = Math.ceil(c);
  // The rounded mean of every possible box sum, looked up rather than computed.
  const mean = new Uint8Array(block * block * 255 + 1);
  for (let s = 0; s < mean.length; s++) mean[s] = roundEven(s * inv);
  const out = new Uint8Array(w * h);
  boxSumRows(data, w, h, block >> 1, "replicate", false, (y, sums) => {
    const o = y * w;
    for (let x = 0; x < w; x++) out[o + x] = data[o + x]! - mean[sums[x]!]! > -idelta ? 255 : 0;
  });
  return out;
}

/** detect._masks' third mask: well above the local mean (by 0.8 local
 * standard deviations, float32 as NumPy computes it) where the window has
 * contrast at all (std over 6). */
export function brightMask(img: Gray, block: number): Uint8Array {
  const { width: w, height: h, data } = img;
  const inv = 1 / (block * block);
  const f = Math.fround;
  const k = f(0.8);
  const out = new Uint8Array(w * h);
  // Two exact shortcuts before the float32 arithmetic: the threshold is never
  // below the mean (std >= 0, and float32 rounding keeps order), and std > 6
  // needs a variance above 36.
  boxSumRows(data, w, h, block >> 1, "reflect101", true, (y, s1, s2) => {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      const v = data[o + x]!;
      const mean = f(s1[x]! * inv);
      if (v <= mean) continue;
      const var_ = f(f(s2[x]! * inv) - f(mean * mean));
      if (!(var_ > 36)) continue;
      const std = f(Math.sqrt(var_));
      if (v > f(mean + f(k * std)) && std > 6) out[o + x] = 255;
    }
  });
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
 * raster order of blocks) that holds a pixel of the component.
 *
 * Labelled by runs: each row's runs of set pixels, merged (union-find) with
 * the runs of the row above they touch, diagonals included. Any labelling
 * gives the same components, and the numbering and stats depend only on
 * those, so the result is OpenCV's whatever the method. */
export function connectedComponents(mask: Uint8Array, w: number, h: number): Components {
  // Runs: start x, end x (exclusive), row, provisional label.
  let cap = 4096;
  let rs = new Int32Array(cap), re = new Int32Array(cap), ry = new Int32Array(cap), rl = new Int32Array(cap);
  let nRuns = 0;
  let parent = new Int32Array(1024);
  let next = 1;
  const find = (a: number): number => {
    while (parent[a] !== a) {
      const g = parent[parent[a]!]!;
      parent[a] = g;
      a = g;
    }
    return a;
  };
  let prevFrom = 0, prevTo = 0;            // the row above's runs
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const from = nRuns;
    let j = prevFrom;
    let x = 0;
    while (x < w) {
      while (x < w && !mask[row + x]) x++;
      if (x === w) break;
      const s = x;
      while (x < w && mask[row + x]) x++;
      const e = x;
      if (nRuns === cap) {
        cap *= 2;
        const grow = (a: Int32Array) => {
          const b = new Int32Array(cap);
          b.set(a);
          return b;
        };
        rs = grow(rs); re = grow(re); ry = grow(ry); rl = grow(rl);
      }
      // Runs above that touch [s, e): end >= s and start <= e (8-connected).
      while (j < prevTo && re[j]! < s) j++;
      let l = 0;
      for (let k = j; k < prevTo && rs[k]! <= e; k++) {
        const m = find(rl[k]!);
        if (!l) l = m;
        else if (m !== l) {
          if (m < l) {
            parent[l] = m;
            l = m;
          } else {
            parent[m] = l;
          }
        }
      }
      if (!l) {
        if (next === parent.length) {
          const grown = new Int32Array(2 * parent.length);
          grown.set(parent);
          parent = grown;
        }
        l = next++;
        parent[l] = l;
      }
      rs[nRuns] = s;
      re[nRuns] = e;
      ry[nRuns] = y;
      rl[nRuns] = l;
      nRuns++;
    }
    prevFrom = from;
    prevTo = nRuns;
  }
  // OpenCV numbers the components by their first 2x2 block, in raster order
  // of blocks and, within a block, of its four pixels: the smallest key
  // ((y >> 1) * blocks across + (x >> 1)) * 4 + (y & 1) * 2 + (x & 1) of any
  // of a component's pixels, which in a run is its first pixel's.
  const root = new Int32Array(next);
  for (let l = 1; l < next; l++) root[l] = find(l);
  const bw = (w + 1) >> 1;
  const key = new Float64Array(next).fill(Infinity);
  const bx0 = new Int32Array(next).fill(w), by0 = new Int32Array(next).fill(h);
  const bx1 = new Int32Array(next).fill(-1), by1 = new Int32Array(next).fill(-1);
  const barea = new Int32Array(next);
  for (let i = 0; i < nRuns; i++) {
    const r = root[rl[i]!]!;
    const s = rs[i]!, e = re[i]!, y = ry[i]!;
    const k = (y >> 1) * bw * 4 + (y & 1) * 2 + (s >> 1) * 4 + (s & 1);
    if (k < key[r]!) key[r] = k;
    if (s < bx0[r]!) bx0[r] = s;
    if (e - 1 > bx1[r]!) bx1[r] = e - 1;
    if (y < by0[r]!) by0[r] = y;
    if (y > by1[r]!) by1[r] = y;
    barea[r] = barea[r]! + e - s;
  }
  const roots: number[] = [];
  for (let l = 1; l < next; l++) if (root[l] === l) roots.push(l);
  roots.sort((a, b) => key[a]! - key[b]!);
  const count = roots.length + 1;
  const final = new Int32Array(next);
  const x0 = new Int32Array(count), y0 = new Int32Array(count), area = new Int32Array(count);
  const ww = new Int32Array(count), hh = new Int32Array(count);
  let fgArea = 0;
  roots.forEach((r, j) => {
    const n = j + 1;
    final[r] = n;
    x0[n] = bx0[r]!;
    y0[n] = by0[r]!;
    ww[n] = bx1[r]! - bx0[r]! + 1;
    hh[n] = by1[r]! - by0[r]! + 1;
    area[n] = barea[r]!;
    fgArea += barea[r]!;
  });
  const labels = new Int32Array(w * h);
  // The background's box, from each row's first and last unset pixel.
  let bgX0 = w, bgX1 = -1, bgY0 = h, bgY1 = -1;
  let i = 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let covered = 0, first = 0, last = w - 1;
    const from = i;
    for (; i < nRuns && ry[i] === y; i++) {
      labels.fill(final[root[rl[i]!]!]!, row + rs[i]!, row + re[i]!);
      covered += re[i]! - rs[i]!;
    }
    if (covered === w) continue;
    if (i > from) {
      if (rs[from] === 0) first = re[from]!;
      if (re[i - 1] === w) last = rs[i - 1]! - 1;
    }
    if (first < bgX0) bgX0 = first;
    if (last > bgX1) bgX1 = last;
    if (y < bgY0) bgY0 = y;
    bgY1 = y;
  }
  // The background's own stats, as OpenCV gives them for label 0.
  x0[0] = bgX0;
  y0[0] = bgY0;
  ww[0] = bgX1 - bgX0 + 1;
  hh[0] = bgY1 - bgY0 + 1;
  area[0] = w * h - fgArea;
  return { count, labels, x: x0, y: y0, w: ww, h: hh, area };
}

// ---------------------------------------------------------------------------
// findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE) for one component

const DX = [1, 1, 0, -1, -1, -1, 0, 1];
const DY = [0, -1, -1, -1, 0, 1, 1, 1];

/** The outer border of component `label` inside its box, as OpenCV's
 * findContours traces it (Suzuki's border following, every pixel kept): from
 * the component's first pixel in raster order, points [x0, y0, x1, y1, ...].
 * Traced on the label image itself: a pixel counts if it lies in the box and
 * carries the label, as if the component were alone in its box. */
export function outerContour(c: Components, width: number, height: number, label: number): Int32Array {
  const bx = c.x[label]!, by = c.y[label]!, bw = c.w[label]!, bh = c.h[label]!;
  const labels = c.labels;
  void height;
  const set = (x: number, y: number) =>
    x >= bx && x < bx + bw && y >= by && y < by + bh && labels[y * width + x] === label;
  // The first pixel in raster order is on the box's top row.
  let sx = -1;
  for (let x = bx; x < bx + bw; x++) {
    if (labels[by * width + x] === label) {
      sx = x;
      break;
    }
  }
  if (sx < 0) return new Int32Array(0);
  const x0 = sx, y0 = by;
  // First neighbour, looking from direction 3 down to 4 (s_end = 4 for an outer border).
  let s = 4;
  let x1 = 0, y1 = 0;
  do {
    s = (s - 1) & 7;
    x1 = x0 + DX[s]!;
    y1 = y0 + DY[s]!;
  } while (!set(x1, y1) && s !== 4);
  if (s === 4) return Int32Array.of(x0, y0);       // a single pixel
  let out = new Int32Array(256);
  let n = 0;
  let px = x0, py = y0;                            // the current pixel (i3)
  for (;;) {
    s = Math.min(s, 15);
    let qx = 0, qy = 0;
    while (s < 15) {
      s++;
      qx = px + DX[s & 7]!;
      qy = py + DY[s & 7]!;
      if (set(qx, qy)) break;
    }
    s &= 7;
    if (n + 2 > out.length) {
      const grown = new Int32Array(2 * out.length);
      grown.set(out);
      out = grown;
    }
    out[n++] = px;
    out[n++] = py;
    const done = qx === x0 && qy === y0 && px === x1 && py === y1;
    px = qx;
    py = qy;
    if (done) break;
    s = (s + 4) & 7;
  }
  return out.slice(0, n);
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
  // Columns of A one after another (column-major), as leastSquares takes them.
  const A = new Float64Array(5 * n), b = new Float64Array(n).fill(10000);
  const P = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const px = f(pts[2 * i]! - cx) * scale, py = f(pts[2 * i + 1]! - cy) * scale;
    P[2 * i] = px;
    P[2 * i + 1] = py;
    A[i] = -px * px;
    A[n + i] = -py * py;
    A[2 * n + i] = -px * py;
    A[3 * n + i] = px;
    A[4 * n + i] = py;
  }
  const gfp = leastSquares(A, n, 5, b);
  // The centre, where the conic's gradient vanishes.
  const rp = solve2([[2 * gfp[0]!, gfp[2]!], [gfp[2]!, 2 * gfp[1]!]], [gfp[3]!, gfp[4]!]);
  const A3 = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const px = P[2 * i]!, py = P[2 * i + 1]!;
    A3[i] = (px - rp[0]) * (px - rp[0]);
    A3[n + i] = (py - rp[1]) * (py - rp[1]);
    A3[2 * n + i] = (px - rp[0]) * (py - rp[1]);
  }
  const g = leastSquares(A3, n, 3, new Float64Array(n).fill(1));
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

/** Least-squares solution of A x = b (Householder QR, columns well scaled),
 * A an m x n matrix stored column by column. A and b are overwritten. */
function leastSquares(R: Float64Array, m: number, n: number, y: Float64Array): number[] {
  const v = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    const ck = k * m;
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[ck + i]! ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[ck + k]! > 0 ? -norm : norm;
    v[k] = R[ck + k]! - alpha;
    for (let i = k + 1; i < m; i++) v[i] = R[ck + i]!;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i]! ** 2;
    if (vv === 0) continue;
    for (let j = k; j < n; j++) {
      const cj = j * m;
      let d = 0;
      for (let i = k; i < m; i++) d += v[i]! * R[cj + i]!;
      const t = (2 * d) / vv;
      for (let i = k; i < m; i++) R[cj + i] = R[cj + i]! - t * v[i]!;
    }
    let d = 0;
    for (let i = k; i < m; i++) d += v[i]! * y[i]!;
    const t = (2 * d) / vv;
    for (let i = k; i < m; i++) y[i] = y[i]! - t * v[i]!;
  }
  const x = new Array<number>(n).fill(0);
  for (let k = n - 1; k >= 0; k--) {
    let s = y[k]!;
    for (let j = k + 1; j < n; j++) s -= R[j * m + k]! * x[j]!;
    x[k] = R[k * m + k]! === 0 ? 0 : s / R[k * m + k]!;
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
