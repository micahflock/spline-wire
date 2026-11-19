// The live view's regions (web/src/overlay/region.ts) on the 12 MP renders of make_frames.py:
//
//   1. Reach by scale: the chain's box (plus margin) drawn at a range of
//      scales: pins found against the rim's size in work pixels. Sets
//      REGIONS.target.
//   2. The tracker frame by frame, from a cold start, with the real analysis
//      (pipeline/live.ts) answering every third frame: which region each frame
//      reads, its work pixels and detection time, and pins found.
//
// A region is drawn from the full-resolution frame by area averaging (what
// cv2.INTER_AREA does and the camera view's drawImage approximates with
// imageSmoothingQuality "high").
//
//   cd web && npx esbuild ../experiments/live_view/regions.ts --bundle --platform=node --format=esm --outfile=../out/live_view/regions.mjs
//   node ../out/live_view/regions.mjs ../out/live_view
import { readFileSync } from "node:fs";
import { detectBlobs, shapeOf, type Window } from "../../web/src/overlay/blobs.ts";
import { REGIONS, RegionTracker, region, type Region } from "../../web/src/overlay/region.ts";
import { analyse } from "../../web/src/pipeline/live.ts";
import { specFromDict } from "../../web/src/pipeline/chain.ts";
import type { GrayFrame } from "../../web/src/overlay/types.ts";

const D = process.argv[2]!;
const meta = JSON.parse(readFileSync(D + "/meta.json", "utf8"));
const spec = specFromDict(meta.chain);
const shape = shapeOf(meta.chain);
const [VW, VH] = meta.size as [number, number];
const spans: number[] = [0.55, 0.4, 0.3, 0.22, 0.16];
const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]!; };

/** Area-average the source rectangle (x, y, w, h) of `src` into dw × dh. */
function draw(src: GrayFrame, r: { x: number; y: number; width: number; height: number; workWidth: number; workHeight: number }): GrayFrame {
  const { workWidth: dw, workHeight: dh } = r;
  const weights = (start: number, len: number, n: number, limit: number) => {
    const s = len / n, out: { i0: number; w: number[] }[] = [];
    for (let k = 0; k < n; k++) {
      const a = start + k * s, b = a + s, i0 = Math.max(0, Math.floor(a)), i1 = Math.min(limit, Math.ceil(b));
      const w: number[] = [];
      for (let i = i0; i < i1; i++) w.push((Math.min(b, i + 1) - Math.max(a, i)) / s);
      out.push({ i0, w });
    }
    return out;
  };
  const wx = weights(r.x, r.width, dw, src.width), wy = weights(r.y, r.height, dh, src.height);
  const rows = new Float32Array(src.height * dw);
  const y0 = wy[0]!.i0, y1 = wy[dh - 1]!.i0 + wy[dh - 1]!.w.length;
  for (let y = y0; y < y1; y++) {
    for (let k = 0; k < dw; k++) {
      const { i0, w } = wx[k]!;
      let s = 0;
      for (let i = 0; i < w.length; i++) s += w[i]! * src.data[y * src.width + i0 + i]!;
      rows[y * dw + k] = s;
    }
  }
  const data = new Uint8Array(dw * dh);
  for (let k = 0; k < dh; k++) {
    const { i0, w } = wy[k]!;
    for (let x = 0; x < dw; x++) {
      let s = 0;
      for (let i = 0; i < w.length; i++) s += w[i]! * rows[(i0 + i) * dw + x]!;
      data[k * dw + x] = Math.round(s);
    }
  }
  return { width: dw, height: dh, data };
}

function timeDetect(work: GrayFrame, win: Window, n = 7) {
  const t: number[] = [];
  let blobs = detectBlobs(work, win, shape);
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    blobs = detectBlobs(work, win, shape);
    t.push(performance.now() - t0);
  }
  return { ms: med(t), blobs };
}

const full: Record<number, GrayFrame> = {};
for (const s of spans) full[s] = { width: VW, height: VH, data: new Uint8Array(readFileSync(`${D}/full-${s}.gray`)) };
const truth = (s: number) => meta.pins[String(s)] as [number, number][];
const found = (s: number, blobs: { x: number; y: number; d: number }[]) =>
  truth(s).filter(([x, y]) => blobs.some((b) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;
const long = Math.max(VW, VH), ref = long / 800;

console.log("1. Reach by scale: the chain's box plus 2 links drawn at each scale (long side of the whole frame at that scale)");
console.log("span  long side  rim (work px)  pins found  region (work px)");
for (const s of spans) {
  const pins = truth(s);
  const xs = pins.map((p) => p[0]), ys = pins.map((p) => p[1]);
  const link = med(pins.slice(1).map((p, k) => Math.hypot(p[0] - pins[k]![0], p[1] - pins[k]![1])));
  const m = 2 * link;
  for (const L of [800, 1008, 1200, 1400, 1600, 1800, 2016, 2400, 3024]) {
    const scale = long / L;
    const r = region(Math.min(...xs) - m, Math.min(...ys) - m, Math.max(...xs) - Math.min(...xs) + 2 * m,
      Math.max(...ys) - Math.min(...ys) + 2 * m, scale, ref, VW, VH, "track");
    const { blobs } = timeDetect(draw(full[s]!, r), r, 1);
    const ds = blobs.map((b) => b.d / scale);
    console.log(`${s.toFixed(2)}  ${String(L).padStart(9)}  ${(ds.length ? med(ds) : NaN).toFixed(1).padStart(13)}  ${String(found(s, blobs)).padStart(7)}/13  ${r.workWidth}x${r.workHeight}`);
  }
}

const tag = (r: Region) => ({ full: "F", tile: "Q", track: "T" })[r.kind];
console.log("\n2. The tracker from a cold start (analysis answers every 3rd frame, with no delay). F full frame, Q a quarter, T tracked;");
console.log("   'moving': the picture slides by a fifth of a link a frame (3 pitches/s at 15 frames/s, twice the coach's limit)");
for (const [s, moving] of spans.flatMap((s) => [[s, false], [s, true]] as [number, boolean][])) {
  const pins = truth(s);
  const v = moving ? 0.2 * med(pins.slice(1).map((p, k) => Math.hypot(p[0] - pins[k]![0], p[1] - pins[k]![1]))) : 0;
  const tracker = new RegionTracker(shape, spec.n_pins);
  const rows: string[] = [];
  let first = -1;
  const times: number[] = [], px: number[] = [];
  for (let f = 0; f < 16; f++) {
    const r: Region = tracker.next(VW, VH);
    // Frame f shows the picture moved left by f·v: drawn from f·v further right, its pins f·v further left
    // (past the render's right edge the picture fades: a full frame or a quarter there loses a sliver).
    const dx = moving ? v * f : 0;
    const work = draw(full[s]!, { ...r, x: r.x + dx });
    const { ms, blobs } = timeDetect(work, r);
    tracker.observe(blobs);
    const n = truth(s).filter(([x, y]) => blobs.some((b) => Math.hypot(b.x - (x - dx), b.y - y) < 0.3 * b.d)).length;
    if (n === 13 && first < 0) first = f;
    if (f >= 8) { times.push(ms); px.push(r.workWidth * r.workHeight); }
    if (f % 3 === 0) {
      const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [VW, VH], spec, meta.focal_px);
      tracker.chainFound(a.order.map((i) => [blobs[i]!.x, blobs[i]!.y]), a.order.map((i) => blobs[i]!.d), a.gaps.length);
    }
    rows.push(`${tag(r)}${r.workWidth}x${r.workHeight}:${n}`);
  }
  console.log(`span ${s}${moving ? " moving" : ""}: all 13 from frame ${first}; frames 8-15 ${med(px) / 1e6 > 0 ? (med(px) / 1e6).toFixed(2) : "-"} MP, ${med(times).toFixed(1)} ms`);
  console.log("   " + rows.join(" "));
}

console.log("\n3. Scanning: each region's cost (no chain in view: a 16% chain's frame)");
{
  const tracker = new RegionTracker(shape, spec.n_pins);
  for (let f = 0; f < 5; f++) {
    const r = tracker.next(VW, VH);
    const { ms } = timeDetect(draw(full[0.16]!, r), r);
    console.log(`${r.kind} ${r.workWidth}x${r.workHeight} (${(r.workWidth * r.workHeight / 1e6).toFixed(2)} MP): ${ms.toFixed(1)} ms`);
  }
}
void REGIONS;

console.log("\n4. Sharpness (median over the pins) read by today's full frame at 800 px and by the tracked region,");
console.log("   on the render blurred (about Gaussian, sigma in video px). The coach flags a median below 0.65 and keeps a hold to 0.55.");
{
  /** Gaussian-like blur: three passes of a box of width b each way. */
  function blur(src: GrayFrame, sigma: number): GrayFrame {
    if (sigma === 0) return src;
    const b = Math.max(1, Math.round(Math.sqrt((12 * sigma * sigma) / 3 + 1))), r = b >> 1;
    let a = Float32Array.from(src.data);
    const { width: w, height: h } = src;
    for (let pass = 0; pass < 3; pass++) {
      for (const [n, m, stride, step] of [[h, w, w, 1], [w, h, 1, w]] as const) {
        const out = new Float32Array(a.length);
        for (let i = 0; i < n; i++) {
          const base = i * stride;
          let s = 0, c = 0;
          for (let k = 0; k <= r && k < m; k++) { s += a[base + k * step]!; c++; }
          for (let k = 0; k < m; k++) {
            out[base + k * step] = s / c;
            if (k + r + 1 < m) { s += a[base + (k + r + 1) * step]!; c++; }
            if (k - r >= 0) { s -= a[base + (k - r) * step]!; c--; }
          }
        }
        a = out;
      }
    }
    return { width: w, height: h, data: Uint8Array.from(a, (v) => Math.round(v)) };
  }
  console.log("span  sigma  full 800: found, sharpness   tracked: scale, found, sharpness");
  for (const s of [0.55, 0.4]) {
    for (const sigma of [0, 2, 4, 7, 10]) {
      const img = blur(full[s]!, sigma);
      const f = region(0, 0, VW, VH, long / 800, ref, VW, VH, "full");
      const a = detectBlobs(draw(img, f), f, shape);
      const tracker = new RegionTracker(shape, spec.n_pins);
      tracker.next(VW, VH);
      tracker.chainFound(truth(s), truth(s).map(() => med(a.map((b) => b.d)) || 60), 0);
      const r = tracker.next(VW, VH);
      const b = detectBlobs(draw(img, r), r, shape);
      const sh = (x: { sharpness: number }[]) => (x.length ? med(x.map((q) => q.sharpness)).toFixed(2) : "-");
      console.log(`${s.toFixed(2)}  ${String(sigma).padStart(5)}  ${String(found(s, a)).padStart(8)}, ${sh(a).padStart(4)}            ${r.scaleX.toFixed(2)}, ${found(s, b)}, ${sh(b)}`);
    }
  }
}
