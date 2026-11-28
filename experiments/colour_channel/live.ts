// Orange rims on off-white links, read through each colour channel: the live
// view (web/src/overlay/blobs.ts, region.ts, pipeline/live.ts) on
// render_still.py's 1440×1920 stream frames at span 0.30.
//
// For each table, channel (luma as the live view's toGray computes it, or one
// of R, G, B) and Colours (light, dark):
//   cold: the tracker from a cold start, 16 frames, the analysis every third
//         frame (as experiments/live_view/regions.ts): the frame where all 13
//         rims were first found, and the pins the last analysis ordered;
//   tracked: the tracker seeded with the true pins (the chain already
//         followed): rims found in the tracked region and pins ordered.
//
//   cd web && npx esbuild ../experiments/colour_channel/live.ts --bundle --platform=node --format=esm --outfile=../out/colour_channel/live.mjs
//   node ../out/colour_channel/live.mjs ../out/colour_channel/live
import { readFileSync } from "node:fs";
import { detectBlobs, shapeOf, toGray } from "../../web/src/overlay/blobs.ts";
import { RegionTracker, type Region } from "../../web/src/overlay/region.ts";
import { analyse } from "../../web/src/pipeline/live.ts";
import { specFromDict } from "../../web/src/pipeline/chain.ts";
import type { GrayFrame } from "../../web/src/overlay/types.ts";

const D = process.argv[2]!;
const meta = JSON.parse(readFileSync(D + "/meta.json", "utf8"));
const CHANNELS = ["luma", "r", "g", "b"] as const;

/** Area-average the source rectangle of `src` into the region's work size (as regions.ts). */
function draw(src: GrayFrame, r: Region): GrayFrame {
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
  for (let y = wy[0]!.i0; y < wy[dh - 1]!.i0 + wy[dh - 1]!.w.length; y++) {
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

console.log("table             channel Colours  cold: all 13 at frame, ordered  | tracked: rims found, ordered");
for (const f of meta.frames) {
  const [W, H] = f.size as [number, number];
  const rgb = new Uint8Array(readFileSync(`${D}/${f.file}`));
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    rgba[j] = rgb[3 * i]!; rgba[j + 1] = rgb[3 * i + 1]!; rgba[j + 2] = rgb[3 * i + 2]!; rgba[j + 3] = 255;
  }
  const truth = f.pins as [number, number][];
  for (const ch of CHANNELS) {
    const full = toGray(rgba, W, H, undefined, ch);
    for (const shade of ["light", "dark"] as const) {
      const chain = { ...meta.chain, fiducial_shade: shade };
      const spec = specFromDict(chain), shape = shapeOf(chain);
      const found = (blobs: { x: number; y: number; d: number }[]) =>
        truth.filter(([x, y]) => blobs.some((b) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;
      const ordered = (blobs: { x: number; y: number; d: number }[], order: number[]) =>
        order.filter((i) => truth.some(([x, y]) => Math.hypot(blobs[i]!.x - x, blobs[i]!.y - y) < 0.3 * blobs[i]!.d)).length;
      // Cold start.
      const tracker = new RegionTracker(shape, spec.n_pins);
      let first = -1, last = 0, best = 0;
      for (let k = 0; k < 16; k++) {
        const r = tracker.next(W, H);
        const blobs = detectBlobs(draw(full, r), r, shape);
        tracker.observe(blobs);
        const n = found(blobs);
        best = Math.max(best, n);
        if (n === truth.length && first < 0) first = k;
        if (k % 3 === 0) {
          const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [W, H], spec, f.focal_px);
          tracker.chainFound(a.order.map((i) => [blobs[i]!.x, blobs[i]!.y]), a.order.map((i) => blobs[i]!.d), a.gaps.length);
          last = ordered(blobs, a.order);
        }
      }
      // Tracked: seeded with the truth.
      const t2 = new RegionTracker(shape, spec.n_pins);
      t2.next(W, H);
      const rim = (meta.chain.fiducial_mm / meta.chain.pitch_mm) * Math.hypot(truth[1]![0] - truth[0]![0], truth[1]![1] - truth[0]![1]);
      t2.chainFound(truth, truth.map(() => rim), 0);
      const r = t2.next(W, H);
      const blobs = detectBlobs(draw(full, r), r, shape);
      const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [W, H], spec, f.focal_px);
      console.log(`${f.table.padEnd(17)} ${ch.padEnd(7)} ${shade.padEnd(7)}  ${(first >= 0 ? String(first) : "never").padStart(5)} (best ${String(best).padStart(2)}), ${String(last).padStart(2)}/13`
        + `            | ${String(found(blobs)).padStart(2)}/13, ${String(ordered(blobs, a.order)).padStart(2)}/13 (${r.kind} ${r.workWidth}x${r.workHeight})`);
    }
  }
}
