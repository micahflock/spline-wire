import { readFileSync } from "node:fs";
import { localThreshold, BLOBS } from "../../web/src/overlay/blobs.ts";
const D = process.argv[2];
const meta = JSON.parse(readFileSync(D + "/meta.json", "utf8"));
const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
let sat = new Uint32Array(0), mLo = new Uint8Array(0), mHi = new Uint8Array(0);
function fused(f: { width: number; height: number; data: Uint8Array }) {
  const { width: w, height: h, data } = f; const n = (w + 1) * (h + 1);
  if (sat.length < n) sat = new Uint32Array(n); if (mLo.length < w * h) { mLo = new Uint8Array(w * h); mHi = new Uint8Array(w * h); }
  for (let y = 0; y < h; y++) { let row = 0; const o = (y + 1) * (w + 1), p = y * (w + 1); for (let x = 0; x < w; x++) { row += data[y * w + x]; sat[o + x + 1] = sat[p + x + 1] + row; } }
  const r = 12, off = BLOBS.offset;
  for (let y = 0; y < h; y++) { const ya = Math.max(0, y - r), yb = Math.min(h, y + r + 1); const A = yb * (w + 1), B = ya * (w + 1);
    for (let x = 0; x < w; x++) { const xa = Math.max(0, x - r), xb = Math.min(w, x + r + 1);
      const sum = sat[A + xb] - sat[B + xb] - sat[A + xa] + sat[B + xa]; const area = (xb - xa) * (yb - ya), v = data[y * w + x];
      mHi[y * w + x] = (v - off) * area > sum ? 1 : 0; mLo[y * w + x] = (v + off) * area < sum ? 1 : 0; } }
}
for (const L of [800, 1200, 1600, 2016]) {
  const f = meta.frames.find((x: any) => x.span === 0.3 && x.L === L);
  const frame = { width: f.w, height: f.h, data: new Uint8Array(readFileSync(D + "/" + f.file)) };
  const a: number[] = [], b: number[] = [];
  for (let i = 0; i < 15; i++) { let t0 = performance.now(); localThreshold(frame, true); localThreshold(frame); a.push(performance.now() - t0);
    t0 = performance.now(); fused(frame); b.push(performance.now() - t0); }
  // check identical
  const ok = localThreshold(frame, true).every((v, i) => v === mLo[i]) && localThreshold(frame).every((v, i) => v === mHi[i]);
  console.log(`L${L}: two localThreshold ${med(a).toFixed(1)} ms, fused+reused ${med(b).toFixed(1)} ms, identical=${ok}`);
}
