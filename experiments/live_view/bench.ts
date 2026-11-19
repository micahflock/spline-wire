import { readFileSync } from "node:fs";
const { detectBlobs, localThreshold, ringAt, shapeOf, toGray, BLOBS } = await import("../../web/src/overlay/blobs.ts");
const { analyse } = await import("../../web/src/pipeline/live.ts");
const { specFromDict } = await import("../../web/src/pipeline/chain.ts");
const { detectFiducials } = await import("../../web/src/pipeline/detect.ts");
const D = process.argv[2];
const meta = JSON.parse(readFileSync(D + "/meta.json", "utf8"));
const spec = specFromDict(meta.chain);
const shape = shapeOf(meta.chain);
const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
function time(fn: () => unknown, n: number) { const t: number[] = []; let r; for (let i = 0; i < n; i++) { const t0 = performance.now(); r = fn(); t.push(performance.now() - t0); } return { ms: med(t), r }; }
// components copy (not exported)
function components(mask: Uint8Array, w: number, h: number) { const out: any[] = []; const stack = new Int32Array(w * h); const maxSide = Math.max(w, h) / 4;
  for (let start = 0; start < mask.length && out.length < BLOBS.maxCandidates; start++) { if (mask[start] !== 1) continue; let top = 0, x0 = w, y0 = h, x1 = -1, y1 = -1; stack[top++] = start; mask[start] = 2;
    while (top > 0) { const p = stack[--top]; const px = p % w, py = (p - px) / w; if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
      for (let dy = -1; dy <= 1; dy++) { const ny = py + dy; if (ny < 0 || ny >= h) continue; for (let dx = -1; dx <= 1; dx++) { const nx = px + dx; if (nx < 0 || nx >= w) continue; const q = ny * w + nx; if (mask[q] === 1) { mask[q] = 2; stack[top++] = q; } } } }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1; const lo = Math.min(bw, bh), hi = Math.max(bw, bh); if (lo < BLOBS.minSide || hi > maxSide || hi > BLOBS.maxAspect * lo) continue; out.push({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, rx: bw / 2, ry: bh / 2 }); }
  return out; }
const [VW, VH] = meta.size;
console.log("span L  wxh       thr(x2) comps(x2) ringAt  total  found/13  analyse(f) analyse(est) tilt");
for (const f of meta.frames) {
  const data = new Uint8Array(readFileSync(D + "/" + f.file));
  const frame = { width: f.w, height: f.h, data };
  const scale = VW / f.w;
  const N = f.L > 1300 ? 5 : 15;
  const thr = time(() => { localThreshold(frame, true); localThreshold(frame); }, N).ms;
  const m1 = localThreshold(frame, true), m2 = localThreshold(frame);
  const comp = time(() => { components(m1.slice(), f.w, f.h); components(m2.slice(), f.w, f.h); }, N).ms;
  const boxes = [...components(m1.slice(), f.w, f.h).map((c: any) => ({ ...c, rx: c.rx / shape.inner, ry: c.ry / shape.inner })), ...components(m2.slice(), f.w, f.h)];
  const ra = time(() => { for (const c of boxes) ringAt(frame, c.cx, c.cy, c.rx, c.ry, shape.inner); }, N).ms;
  const { ms: tot, r: blobs } = time(() => detectBlobs(frame, scale, shape), N) as any;
  const pins: number[][] = meta.pins[String(f.span)];
  const found = pins.filter(([x, y]) => blobs.some((b: any) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;
  const pts = blobs.map((b: any) => [b.x, b.y]), ds = blobs.map((b: any) => b.d);
  const af = time(() => analyse(pts, ds, [VW, VH], spec, meta.focal_px), 5);
  const ae = time(() => analyse(pts, ds, [VW, VH], spec, null), 3);
  console.log(`${f.span} ${String(f.L).padEnd(4)} ${(f.w + "x" + f.h).padEnd(9)} ${thr.toFixed(1).padStart(6)} ${comp.toFixed(1).padStart(8)} ${ra.toFixed(1).padStart(7)} ${tot.toFixed(1).padStart(6)}  ${found}/${pins.length} (${blobs.length} blobs, ${boxes.length} boxes)  ${af.ms.toFixed(1)}  ${ae.ms.toFixed(1)}  ${(af.r as any).tilt_deg} order=${(af.r as any).order.length}`);
}
// rgba->gray at work sizes
for (const [w, h] of [[600, 800], [1512, 2016], [3024, 4032]]) { const rgba = new Uint8ClampedArray(w * h * 4); console.log(`toGray ${w}x${h}: ${time(() => toGray(rgba, w, h), 5).ms.toFixed(1)} ms`); }
// Full-res still detection
for (const span of [0.55, 0.22]) {
  const data = new Uint8Array(readFileSync(`${D}/full-${span}.gray`));
  const { ms, r } = time(() => detectFiducials({ width: VW, height: VH, data }, spec), 3) as any;
  console.log(`still detectFiducials 3024x4032 span ${span}: ${ms.toFixed(0)} ms, ${r.length} fiducials`);
}
