// The live detector (web/src/overlay/blobs.ts) on make_frames.py's frames: per
// work size and span, the threshold (today's one pass, against the two
// localThreshold passes it replaced), the whole detector, pins found, and the
// analysis (pipeline/live.ts); then toGray and the still's detector. The
// regions the camera view now draws are timed by regions.ts.
import { readFileSync } from "node:fs";
const { detectBlobs, localThreshold, thresholds, shapeOf, toGray } = await import("../../web/src/overlay/blobs.ts");
const { analyse } = await import("../../web/src/pipeline/live.ts");
const { specFromDict } = await import("../../web/src/pipeline/chain.ts");
const { detectFiducials } = await import("../../web/src/pipeline/detect.ts");
const D = process.argv[2];
const meta = JSON.parse(readFileSync(D + "/meta.json", "utf8"));
const spec = specFromDict(meta.chain);
const shape = shapeOf(meta.chain);
const med = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
function time(fn: () => unknown, n: number) { const t: number[] = []; let r; for (let i = 0; i < n; i++) { const t0 = performance.now(); r = fn(); t.push(performance.now() - t0); } return { ms: med(t), r }; }
const [VW, VH] = meta.size;
console.log("span L    wxh        2 thresholds  1 pass  detectBlobs  found/13       analyse(focal) analyse(est) tilt");
for (const f of meta.frames) {
  const data = new Uint8Array(readFileSync(D + "/" + f.file));
  const frame = { width: f.w, height: f.h, data };
  const scale = VW / f.w;
  const N = f.L > 1300 ? 9 : 21;
  const two = time(() => { localThreshold(frame, true); localThreshold(frame); }, N).ms;
  const one = time(() => thresholds(frame, true), N).ms;
  const { ms: tot, r: blobs } = time(() => detectBlobs(frame, scale, shape), N) as any;
  const pins: number[][] = meta.pins[String(f.span)];
  const found = pins.filter(([x, y]) => blobs.some((b: any) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;
  const pts = blobs.map((b: any) => [b.x, b.y]), ds = blobs.map((b: any) => b.d);
  const af = time(() => analyse(pts, ds, [VW, VH], spec, meta.focal_px), 5);
  const ae = time(() => analyse(pts, ds, [VW, VH], spec, null), 3);
  console.log(`${f.span} ${String(f.L).padEnd(4)} ${(f.w + "x" + f.h).padEnd(9)} ${two.toFixed(1).padStart(12)} ${one.toFixed(1).padStart(7)} ${tot.toFixed(1).padStart(12)}  ${found}/${pins.length} (${blobs.length} blobs)  ${af.ms.toFixed(1).padStart(13)} ${ae.ms.toFixed(1).padStart(12)}  ${(af.r as any).tilt_deg} order=${(af.r as any).order.length}`);
}
// rgba->gray at work sizes, into a new buffer and a reused one
for (const [w, h] of [[600, 800], [900, 1200], [1512, 2016]]) {
  const rgba = new Uint8ClampedArray(w * h * 4), into = new Uint8Array(w * h);
  console.log(`toGray ${w}x${h}: ${time(() => toGray(rgba, w, h), 9).ms.toFixed(1)} ms, into a reused buffer ${time(() => toGray(rgba, w, h, into), 9).ms.toFixed(1)} ms`);
}
// Full-res still detection
for (const span of [0.55, 0.22]) {
  const data = new Uint8Array(readFileSync(`${D}/full-${span}.gray`));
  const { ms, r } = time(() => detectFiducials({ width: VW, height: VH, data }, spec), 3) as any;
  console.log(`still detectFiducials 3024x4032 span ${span}: ${ms.toFixed(0)} ms, ${r.length} fiducials`);
}
