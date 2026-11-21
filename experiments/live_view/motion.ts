// The tracker (web/src/overlay/region.ts) against a moving picture: how fast the
// chain can move between processed frames and still be followed. The snap-fit
// chain (snap-frames.ts, span 0.3, about 70 video px a link) slides back and
// forth across the frame at a set speed in links per processed frame, turning
// at ±1000 px; each frame the tracker's region is drawn and searched, and the
// analysis (pipeline/live.ts) answers every frame (a known focal length: the
// camera view then analyses every frame) or every 15th (none: once a second).
// Also a jump: the picture still, then moved by N links in one frame (a tilt
// of the phone), then still again: frames until all 13 pins are back.
//
// Counts frames 3 to 40 (after the cold start) with all 13 pins found, and
// those not drawn from the tracked region.
//
//   cd web && npx esbuild ../experiments/live_view/motion.ts --bundle --platform=node --format=esm --outfile=../out/live_view/motion.mjs
//   node ../out/live_view/motion.mjs
import { detectBlobs, shapeOf } from "../../web/src/overlay/blobs.ts";
import { RegionTracker, type Region } from "../../web/src/overlay/region.ts";
import { SNAP_SPEC, snapScene, VIDEO } from "../../web/src/overlay/snap-frames.ts";
import { analyse } from "../../web/src/pipeline/live.ts";
import { specFromDict } from "../../web/src/pipeline/chain.ts";
import type { Blob, Point } from "../../web/src/overlay/types.ts";

const RING = shapeOf(SNAP_SPEC), SPEC = specFromDict(SNAP_SPEC);
const { width: VW, height: VH } = VIDEO;
const sc = snapScene(0.3, 128);
const link = Math.hypot(sc.pins[1]![0] - sc.pins[0]![0], sc.pins[1]![1] - sc.pins[0]![1]);

/** One run: `offset(f)` is the picture's shift at frame f (video px). */
function track(offset: (f: number) => Point, frames: number, every: number) {
  const t = new RegionTracker(RING, 13);
  const rows: { all: boolean; kind: Region["kind"] }[] = [];
  for (let f = 0; f < frames; f++) {
    const [dx, dy] = offset(f);
    const r = t.next(VW, VH);
    const at = { ...r, x: r.x - dx, y: r.y - dy };
    const blobs: Blob[] = detectBlobs(sc.draw(at), at, RING).map((b) => ({ ...b, x: b.x + dx, y: b.y + dy }));
    t.observe(blobs);
    if (f % every === 0) {
      const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [VW, VH], SPEC, 3000);
      t.chainFound(a.order.map((i) => [blobs[i]!.x, blobs[i]!.y] as Point), a.order.map((i) => blobs[i]!.d), a.gaps.length);
    }
    const all = sc.pins.every(([x, y]) => blobs.some((b) => Math.hypot(b.x - x - dx, b.y - y - dy) < 0.3 * b.d));
    rows.push({ all, kind: r.kind });
  }
  return rows;
}

/** Back and forth along (ux, uy) at `speed` links a frame, turning at ±1000 px, starting still for 3 frames. */
const bounce = (speed: number, ux: number, uy: number) => (f: number): Point => {
  const s = Math.max(0, f - 3) * speed * link, A = 1000, p = s % (4 * A);
  const pos = p < A ? p : p < 3 * A ? 2 * A - p : p - 4 * A;
  return [pos * ux, pos * uy];
};

console.log(`link ${link.toFixed(0)} video px`);
console.log("speed (links/frame)  analysis  across: 13 pins / frames, scans   diagonal: 13 pins / frames, scans");
for (const speed of [0.25, 0.5, 0.75, 1, 1.5, 2, 3]) {
  for (const every of [1, 15]) {
    const cols = [[1, 0], [Math.SQRT1_2, Math.SQRT1_2]].map(([ux, uy]) => {
      const rows = track(bounce(speed, ux!, uy!), 41, every).slice(3);
      return `${String(rows.filter((r) => r.all).length).padStart(2)}/${rows.length}, ${String(rows.filter((r) => r.kind !== "track").length).padStart(2)}`;
    });
    console.log(`${String(speed).padStart(19)}  ${(every === 1 ? "every" : "1 in 15").padStart(8)}  ${cols[0]!.padStart(28)}   ${cols[1]!.padStart(28)}`);
  }
}

console.log("\njump (links, in one frame)  analysis  frames until 13 pins again (of 20), scans");
for (const jump of [1, 2, 3, 5, 8]) {
  for (const every of [1, 15]) {
    const rows = track((f) => [f >= 6 ? jump * link : 0, f >= 6 ? 0.4 * jump * link : 0], 26, every).slice(6);
    const back = rows.findIndex((r) => r.all);
    console.log(`${String(jump).padStart(26)}  ${(every === 1 ? "every" : "1 in 15").padStart(8)}  ${String(back < 0 ? "never" : back).padStart(5)}, ${rows.filter((r) => r.kind !== "track").length}`);
  }
}
