import { describe, expect, it } from "vitest";
import { specFromDict } from "../pipeline/chain";
import { analyse } from "../pipeline/live";
import { detectBlobs, shapeOf } from "./blobs";
import { REGIONS, RegionTracker, fullRegion, region, type Region } from "./region";
import { SNAP_SPEC, snapScene, VIDEO } from "./snap-frames";
import type { Blob, Point } from "./types";

const RING = shapeOf(SNAP_SPEC);
const SPEC = specFromDict(SNAP_SPEC);
const { width: VW, height: VH } = VIDEO;
/** The scan's full frame: no region has more work pixels (a quarter at 2016 px has 1.6% more). */
const BUDGET = (VW / (VH / REGIONS.fullLongSide.ring)) * REGIONS.fullLongSide.ring;

/** Whole video pixels, inside the frame, and the scales exactly the source over the work size. */
function wellFormed(r: Region): void {
  for (const v of [r.x, r.y, r.width, r.height, r.workWidth, r.workHeight]) expect(Number.isInteger(v)).toBe(true);
  expect(r.x).toBeGreaterThanOrEqual(0);
  expect(r.y).toBeGreaterThanOrEqual(0);
  expect(r.x + r.width).toBeLessThanOrEqual(VW);
  expect(r.y + r.height).toBeLessThanOrEqual(VH);
  expect(r.scaleX).toBe(r.width / r.workWidth);
  expect(r.scaleY).toBe(r.height / r.workHeight);
  expect(r.workWidth * r.workHeight).toBeLessThanOrEqual(BUDGET * 1.02);
}

/** One frame of the camera view: the region the tracker asks for, drawn and searched;
 * the analysis (pipeline/live.ts, as the live worker runs it) on every `every`-th frame. */
function run(tracker: RegionTracker, draw: (r: Region) => Blob[], frames: number, every = 1) {
  const out: { r: Region; blobs: Blob[] }[] = [];
  for (let f = 0; f < frames; f++) {
    const r = tracker.next(VW, VH);
    wellFormed(r);
    const blobs = draw(r);
    tracker.observe(blobs);
    if (f % every === 0) {
      const a = analyse(blobs.map((b) => [b.x, b.y]), blobs.map((b) => b.d), [VW, VH], SPEC, null);
      tracker.chainFound(a.order.map((i) => [blobs[i]!.x, blobs[i]!.y] as Point), a.order.map((i) => blobs[i]!.d), a.gaps.length);
    }
    out.push({ r, blobs });
  }
  return out;
}

const found = (pins: Point[], blobs: Blob[]) => pins.filter(([x, y]) => blobs.some((b) => Math.hypot(b.x - x, b.y - y) < 0.3 * b.d)).length;

describe("regions", { timeout: 120_000 }, () => {
  it("scans the full frame, then each quarter finer, in turn, while there is no chain", () => {
    const t = new RegionTracker(RING, 13);
    const rs = Array.from({ length: 6 }, () => t.next(VW, VH));
    rs.forEach(wellFormed);
    expect(rs.map((r) => r.kind)).toEqual(["full", "tile", "tile", "tile", "tile", "full"]);
    expect([rs[0]!.workWidth, rs[0]!.workHeight]).toEqual([900, 1200]);
    // The quarters cover the frame, overlapping, at the scan's finer scale.
    const tiles = rs.slice(1, 5);
    expect(tiles.map((r) => [r.x, r.y])).toEqual([[0, 0], [VW - tiles[1]!.width, 0], [0, VH - tiles[2]!.height], [VW - tiles[3]!.width, VH - tiles[3]!.height]]);
    for (const r of tiles) {
      expect(r.width).toBeGreaterThan(VW / 2);
      expect(r.height).toBeGreaterThan(VH / 2);
      expect(r.scaleX).toBeCloseTo(4032 / REGIONS.tileLongSide.ring, 2);
    }
    // Sharpness is read at the old full frame's scale (800 px) whatever is drawn.
    for (const r of rs) expect(r.refScale).toBeCloseTo(4032 / 800, 9);
  });

  it("finds a chain spanning 0.22 from a cold start in a quarter, then draws only the region round it", () => {
    const sc = snapScene(0.22, 128, 0.3, 0.7);           // off centre: in the lower-left quarter
    const frames = run(new RegionTracker(RING, 13), (r) => detectBlobs(sc.draw(r), r, RING), 9);
    expect(found(sc.pins, frames[0]!.blobs)).toBeLessThan(13);        // the full frame at 1200 px: too coarse
    const first = frames.findIndex((f) => found(sc.pins, f.blobs) === 13);
    expect(first).toBeGreaterThan(0);
    expect(first).toBeLessThanOrEqual(4);
    const tracked = frames.filter((f) => f.r.kind === "track");
    expect(tracked.length).toBeGreaterThanOrEqual(3);
    for (const f of tracked) {
      expect(found(sc.pins, f.blobs)).toBe(13);
      expect(f.r.workWidth * f.r.workHeight).toBeLessThan(BUDGET / 4);
      expect(sc.rim / f.r.scaleX).toBeGreaterThan(17);              // rims about REGIONS.target across
    }
  });

  it("follows a chain that moves, and scans again once it is gone", () => {
    const sc = snapScene(0.3, 128);
    let dx = 0, gone = false;
    const t = new RegionTracker(RING, 13);
    const draw = (r: Region) => {
      if (gone) return [];
      // The picture moved right by dx: draw from dx further left; blobs are dx further right.
      const blobs = detectBlobs(sc.draw({ ...r, x: r.x - dx }), { ...r, x: r.x - dx }, RING);
      return blobs.map((b) => ({ ...b, x: b.x + dx }));
    };
    run(t, draw, 2);
    expect(t.tracking).toBe(true);
    for (let f = 0; f < 10; f++) {
      dx += 20;                                           // a fifth of a link a frame
      const [frame] = run(t, draw, 1, 1000);              // no analyses: followed from frame to frame
      expect(frame!.r.kind).toBe("track");
      expect(found(sc.pins.map(([x, y]) => [x + dx, y] as Point), frame!.blobs)).toBe(13);
    }
    gone = true;
    const after = run(t, draw, REGIONS.lostFrames + 1, 1000);
    expect(after.slice(0, REGIONS.lostFrames).every((f) => f.r.kind === "track")).toBe(true);
    expect(after.at(-1)!.r.kind).toBe("full");
  });

  it("scans again after analyses that find no chain in the tracked region, though its blobs stay", () => {
    const t = new RegionTracker(RING, 13);
    t.next(VW, VH);
    const pins: Point[] = [[1000, 1000], [1100, 1000], [1200, 1000]];
    t.chainFound(pins, [60, 60, 60], 0);
    const blobs = pins.map(([x, y]) => ({ x, y, d: 60, contrast: 50, sharpness: 1 }));
    for (let k = 0; k < REGIONS.emptyAnalyses; k++) {
      expect(t.next(VW, VH).kind).toBe("track");
      t.observe(blobs);
      t.chainFound([], [], 0);
    }
    expect(t.next(VW, VH).kind).toBe("full");
  });

  it("grows the region by a link for each pin not seen, and draws a large one coarser", () => {
    const t = new RegionTracker(RING, 13);
    t.next(VW, VH);
    const line = (n: number): Point[] => Array.from({ length: n }, (_, k) => [1200 + 100 * k, 2000] as Point);
    t.chainFound(line(13), Array(13).fill(40), 0);
    const whole = t.next(VW, VH);
    t.chainFound(line(6), Array(6).fill(40), 0);         // 7 pins not seen: 7 links more each way
    const part = t.next(VW, VH);
    expect(part.y).toBeLessThan(whole.y - 600);
    expect(part.height).toBeGreaterThan(whole.height + 1200);
    expect(part.scaleX).toBeGreaterThanOrEqual(whole.scaleX);
    t.chainFound(Array.from({ length: 13 }, (_, k) => [200 + 220 * k, 300 + 280 * k] as Point), Array(13).fill(40), 0);
    const huge = t.next(VW, VH);                          // most of the frame: no more work pixels than the scan's full frame
    wellFormed(huge);
    expect(huge.scaleX).toBeGreaterThan(40 / REGIONS.target.ring);
  });

  it("keeps its scale while the rims' size changes little, and starts again for another video size", () => {
    const t = new RegionTracker(RING, 13);
    t.next(VW, VH);
    const pins: Point[] = Array.from({ length: 13 }, (_, k) => [1000 + 60 * k, 2000] as Point);
    t.chainFound(pins, Array(13).fill(40), 0);
    const a = t.next(VW, VH);
    t.chainFound(pins, Array(13).fill(44), 0);           // 10% larger: the same scale
    expect(t.next(VW, VH).scaleX).toBeCloseTo(a.scaleX, 2);
    t.chainFound(pins, Array(13).fill(60), 0);           // 50% larger: coarser
    expect(t.next(VW, VH).scaleX).toBeGreaterThan(a.scaleX * 1.3);
    expect(t.next(1080, 1920).kind).toBe("full");          // another camera: no chain yet
  });

  it("never draws a small video larger than it is", () => {
    const t = new RegionTracker(RING, 13);
    for (let k = 0; k < 5; k++) {
      const r = t.next(480, 640);
      expect(r.scaleX).toBeGreaterThanOrEqual(1);
      expect(r.workWidth).toBeLessThanOrEqual(480);
    }
    expect(fullRegion(480, 640, 800)).toMatchObject({ workWidth: 480, workHeight: 640, scaleX: 1, refScale: 1 });
    expect(region(10.4, 20.6, 100, 100, 2, 2, 480, 640, "track")).toMatchObject({ x: 10, y: 21, workWidth: 50, width: 100 });
  });
});
