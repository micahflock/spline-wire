import { describe, expect, it } from "vitest";
import type { FramePoints } from "../core/protocol";
import { specFromDict } from "../pipeline/chain";
import { analyse as liveAnalyse } from "../pipeline/live";
import { motion } from "./reading";
import { OverlaySession, type OverlayView } from "./session";
import { SNAP_SPEC, snapScene, VIDEO } from "./snap-frames";
import { followChain } from "./trace";
import type { FrameReading, GrayFrame, Point } from "./types";

/** A 640×360 work frame with a 9-pin chain: dark links, light dots, drawn without antialiasing. */
function workFrame(shift = 0): { frame: GrayFrame; pins: Point[] } {
  const w = 640, h = 360;
  const pins: Point[] = Array.from({ length: 9 }, (_, k) => [80 + 60 * k + shift, 180 + 40 * Math.sin(k * 0.8)]);
  const data = new Uint8Array(w * h).fill(128);
  const disc = (cx: number, cy: number, r: number, v: number) => {
    for (let y = Math.floor(cy - r); y <= cy + r; y++) {
      for (let x = Math.floor(cx - r); x <= cx + r; x++) {
        if (x >= 0 && x < w && y >= 0 && y < h && Math.hypot(x - cx, y - cy) <= r) data[y * w + x] = v;
      }
    }
  };
  for (let k = 0; k + 1 < pins.length; k++) {
    for (let i = 0; i <= 60; i++) {
      disc(pins[k]![0] + ((pins[k + 1]![0] - pins[k]![0]) * i) / 60, pins[k]![1] + ((pins[k + 1]![1] - pins[k]![1]) * i) / 60, 18, 30);
    }
  }
  for (const [x, y] of pins) disc(x, y, 11, 230);
  return { frame: { width: w, height: h, data }, pins };
}

/** Stands in for splinewire.live.analyse: orders the points left to right. */
async function analyse(p: FramePoints) {
  const order = p.points.map((_, i) => i).sort((a, b) => p.points[a]![0] - p.points[b]![0]);
  return { order, gaps: [], tilt_deg: 8, link_residual_max_mm: 0.05 };
}

const chain = { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 9 };
const turn = () => new Promise((r) => setTimeout(r, 0));

describe("overlay session", () => {
  it("coaches from video frames and takes the still once the chain has been steady", async () => {
    const views: OverlayView[] = [];
    let captures = 0;
    let finish: () => void = () => {};
    const overlay = new OverlaySession({
      chain, focalPx: () => 1500, analyse, show: (v) => views.push(v),
      capture: () => new Promise<void>((r) => { captures++; finish = r; }),
    });
    const { frame, pins } = workFrame();
    for (let t = 0; t <= 800 && captures === 0; t += 100) {
      overlay.frame(frame, 1920, 1080, t);
      await turn();
    }
    expect(captures).toBe(1);
    const last = views.at(-1)!;
    expect(last.blobs.length).toBe(9);
    expect(last.chain.length).toBe(9);
    expect(Math.abs(last.chain[0]![0] - (3 * pins[0]![0] + 1))).toBeLessThan(2);     // in video pixels
    overlay.frame(frame, 1920, 1080, 900);                    // while the still is being measured: no analysis
    await turn();
    expect(captures).toBe(1);
    finish();
    await turn();
    expect(overlay.coach.current.phase).toBe("searching");    // coaching again
  });

  it("does not take the still while the chain is moving", async () => {
    let captures = 0;
    const overlay = new OverlaySession({
      chain, focalPx: () => null, analyse, show: () => {}, capture: async () => void captures++,
    });
    for (let i = 0; i < 12; i++) {
      overlay.frame(workFrame(i * 12).frame, 1920, 1080, i * 100);    // two pitches a second
      await turn();
    }
    expect(captures).toBe(0);
    expect(overlay.coach.current.problems).toContain("moving");
  });

  it("analyses at most once per minIntervalMs, and never two at once", async () => {
    let calls = 0, open = 0, most = 0;
    const slow = async (p: FramePoints) => {
      calls++;
      most = Math.max(most, ++open);
      await turn();
      open--;
      return analyse(p);
    };
    const overlay = new OverlaySession({
      chain, focalPx: () => null, analyse: slow, show: () => {}, capture: async () => {}, minIntervalMs: () => 1000,
    });
    for (let t = 0; t <= 2500; t += 100) {
      overlay.frame(workFrame().frame, 1920, 1080, t);
      overlay.frame(workFrame().frame, 1920, 1080, t + 1);       // a second frame while the first is analysed
      await turn();
      await turn();
    }
    expect(calls).toBe(3);                                      // at 0, 1000 and 2000 ms
    expect(most).toBe(1);
    expect(overlay.stats).toMatchObject({ frames: 52, analysed: 3 });
  });

  it("measures motion in pitches per second, whichever end the order starts from", () => {
    const pts: Point[] = [[0, 0], [100, 0], [200, 0], [300, 0]];
    const prev = { t: 0, chain: pts } as FrameReading;
    const moved = pts.map(([x, y]) => [x + 10, y] as Point).reverse();
    expect(motion(prev, moved, 500)).toBeCloseTo(0.2);
    expect(motion(prev, [], 500)).toBeNull();
  });
});

it("searches for the fiducial its chain preset names", () => {
  const snap = { pitch_mm: 13, half_width_mm: 4.68, fiducial_mm: 7.2, n_pins: 13, fiducial: "ring" as const, ring_inner_mm: 5.48 };
  const o = { focalPx: () => null, analyse, show: () => {}, capture: async () => {} };
  expect(new OverlaySession({ ...o, chain: snap }).shape.inner).toBeCloseTo(0.761, 3);
  expect(new OverlaySession({ ...o, chain }).shape.inner).toBe(0);
});

it("draws regions round the snap-fit chain at span 0.29, and the trace lands on its pins in video pixels", { timeout: 120_000 }, async () => {
  // 0.29 with a rim; the pins alone span 0.27 of the short side, past the coach's minSpan.enter (0.26).
  const sc = snapScene(0.29, 128, 0.62, 0.35);
  const spec = specFromDict(SNAP_SPEC);
  const views: OverlayView[] = [];
  const kinds: string[] = [];
  let captures = 0;
  const overlay = new OverlaySession({
    chain: SNAP_SPEC, focalPx: () => null, show: (v) => views.push(v), capture: async () => void captures++,
    // The live worker's analysis, answered at once.
    analyse: async (p) => liveAnalyse(p.points, p.diameters, [p.width, p.height], spec, p.focal_px),
  });
  for (let t = 0; t <= 1500 && captures === 0; t += 100) {
    const r = overlay.region(VIDEO.width, VIDEO.height);
    kinds.push(r.kind);
    overlay.frame(sc.draw(r), VIDEO.width, VIDEO.height, t, r);
    await turn();
  }
  expect(kinds[0]).toBe("full");
  expect(kinds.indexOf("track")).toBeGreaterThan(0);           // tracked once an analysis has ordered it
  expect(kinds.indexOf("track")).toBeLessThanOrEqual(5);
  expect(overlay.tracking).toBe(true);
  expect(captures).toBe(1);                                     // near enough to shoot
  const last = views.at(-1)!;
  const near = (pts: Point[]) => Math.max(...sc.pins.map(([x, y]) => Math.min(...pts.map(([px, py]) => Math.hypot(px - x, py - y)))));
  expect(last.chain.length).toBe(13);
  expect(near(last.chain)).toBeLessThan(0.3);
  expect(near(followChain(last.chain, last.blobs))).toBeLessThan(0.3);
});
