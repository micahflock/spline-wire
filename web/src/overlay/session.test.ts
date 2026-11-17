import { describe, expect, it } from "vitest";
import type { FramePoints } from "../core/protocol";
import { motion } from "./reading";
import { OverlaySession, type OverlayView } from "./session";
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
      overlay.frame(workFrame(i * 6).frame, 1920, 1080, i * 100);     // one pitch a second
      await turn();
    }
    expect(captures).toBe(0);
    expect(overlay.coach.current.problems).toContain("moving");
  });

  it("measures motion in pitches per second, whichever end the order starts from", () => {
    const pts: Point[] = [[0, 0], [100, 0], [200, 0], [300, 0]];
    const prev = { t: 0, chain: pts } as FrameReading;
    const moved = pts.map(([x, y]) => [x + 10, y] as Point).reverse();
    expect(motion(prev, moved, 500)).toBeCloseTo(0.2);
    expect(motion(prev, [], 500)).toBeNull();
  });
});
