import { describe, expect, it } from "vitest";
import type { FramePoints } from "../core/protocol";
import { specFromDict } from "./chain";
import { analyse } from "./live";
import { LiveAnalyser } from "./live-client";

/** A worker that runs live.ts in the same thread, as live-worker.ts does in its own. */
function fakeWorker(opts: { crash?: boolean } = {}) {
  const w = {
    onmessage: null as ((e: MessageEvent) => void) | null,
    onerror: null as ((e: ErrorEvent) => void) | null,
    terminated: false,
    postMessage(msg: { id: number; points: FramePoints }) {
      queueMicrotask(() => {
        if (opts.crash) return w.onerror?.({ message: "boom" } as ErrorEvent);
        const p = msg.points;
        const result = analyse(p.points, p.diameters, [p.width, p.height], specFromDict(p.chain as never), p.focal_px);
        w.onmessage?.({ data: { id: msg.id, result } } as MessageEvent);
      });
    },
    terminate() {
      w.terminated = true;
    },
  };
  return w;
}

const chain = { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 5, max_bend_deg: 80 };
const line: FramePoints = {
  points: [[100, 100], [120, 100], [140, 101], [160, 103], [180, 106]], diameters: [10, 10, 10, 10, 10],
  width: 640, height: 480, chain, focal_px: 500,
};

describe("LiveAnalyser", () => {
  it("answers each call with its own analysis", async () => {
    const a = new LiveAnalyser(() => fakeWorker());
    const [one, none] = await Promise.all([a.analyse(line), a.analyse({ ...line, points: line.points.slice(0, 2), diameters: [10, 10] })]);
    expect(one.order).toEqual([0, 1, 2, 3, 4]);
    expect(none.order).toEqual([]);
  });

  it("fails the calls in flight when the worker dies, and starts a new one next time", async () => {
    let made = 0;
    const a = new LiveAnalyser(() => fakeWorker({ crash: made++ === 0 }));
    await expect(a.analyse(line)).rejects.toThrow("the live analysis stopped: boom");
    expect((await a.analyse(line)).order).toHaveLength(5);
    expect(made).toBe(2);
  });
});
