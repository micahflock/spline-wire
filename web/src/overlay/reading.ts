// From one frame's blobs and their analysis to the FrameReading the coach reads.

import type { FrameAnalysis } from "../core/protocol";
import { median } from "./blobs";
import type { Blob, FrameReading, Point } from "./types";

export function makeReading(
  t: number,
  size: { width: number; height: number },
  blobs: readonly Blob[],
  analysis: FrameAnalysis,
  pinsExpected: number,
  previous: FrameReading | null,
): FrameReading {
  const on = analysis.order.map((i) => blobs[i]).filter((b): b is Blob => b !== undefined);
  const chain: Point[] = on.map((b) => [b.x, b.y]);
  return {
    t, width: size.width, height: size.height, chain, pinsExpected,
    missing: analysis.gaps.length,
    tiltDeg: analysis.tilt_deg,
    sharpness: on.length ? median(on.map((b) => b.sharpness)) : 0,
    motion: previous ? motion(previous, chain, t) : null,
  };
}

/** How far the chain moved since `previous`, in pitches per second: the median
 * over pins of the distance to the nearest pin of the previous chain. Nearest
 * rather than same-index, because the ordering may start from either end and
 * a pin may come and go. null if either frame has no chain. */
export function motion(previous: FrameReading, chain: readonly Point[], t: number): number | null {
  const dt = (t - previous.t) / 1000;
  if (previous.chain.length < 3 || chain.length < 3 || dt <= 0) return null;
  const pitch = median(chain.slice(1).map((p, k) => Math.hypot(p[0] - chain[k]![0], p[1] - chain[k]![1])));
  if (!(pitch > 0)) return null;
  const moved = chain.map(([x, y]) => Math.min(...previous.chain.map(([px, py]) => Math.hypot(x - px, y - py))));
  return median(moved) / pitch / dt;
}
