// The chain drawn over the live video. The analysis that ordered it ran on an
// earlier frame (tens to hundreds of ms ago), so each pin is moved to the
// nearest blob of the current frame when there is one close by: the trace then
// follows the video at frame rate instead of lagging behind a moving phone.

import type { Blob, Point } from "./types";

/** `chain` with each pin moved to the nearest of `blobs` within `reach`
 * (px); pins with no blob that close stay where the analysis put them. With
 * no `reach`, half the chain's median link. */
export function followChain(chain: readonly Point[], blobs: readonly Blob[], reach?: number): Point[] {
  if (chain.length === 0) return [];
  let r = reach;
  if (r === undefined) {
    const links = chain.slice(1).map((p, k) => Math.hypot(p[0] - chain[k]![0], p[1] - chain[k]![1])).sort((a, b) => a - b);
    r = links.length ? links[links.length >> 1]! / 2 : 0;
  }
  return chain.map(([x, y]) => {
    let best: Point = [x, y], bd = r!;
    for (const b of blobs) {
      const d = Math.hypot(b.x - x, b.y - y);
      if (d < bd) {
        bd = d;
        best = [b.x, b.y];
      }
    }
    return best;
  });
}
