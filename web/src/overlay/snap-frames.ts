// Test frames of the snap-fit chain (data/chain-snap.yaml) for the live view's
// tests: a 13-pin chain seen straight on in a phone's 3024×4032 portrait
// video, drawn into any region of it at any scale as the camera view's
// drawImage would (each work pixel the mean of the video it covers, here by
// supersampling). Only the tests import this.
//
// The look is blobs.test.ts's ringChain: a thin light rim round a dark hole
// on a head as wide as the rim, the dark arms between heads, a flat table.

import type { Window } from "./blobs";
import type { GrayFrame, Point } from "./types";

export const SNAP_SPEC = {
  pitch_mm: 13, half_width_mm: 4.39, fiducial: "ring", fiducial_mm: 8.79, ring_inner_mm: 7.07,
  n_pins: 13, max_bend_deg: 110, contact: "round",
} as const;

export const VIDEO = { width: 3024, height: 4032 };

export interface SnapScene {
  /** Pin centres, video pixels (pixel-centre coordinates, as blobs are reported). */
  pins: Point[];
  /** The rim's outer diameter, video pixels. */
  rim: number;
  /** Gray levels. */
  table: number;
  draw(win: Window & { workWidth: number; workHeight: number }): GrayFrame;
}

/** The chain spanning `span` of the frame's short side (its pins' extent plus
 * a rim), centred at (cx, cy) as shares of the frame, on a table of gray level `table`. */
export function snapScene(span: number, table = 128, cx = 0.5, cy = 0.45): SnapScene {
  // A gentle S: the heading swings ±0.6 rad over the chain, each link exactly one pitch.
  const mm: Point[] = [[0, 0]];
  for (let k = 1; k < SNAP_SPEC.n_pins; k++) {
    const a = 0.6 * Math.sin((2 * Math.PI * (k - 0.5)) / (SNAP_SPEC.n_pins - 1));
    const [x, y] = mm[k - 1]!;
    mm.push([x + SNAP_SPEC.pitch_mm * Math.cos(a), y + SNAP_SPEC.pitch_mm * Math.sin(a)]);
  }
  const xs = mm.map((p) => p[0]), ys = mm.map((p) => p[1]);
  const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + SNAP_SPEC.fiducial_mm;
  const k = (span * Math.min(VIDEO.width, VIDEO.height)) / extent;           // video px per mm
  const mx = (Math.max(...xs) + Math.min(...xs)) / 2, my = (Math.max(...ys) + Math.min(...ys)) / 2;
  const pins: Point[] = mm.map(([x, y]) => [cx * VIDEO.width + (x - mx) * k, cy * VIDEO.height + (y - my) * k]);
  const outer = (SNAP_SPEC.fiducial_mm / 2) * k, hole = (SNAP_SPEC.ring_inner_mm / 2) * k, arm = 0.57 * outer;
  const x0 = Math.min(...pins.map((p) => p[0])) - outer - 1, x1 = Math.max(...pins.map((p) => p[0])) + outer + 1;
  const y0 = Math.min(...pins.map((p) => p[1])) - outer - 1, y1 = Math.max(...pins.map((p) => p[1])) + outer + 1;
  const at = (x: number, y: number): number => {
    if (x < x0 || x > x1 || y < y0 || y > y1) return table;
    for (const [px, py] of pins) {
      const r = Math.hypot(x - px, y - py);
      if (r <= hole) return 20;
      if (r <= outer) return 225;
    }
    for (let i = 0; i + 1 < pins.length; i++) {
      const [ax, ay] = pins[i]!, [bx, by] = pins[i + 1]!;
      const t = Math.min(1, Math.max(0, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)));
      if (Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)) <= arm) return 40;
    }
    return table;
  };
  return {
    pins, rim: 2 * outer, table,
    draw(win) {
      const { workWidth: w, workHeight: h } = win;
      const data = new Uint8Array(w * h);
      const ssx = Math.max(2, Math.ceil(win.scaleX)), ssy = Math.max(2, Math.ceil(win.scaleY));
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) {
          // The work pixel covers video [x + i·sx, x + (i+1)·sx); a video pixel's centre is at its index + 0.5.
          const u0 = win.x + i * win.scaleX - 0.5, v0 = win.y + j * win.scaleY - 0.5;
          const inside = u0 + win.scaleX >= x0 && u0 <= x1 && v0 + win.scaleY >= y0 && v0 <= y1;
          if (!inside) {
            data[j * w + i] = table;
            continue;
          }
          let sum = 0;
          for (let b = 0; b < ssy; b++) {
            for (let a = 0; a < ssx; a++) sum += at(u0 + ((a + 0.5) / ssx) * win.scaleX, v0 + ((b + 0.5) / ssy) * win.scaleY);
          }
          data[j * w + i] = Math.round(sum / (ssx * ssy));
        }
      }
      return { width: w, height: h, data };
    },
  };
}
