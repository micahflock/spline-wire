// Which part of the video frame the live detector reads, and how finely
// (docs/open-items.md, live view resolution).
//
// The snap-fit chain's rims are thin: on a full frame drawn at 800 px they are
// lost once the chain spans less than ~0.40 of the frame's short side. Reading
// the whole frame finer costs in proportion to its area, so instead:
//
//   - Tracking. Once the chain is known, only the box round it (plus a margin
//     of links, and a link more for each pin not yet seen) is drawn, at the
//     scale that makes its rims REGIONS.target work pixels across. The box
//     then has about the same number of work pixels however far the phone is,
//     and fewer than a full frame. The box goes where the chain is heading:
//     each frame's pins are found again by the shift that best fits the
//     frame's blobs (the last frame's motion first, then any blob against a
//     pin), and the next box is drawn that much further on. A frame with
//     fewer than 3 pins grows the box by REGIONS.lostGrowLinks links a side
//     before the scan starts again.
//   - Scanning, when there is no chain: the full frame at REGIONS.fullLongSide,
//     then each quarter of the frame (overlapping) at REGIONS.tileLongSide, in
//     turn. A chain too small for the full frame is found in a quarter; part
//     of one at a quarter's edge is enough, as the tracked box grows by the
//     pins it has not seen.
//
// Pure: the page asks next() for the region to draw, and reports what each
// frame found (observe) and what each analysis ordered (chainFound).

import type { Shape } from "./blobs";
import { WORK_LONG_SIDE, WORK_LONG_SIDE_RINGS, median, type Window } from "./blobs";
import type { Blob, Point } from "./types";

export interface Region extends Window {
  /** The source rectangle in video pixels: drawImage's sx, sy, sw, sh. */
  width: number;
  height: number;
  /** The work frame it is drawn into: drawImage's dw, dh. */
  workWidth: number;
  workHeight: number;
  kind: "full" | "tile" | "track";
}

/** Measured for the snap-fit chain's rims (experiments/live_view/regions.ts).
 * The dot chain's values are the rims' scaled as their old work sizes were
 * (640 against 800 px), with its dots drawn at the size the coach's limits were
 * calibrated on (coach.ts, minSpan): not measured on their own. */
export const REGIONS = {
  /** The full frame's long side when scanning, for rings and for dots. */
  fullLongSide: { ring: 1200, dot: 960 },
  /** Each quarter's scale when scanning, as the long side of the whole frame
   * drawn at it: a quarter then costs about what the full frame does (0.6² of
   * 2016² against 1200², experiments/live_view/regions.ts), and a chain spanning
   * 0.22 of the short side shows rims ~19 px across, found at 13 of 13 from ~14.5. */
  tileLongSide: { ring: 2016, dot: 1600 },
  /** A quarter's share of each side; quarters start at 0 and 1 - tile, so they overlap by 2 × tile - 1. */
  tile: 0.6,
  /** The finest a tracked region is drawn, as the long side of the whole frame at that scale. */
  finestLongSide: 2016,
  /** A tracked fiducial's outer diameter in work pixels. Snap-fit rims are all
   * found from ~14.5 px (experiments/live_view/regions.ts: 13 of 13 at 14.3-14.9 px,
   * 9 of 13 at 13.1-13.7): 18 leaves a fifth to spare. Dots from ~10 px (coach.ts). */
  target: { ring: 18, dot: 12 },
  /** The scale is kept while the target scale is within this factor of it. */
  hysteresis: 1.25,
  /** Margin round the tracked pins, in links, plus one for each pin not seen. */
  marginLinks: 1.5,
  /** Frames in a row with fewer than 3 tracked pins found before scanning again. */
  lostFrames: 4,
  /** Links the box grows by on each side for each frame in a row with fewer than 3 pins found. */
  lostGrowLinks: 2,
  /** The tracked pins are matched to a frame's blobs within this share of a link. */
  reachLinks: 0.5,
  /** The last motion is taken as the frame's shift if it finds at least this
   * share of the pins; else every blob is tried against the pins as a shift. */
  keepShift: 0.75,
  /** Analyses in a row that find no chain in the tracked region before scanning again. */
  emptyAnalyses: 2,
};

export class RegionTracker {
  /** The tracked pins, video pixels, moved to each frame's blobs (as trace.ts does). */
  private pins: Point[] = [];
  /** How far the pins moved over the last frame, video pixels: where the next box is drawn. */
  private velocity: Point = [0, 0];
  /** Pins the last chain lacked: bridged over, or not seen at all. */
  private missing = 0;
  private diameter = 0;
  private scale = 0;
  private lost = 0;
  private empty = 0;
  private scan = 0;
  private size = { width: 0, height: 0 };
  private readonly kind: "ring" | "dot";

  constructor(shape: Shape, private readonly nPins: number) {
    this.kind = shape.inner > 0 ? "ring" : "dot";
  }

  /** True while a chain is being tracked. */
  get tracking(): boolean {
    return this.pins.length >= 3;
  }

  /** The region to draw the next frame from. */
  next(videoWidth: number, videoHeight: number): Region {
    if (videoWidth !== this.size.width || videoHeight !== this.size.height) {
      if (this.size.width) this.reset();              // another camera, or the stream changed size
      this.size = { width: videoWidth, height: videoHeight };
    }
    if (this.tracking) return this.tracked(videoWidth, videoHeight);
    const step = this.scan++ % 5;
    const { ref, coarsest } = this.scales(videoWidth, videoHeight);
    if (step === 0) return region(0, 0, videoWidth, videoHeight, coarsest, ref, videoWidth, videoHeight, "full");
    const t = REGIONS.tile, i = (step - 1) % 2, j = (step - 1) >> 1;
    const tile = Math.max(1, Math.max(videoWidth, videoHeight) / REGIONS.tileLongSide[this.kind]);
    return region(i * (1 - t) * videoWidth, j * (1 - t) * videoHeight, t * videoWidth, t * videoHeight,
      tile, ref, videoWidth, videoHeight, "tile");
  }

  /** What the frame drawn from the last region found (video pixels). */
  observe(blobs: readonly Blob[]): void {
    if (!this.tracking) return;
    const reach = this.reach();
    const shift = bestShift(this.pins, blobs, this.velocity, reach);
    let seen = 0;
    const ds: number[] = [], moved: Point[] = [];
    this.pins = this.pins.map(([x, y]) => {
      const b = nearest(blobs, x + shift[0], y + shift[1], reach);
      if (!b) return [x + shift[0], y + shift[1]];
      seen++;
      ds.push(b.d);
      moved.push([b.x - x, b.y - y]);
      return [b.x, b.y];
    });
    if (ds.length) this.diameter = median(ds);
    if (seen >= 3) {
      this.lost = 0;
      this.velocity = [median(moved.map((m) => m[0])), median(moved.map((m) => m[1]))];
    } else if (++this.lost >= REGIONS.lostFrames) {
      this.drop();
    }
  }

  /** An analysis ordered `chain` (video pixels) among a frame's blobs, `missing` of
   * them bridged over: track it. With fewer than 3 pins it found no chain:
   * after REGIONS.emptyAnalyses of those in a row, scan again, even if the
   * tracked blobs are still there (they may be look-alikes that once ordered). */
  chainFound(chain: readonly Point[], diameters: readonly number[], missing: number): void {
    if (chain.length < 3) {
      if (this.tracking && ++this.empty >= REGIONS.emptyAnalyses) this.drop();
      return;
    }
    this.empty = 0;
    this.pins = chain.map((p) => [p[0], p[1]]);
    this.missing = Math.max(missing, this.nPins - chain.length);
    if (diameters.length) this.diameter = median([...diameters]);
    this.lost = 0;
  }

  reset(): void {
    this.drop();
    this.scale = 0;
  }

  private drop(): void {
    this.pins = [];
    this.velocity = [0, 0];
    this.missing = 0;
    this.lost = 0;
    this.empty = 0;
    this.scan = 0;
  }

  /** How far a pin may be from where the frame's shift puts it and still be followed. */
  private reach(): number {
    return REGIONS.reachLinks * this.link();
  }

  private link(): number {
    const p = this.pins;
    return median(p.slice(1).map((q, k) => Math.hypot(q[0] - p[k]![0], q[1] - p[k]![1])));
  }

  /** Video pixels per work pixel: of the full-frame work frame sharpness is read
   * at (blobs.workSize), of the full frame when scanning, and the finest. Never below 1. */
  private scales(vw: number, vh: number): { ref: number; coarsest: number; finest: number } {
    const long = Math.max(vw, vh);
    return {
      ref: Math.max(1, long / (this.kind === "ring" ? WORK_LONG_SIDE_RINGS : WORK_LONG_SIDE)),
      coarsest: Math.max(1, long / REGIONS.fullLongSide[this.kind]),
      finest: Math.max(1, long / REGIONS.finestLongSide),
    };
  }

  private tracked(vw: number, vh: number): Region {
    const { ref, coarsest, finest } = this.scales(vw, vh);
    // The scale that draws a fiducial `target` work pixels across, kept unless it moved by more than the hysteresis.
    const want = clamp(this.diameter / REGIONS.target[this.kind], finest, coarsest);
    if (!(this.scale > 0) || want > this.scale * REGIONS.hysteresis || want < this.scale / REGIONS.hysteresis) this.scale = want;
    // Where the pins will be if they move as they did over the last frame.
    const [vx, vy] = this.velocity;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [px, py] of this.pins) {
      const x = px + vx, y = py + vy;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    // The margin: links, one more for each pin not seen, more for each frame
    // the chain was lost, and half the last motion again (it may speed up).
    const m = (REGIONS.marginLinks + this.missing + this.lost * REGIONS.lostGrowLinks) * this.link()
      + this.diameter / 2 + 0.5 * Math.hypot(vx, vy);
    x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m);
    x1 = Math.min(vw, x1 + m); y1 = Math.min(vh, y1 + m);
    // No more work pixels than the full frame when scanning: draw a large box coarser.
    const budget = (vw / coarsest) * (vh / coarsest);
    const scale = Math.max(this.scale, Math.sqrt(((x1 - x0) * (y1 - y0)) / budget));
    return region(x0, y0, x1 - x0, y1 - y0, Math.min(scale, coarsest), ref, vw, vh, "track");
  }
}

/** The blob nearest (x, y) within `reach`, or null. */
function nearest(blobs: readonly Blob[], x: number, y: number, reach: number): Blob | null {
  let best: Blob | null = null, bd = reach;
  for (const b of blobs) {
    const d = Math.hypot(b.x - x, b.y - y);
    if (d < bd) {
      bd = d;
      best = b;
    }
  }
  return best;
}

/** How far `pins` moved to reach `blobs`: `guess` (the last frame's motion) if
 * it puts REGIONS.keepShift of them on a blob within `reach`; else the shift
 * from some pin to some blob that puts the most there, the nearest `guess` among
 * equals (a chain shifted a link along itself fits all but an end pin: the true
 * shift fits more). Tried from three pins (the ends and the middle), so it
 * costs 3 × blobs candidates, not pins × blobs. Under 3 pins placed: `guess`. */
export function bestShift(pins: readonly Point[], blobs: readonly Blob[], guess: Point, reach: number): Point {
  const fit = (sx: number, sy: number) => {
    let n = 0;
    for (const [x, y] of pins) if (nearest(blobs, x + sx, y + sy, reach)) n++;
    return n;
  };
  let best = guess, bestN = fit(guess[0], guess[1]);
  if (bestN >= REGIONS.keepShift * pins.length) return guess;
  const from = [...new Set([0, pins.length >> 1, pins.length - 1])].map((i) => pins[i]!);
  let bestD = 0;
  for (const [px, py] of from) {
    for (const b of blobs) {
      const sx = b.x - px, sy = b.y - py;
      const n = fit(sx, sy);
      const d = Math.hypot(sx - guess[0], sy - guess[1]);
      if (n > bestN || (n === bestN && n > 0 && d < bestD)) {
        best = [sx, sy];
        bestN = n;
        bestD = d;
      }
    }
  }
  return bestN >= 3 ? best : guess;
}

/** The region of the video frame (vw × vh) from (x, y), w × h, drawn at about
 * `scale` video pixels per work pixel. The source rectangle is whole video
 * pixels, about the same centre, inside the frame (older Safari draws nothing
 * from a source rectangle that crosses the frame's edge); scaleX and scaleY
 * are exactly its size over the work frame's. */
export function region(x: number, y: number, w: number, h: number, scale: number, refScale: number,
  vw: number, vh: number, kind: Region["kind"]): Region {
  const workWidth = Math.max(1, Math.round(w / scale)), workHeight = Math.max(1, Math.round(h / scale));
  const width = kind === "full" ? vw : Math.min(vw, Math.max(1, Math.round(workWidth * scale)));
  const height = kind === "full" ? vh : Math.min(vh, Math.max(1, Math.round(workHeight * scale)));
  const sx = clamp(Math.round(x + (w - width) / 2), 0, vw - width), sy = clamp(Math.round(y + (h - height) / 2), 0, vh - height);
  return { x: sx, y: sy, width, height, workWidth, workHeight, scaleX: width / workWidth, scaleY: height / workHeight, refScale, kind };
}

/** A whole frame at `longSide` on its long side: what workSize() gives, as a region. */
export function fullRegion(vw: number, vh: number, longSide: number, refLongSide = longSide): Region {
  const long = Math.max(vw, vh);
  return region(0, 0, vw, vh, Math.max(1, long / longSide), Math.max(1, long / refLongSide), vw, vh, "full");
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
