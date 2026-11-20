// Runs the overlay: frames in, coaching out, and the still at the right moment.
//
// The page owns the camera, the canvas and the worker; this owns the order of
// events. It is driven by the page calling frame() for each video frame
// (requestVideoFrameCallback) and has no timers of its own.
//
//   const overlay = new OverlaySession({
//     chain, focalPx: () => focalForThisVideoSize,          // null: estimate from the chain
//     analyse: (points) => live.analyse(points),            // src/pipeline/live-client.ts
//     show: (view) => draw(view),                           // every frame
//     capture: () => takeStillAndMeasure(),                 // once per good hold; resolves when done
//   });
//   video.requestVideoFrameCallback(function each(now) {
//     const r = overlay.region(video.videoWidth, video.videoHeight);   // which part, how finely (region.ts)
//     overlay.frame(grabWorkFrame(video, r), video.videoWidth, video.videoHeight, now, r);
//     video.requestVideoFrameCallback(each);
//   });
//
// Blobs are found on every frame (a few ms). The analysis is slower (a few to
// ~160 ms in its worker), so only one runs at a time and frames
// that arrive meanwhile are not analysed: the coach hears a few readings a
// second, which is what its timings assume.

import type { FrameAnalysis, FramePoints } from "../core/protocol";
import type { ChainSpec } from "../core/types";
import { detectBlobs, shapeOf, type Shape, type Window } from "./blobs";
import { Coach } from "./coach";
import { makeReading } from "./reading";
import { RegionTracker, type Region } from "./region";
import type { Blob, CoachState, FrameReading, GrayFrame, Point } from "./types";

export interface OverlayView {
  state: CoachState;
  /** The chain to trace over the video, in video pixels, from the latest analysis. */
  chain: Point[];
  /** Every dot candidate of this frame, for a debug view. */
  blobs: Blob[];
}

export interface OverlayOptions {
  chain: ChainSpec;
  /** The focal length in pixels of the video frame, if known. */
  focalPx: () => number | null;
  analyse: (points: FramePoints) => Promise<FrameAnalysis>;
  show: (view: OverlayView) => void;
  /** Take the still and deal with it. The overlay coaches again when the promise settles. */
  capture: () => Promise<void>;
  /** With false, the overlay coaches but never takes the still itself (a shutter button does). */
  autoCapture?: boolean;
  /** Start an analysis at most this often, ms (default 0: whenever the last
   * one is done). For when each costs more than the phone should spend, such
   * as estimating the focal length (400-640 ms). */
  minIntervalMs?: () => number;
}

/** What the loop has done, for the page's diagnostics and the performance test. */
export interface OverlayStats {
  frames: number;
  analysed: number;
  /** Time spent finding blobs, ms, over all frames. */
  blobMs: number;
  /** The last analysis, ms from asking to the answer. */
  lastAnalyseMs: number;
}

export class OverlaySession {
  readonly coach = new Coach();
  private busy = false;
  private capturing = false;
  private previous: FrameReading | null = null;
  private chain: Point[] = [];
  private lastStart = -Infinity;
  private readonly regions: RegionTracker;
  readonly stats: OverlayStats = { frames: 0, analysed: 0, blobMs: 0, lastAnalyseMs: 0 };

  /** The chain's fiducial (a dot or a ring): what the frames are searched for,
   * and how large the page should draw them (blobs.workSize). */
  get shape(): Shape {
    return shapeOf(this.o.chain);
  }

  constructor(private readonly o: OverlayOptions) {
    this.regions = new RegionTracker(this.shape, o.chain.n_pins);
  }

  /** The part of the next video frame to draw, and its work size (region.ts):
   * round the chain while it is tracked, else the whole frame or a quarter of it. */
  region(videoWidth: number, videoHeight: number): Region {
    return this.regions.next(videoWidth, videoHeight);
  }

  /** True while the chain is tracked (the regions are drawn round it). */
  get tracking(): boolean {
    return this.regions.tracking;
  }

  /** One video frame: `work` is the frame drawn small and gray (toGray), from
   * `region` (this.region()); without one, from the whole frame (blobs.workSize). */
  frame(work: GrayFrame, videoWidth: number, videoHeight: number, t: number, region?: Window): void {
    const from: Window = region ?? { x: 0, y: 0, scaleX: videoWidth / work.width, scaleY: videoWidth / work.width };
    const t0 = performance.now();
    // The chain's own fiducial: a dot, or a ring like the snap-fit chain's rim.
    const blobs = detectBlobs(work, from, this.shape);
    this.found(blobs, videoWidth, videoHeight, t, performance.now() - t0);
  }

  /** One video frame's blobs (video pixels), found elsewhere (camera/grab-worker.ts)
   * from this.region() with this.shape, in `blobMs`. */
  found(blobs: Blob[], videoWidth: number, videoHeight: number, t: number, blobMs = 0): void {
    this.regions.observe(blobs);
    this.stats.blobMs += blobMs;
    this.stats.frames++;
    this.o.show({ state: this.coach.current, chain: this.chain, blobs });
    if (this.busy || this.capturing) return;
    if (t - this.lastStart < (this.o.minIntervalMs?.() ?? 0)) return;
    this.lastStart = t;
    this.busy = true;
    void this.analyse(blobs, videoWidth, videoHeight, t).finally(() => (this.busy = false));
  }

  private async analyse(blobs: Blob[], width: number, height: number, t: number): Promise<void> {
    let analysis: FrameAnalysis = { order: [], gaps: [], tilt_deg: null, link_residual_max_mm: null, tilt_dir: null };
    if (blobs.length >= 3) {
      const t0 = performance.now();
      try {
        analysis = await this.o.analyse({
          points: blobs.map((b) => [b.x, b.y]), diameters: blobs.map((b) => b.d),
          width, height, chain: this.o.chain, focal_px: this.o.focalPx(),
        });
      } catch {
        return;                 // the worker is loading or restarting: no reading for this frame
      }
      this.stats.lastAnalyseMs = performance.now() - t0;
      this.stats.analysed++;
    }
    if (this.capturing) return;
    const reading = makeReading(t, { width, height }, blobs, analysis, this.o.chain.n_pins, this.previous);
    this.previous = reading;
    this.chain = reading.chain;
    this.regions.chainFound(reading.chain, analysis.order.map((i) => blobs[i]?.d ?? 0), reading.missing);
    const state = this.coach.update(reading);
    if (state.phase !== "capture") return;
    if (this.o.autoCapture === false) {
      this.coach.reset();       // keep coaching; the person presses the shutter
      return;
    }
    this.capturing = true;
    try {
      await this.o.capture();
    } catch {
      /* the page reports its own failure */
    } finally {
      this.capturing = false;
      this.previous = null;
      this.coach.reset();
    }
  }
}
