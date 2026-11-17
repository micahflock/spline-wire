// Runs the overlay: frames in, coaching out, and the still at the right moment.
//
// The page owns the camera, the canvas and the worker; this owns the order of
// events. It is driven by the page calling frame() for each video frame
// (requestVideoFrameCallback) and has no timers of its own.
//
//   const overlay = new OverlaySession({
//     chain, focalPx: () => focalForThisVideoSize,          // null: estimate from the chain
//     analyse: (points) => core.call("analyse", points),
//     show: (view) => draw(view),                           // every frame
//     capture: () => takeStillAndMeasure(),                 // once per good hold; resolves when done
//   });
//   video.requestVideoFrameCallback(function each(now) {
//     overlay.frame(grabWorkFrame(video), video.videoWidth, video.videoHeight, now);
//     video.requestVideoFrameCallback(each);
//   });
//
// Blobs are found on every frame (a few ms). The analysis is slower (tens to
// hundreds of ms in the Pyodide worker), so only one runs at a time and frames
// that arrive meanwhile are not analysed: the coach hears a few readings a
// second, which is what its timings assume.

import type { FrameAnalysis, FramePoints } from "../core/protocol";
import type { ChainSpec } from "../core/types";
import { detectBlobs } from "./blobs";
import { Coach } from "./coach";
import { makeReading } from "./reading";
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
}

export class OverlaySession {
  readonly coach = new Coach();
  private busy = false;
  private capturing = false;
  private previous: FrameReading | null = null;
  private chain: Point[] = [];

  constructor(private readonly o: OverlayOptions) {}

  /** One video frame: `work` is the frame drawn small and gray (blobs.workSize, toGray). */
  frame(work: GrayFrame, videoWidth: number, videoHeight: number, t: number): void {
    const scale = videoWidth / work.width;
    const blobs = detectBlobs(work, scale);
    this.o.show({ state: this.coach.current, chain: this.chain, blobs });
    if (this.busy || this.capturing) return;
    this.busy = true;
    void this.analyse(blobs, videoWidth, videoHeight, t).finally(() => (this.busy = false));
  }

  private async analyse(blobs: Blob[], width: number, height: number, t: number): Promise<void> {
    let analysis: FrameAnalysis = { order: [], gaps: [], tilt_deg: null, link_residual_max_mm: null };
    if (blobs.length >= 3) {
      try {
        analysis = await this.o.analyse({
          points: blobs.map((b) => [b.x, b.y]), diameters: blobs.map((b) => b.d),
          width, height, chain: this.o.chain, focal_px: this.o.focalPx(),
        });
      } catch {
        return;                 // the worker is loading or restarting: no reading for this frame
      }
    }
    if (this.capturing) return;
    const reading = makeReading(t, { width, height }, blobs, analysis, this.o.chain.n_pins, this.previous);
    this.previous = reading;
    this.chain = reading.chain;
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
