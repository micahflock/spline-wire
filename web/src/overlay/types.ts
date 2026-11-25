// The live camera overlay (web-build.md, step 5), as a pipeline of small parts:
//
//   camera frame ──▶ blobs.ts        bright dots in the frame, every frame, in the page (TypeScript, ~ms)
//                      │
//                      ▼
//                core "analyse"      which blobs are the chain, in order, and the tilt
//                (pipeline/live.ts, as splinewire/live.py, in its own worker)
//                      │
//                      ▼
//                  reading.ts        one FrameReading: what the coach needs to know about a frame
//                      │
//                      ▼
//                   coach.ts         a state machine: what to tell the person, and when to take the still
//                      │
//                      ▼
//                  session.ts        runs the loop, draws nothing itself: hands the page a CoachState
//                                    and the chain to trace over the video
//
// The still taken at the end is measured by the full pipeline (decision D7).
// Nothing the overlay computes ends up in a measurement.
//
// Each part is pure or has its browser dependencies injected, so all of it is
// tested without a camera.

export type Point = [number, number];

/** A bright dot candidate in a frame, in the frame's own pixels. */
export interface Blob {
  x: number;
  y: number;
  /** Diameter of a disc of the blob's area. */
  d: number;
  /** Blob brightness minus its surround, in gray levels (0-255). */
  contrast: number;
  /** How sharp its edge is: 1 is a step, smaller is softer. See blobs.ts edgeSharpness. */
  sharpness: number;
}

/** A gray frame. `data` is row-major, one byte per pixel. */
export interface GrayFrame {
  width: number;
  height: number;
  data: Uint8Array;
}

/** Everything the coach is told about one analysed frame. */
export interface FrameReading {
  /** When the frame was taken, ms (performance.now or the video frame's time). */
  t: number;
  width: number;
  height: number;
  /** The chain's pins in order, frame pixels. Empty if no chain was found. */
  chain: Point[];
  pinsExpected: number;
  /** Pins the ordering bridged over. */
  missing: number;
  /** null when it cannot be told (too few pins). */
  tiltDeg: number | null;
  /** The way across the frame (unit vector, x right, y down) to move the phone
   * to look straight down on the chain; null with tiltDeg, or straight on. */
  tiltDir: Point | null;
  /** Median edge sharpness of the chain's blobs, 0-1. */
  sharpness: number;
  /** How far the chain moved since the previous reading, in pitches per second. null on the first. */
  motion: number | null;
}

export type Problem =
  | "no-chain"       // nothing like a chain in view
  | "pins-missing"   // part of the chain is out of view, hidden or lost in glare
  | "at-edge"        // the chain touches the frame's edge
  | "too-far"        // the chain is small in the frame
  | "tilted"         // the camera is far from straight-on
  | "moving"         // the camera or the chain is moving
  | "blurred";       // out of focus

export interface CoachState {
  phase:
    | "searching"    // no chain yet
    | "adjusting"    // a chain, with problems to fix
    | "steady"       // good: holding for HOLD_MS before the shot
    | "capture"      // take the still now (reported once)
    | "captured";    // waiting for the page to call reset()
  /** What to fix, most important first. Empty when steady. */
  problems: Problem[];
  /** 0 to 1 through the hold, for a ring that fills. */
  hold: number;
  /** Pins found and expected, for "11 of 13". */
  pinsFound: number;
  pinsExpected: number;
  /** A milder note that does not block the shot: the tilt is acceptable but straighter is better. */
  straighter: boolean;
  /** Which way to move the phone to look straighter down on the chain, for an
   * arrow over it: a unit vector in the frame's pixel axes (x right, y down),
   * smoothed over readings, with the tilt (degrees) it was last read at; null
   * when the tilt is small enough not to show one. */
  arrow: { dir: Point; tiltDeg: number } | null;
}
