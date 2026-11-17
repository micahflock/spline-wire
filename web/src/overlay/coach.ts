// The coach: from a stream of frame readings to what the person is told and
// the moment the still is taken. A pure state machine: feed it readings, read
// its state. No timers, no DOM.
//
// Entering "steady" needs every check to pass; staying there tolerates a
// little more (the `keep` thresholds), so a reading on the borderline does not
// make the hold flicker. A problem must persist for PROBLEM_MS before it is
// shown, so one bad frame does not flash a message.

import type { CoachState, FrameReading, Problem } from "./types";

export interface Limits {
  enter: number;
  keep: number;
}

/** The thresholds, in one place. Calibrated (2026-11-17) on S5's renders
 * (experiments/video_frame_accuracy.py, its 167 scenes) drawn small and gray
 * as the camera view does: each frame area-averaged to a 640 px work frame and
 * run through blobs.ts in Node. "1080p" and "4K" below are those renders; the
 * phone streams 3024×4032 (S3), which shrinks to its work frame by 6.3, like
 * the 4K renders (6). Not yet checked against a real phone's video. */
export const COACH = {
  /** Good readings must span this long before the shot. */
  HOLD_MS: 700,
  /** A gap between readings longer than this starts the hold again: the page
   * was hidden or the analysis stalled. Analysing a frame takes 76-105 ms in
   * desktop Chromium with a known focal length and 400-640 ms estimating it
   * (web-build.md, step 1), and the camera view then analyses at most once a
   * second (camera-view.ts), so readings can be ~1 s apart and a phone is
   * slower: 1.5 s. */
  STALE_MS: 1500,
  /** A problem is shown once it has lasted this long. */
  PROBLEM_MS: 250,
  /** Camera tilt from straight-on, degrees. In simulation a 2% focal error costs
   * up to 0.05 mm at 25° and 0.11 mm at 40° (open-questions.md): block the shot
   * above 25°, and nudge toward straighter above 12°. */
  tiltDeg: { enter: 25, keep: 30 } as Limits,
  straighterDeg: 12,
  /** The chain's extent as a fraction of the frame's shorter side. This guards
   * the overlay's own detector more than the still: the docs' benchmark still
   * spans 0.28 and measures 0.025 / 0.056 mm (median / p95 worst pin). Every
   * pin was found by blobs.ts in 120 of 167 4K renders at span 0.55 (dots 12 px
   * across in the work frame), 123 at 0.44 (10 px), 115 at 0.36 (8 px), then 84
   * at 0.28 (6 px) and 59 at 0.22. So the shot needs 0.40, and keeps going to
   * 0.33. (Was 0.45 / 0.38; asking for more also pushes the phone toward its
   * closest focus: a 120 mm chain at 0.45 of a 24 mm lens's view is 25 cm away.) */
  minSpan: { enter: 0.40, keep: 0.33 } as Limits,
  /** Clear frame around the chain, as a fraction of the shorter side. A pin at
   * the edge may be cut off in the still, whose field of view can differ (S3). */
  minMargin: { enter: 0.04, keep: 0.025 } as Limits,
  /** Motion in pitches per second. What matters is the smear during the still's
   * exposure, on the chain: S5 found 4 px of blur at 1080p (0.5 mm at its
   * 7.8 px/mm) costs nothing measurable and 8 px (1.0 mm) costs (p95 0.061 ->
   * 0.092 mm). With a 10 mm pitch at 1/30 s, 1 pitch/s smears 0.33 mm and 1.5
   * pitch/s 0.5 mm. A phone held still turns 0.5-3°/s (S5), which at 25 cm is
   * 0.2-1.3 pitch/s: the old 0.3 would have stopped most steady hands. */
  maxMotion: { enter: 1.0, keep: 1.5 } as Limits,
  /** Median edge sharpness of the pins (blobs.ts), 0-1. On the 4K renders,
   * blur that costs nothing (S5: up to 4 px at 1080p) reads 0.79 or more in 90%
   * of scenes (median 0.96); blur that costs (8 px at 1080p) reads a median of
   * 0.67 and 0.73 at the 90th percentile. The 1080p renders, shrunk by 3, read
   * a little lower: 0.74 and 0.65 / 0.73. Enter at 0.65, keep to 0.55: what is
   * flagged is blur that costs. (Was 0.35 / 0.28, which only a frame blurred
   * far past 8 px reached.) Defocus was not rendered; it reads the same way. */
  minSharpness: { enter: 0.65, keep: 0.55 } as Limits,
};

/** Shown in this order: fix the first before worrying about the rest. */
const PRIORITY: Problem[] = ["no-chain", "pins-missing", "at-edge", "too-far", "tilted", "moving", "blurred"];

export class Coach {
  private goodSince: number | null = null;
  private lastT: number | null = null;
  private since = new Map<Problem, number>();        // when each current problem began
  private done = false;
  private state: CoachState = blank(0);

  get current(): CoachState {
    return this.state;
  }

  /** After the still has been dealt with (measured, or it failed): coach again. */
  reset(): void {
    this.done = false;
    this.goodSince = null;
    this.lastT = null;
    this.since.clear();
    this.state = blank(this.state.pinsExpected);
  }

  update(r: FrameReading): CoachState {
    const base = { pinsFound: r.chain.length, pinsExpected: r.pinsExpected };
    if (this.done) return (this.state = { ...this.state, ...base, phase: "captured" });

    const holding = this.goodSince !== null;
    const found = check(r, holding);
    const stale = this.lastT !== null && r.t - this.lastT > COACH.STALE_MS;
    this.lastT = r.t;

    // Track how long each problem has lasted.
    for (const p of [...this.since.keys()]) if (!found.includes(p)) this.since.delete(p);
    for (const p of found) if (!this.since.has(p)) this.since.set(p, r.t);
    const shown = PRIORITY.filter((p) => found.includes(p) && r.t - this.since.get(p)! >= COACH.PROBLEM_MS);

    const straighter = r.tiltDeg !== null && r.tiltDeg > COACH.straighterDeg;
    if (found.length > 0 || stale) {
      this.goodSince = found.length > 0 ? null : r.t;
      const searching = found.includes("no-chain");
      return (this.state = {
        ...base, phase: searching ? "searching" : found.length ? "adjusting" : "steady",
        // While searching, the only thing to say is "find the chain".
        problems: searching ? ["no-chain"] : shown, hold: 0, straighter,
      });
    }
    if (this.goodSince === null) this.goodSince = r.t;
    const hold = Math.min(1, (r.t - this.goodSince) / COACH.HOLD_MS);
    if (hold >= 1) {
      this.done = true;
      return (this.state = { ...base, phase: "capture", problems: [], hold: 1, straighter });
    }
    return (this.state = { ...base, phase: "steady", problems: [], hold, straighter });
  }
}

/** The problems of one reading. `holding` applies the looser `keep` limits. */
export function check(r: FrameReading, holding: boolean): Problem[] {
  const lim = (l: Limits) => (holding ? l.keep : l.enter);
  if (r.chain.length < 3) return ["no-chain"];
  const out: Problem[] = [];
  if (r.chain.length + r.missing < r.pinsExpected || r.missing > 0) out.push("pins-missing");
  const { span, margin } = extent(r);
  if (margin < lim(COACH.minMargin)) out.push("at-edge");
  // A chain that is cut off looks small: say "at the edge" or "pins missing", not "too far".
  else if (span < lim(COACH.minSpan) && !out.includes("pins-missing")) out.push("too-far");
  if (r.tiltDeg !== null && r.tiltDeg > lim(COACH.tiltDeg)) out.push("tilted");
  if (r.motion !== null && r.motion > lim(COACH.maxMotion)) out.push("moving");
  if (r.sharpness < lim(COACH.minSharpness)) out.push("blurred");
  return out;
}

/** The chain's larger extent and its smallest distance to the frame's edge,
 * both as fractions of the frame's shorter side. */
export function extent(r: Pick<FrameReading, "chain" | "width" | "height">): { span: number; margin: number } {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of r.chain) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const short = Math.min(r.width, r.height);
  return {
    span: Math.max(x1 - x0, y1 - y0) / short,
    margin: Math.min(x0, y0, r.width - 1 - x1, r.height - 1 - y1) / short,
  };
}

function blank(pinsExpected: number): CoachState {
  return { phase: "searching", problems: [], hold: 0, pinsFound: 0, pinsExpected, straighter: false };
}
