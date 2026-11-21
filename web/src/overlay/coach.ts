// The coach: from a stream of frame readings to what the person is told and
// the moment the still is taken. A pure state machine: feed it readings, read
// its state. No timers, no DOM.
//
// Entering "steady" needs every check to pass; staying there tolerates a
// little more (the `keep` thresholds), so a reading on the borderline does not
// make the hold flicker. A problem must persist for PROBLEM_MS before it is
// shown, so one bad frame does not flash a message, and before it breaks a
// hold, so one bad frame does not turn the trace from green to yellow and back
// (the owner's iPhone, build 24: flickering while held still, once the live
// view read ~40 frames a second). The still is taken only on a reading with no
// problem at all.

import type { CoachState, FrameReading, Point, Problem } from "./types";

export interface Limits {
  enter: number;
  keep: number;
}

/** The thresholds, in one place. Calibrated (2026-11-17) on S5's renders
 * (experiments/video_frame_accuracy.py, its 167 scenes) drawn small and gray
 * as the camera view does: each frame area-averaged to a 640 px work frame and
 * run through blobs.ts in Node. "1080p" and "4K" below are those renders; the
 * phone streams 3024×4032 (S3), which shrinks to its work frame by 6.3, like
 * the 4K renders (6). Not yet checked against a real phone's video. The
 * live view now draws the region round the chain finer than that (region.ts);
 * sharpness is still read in the full-frame work frame's pixels (blobs.Window
 * refScale), so these limits keep their meaning. */
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
  /** A problem is shown, and breaks a hold, once it has lasted this long. */
  PROBLEM_MS: 250,
  /** Motion is measured against the newest reading at least this much older
   * (session.ts): pin centres jitter by a fraction of a pixel from frame to
   * frame, and divided by the ~25 ms between readings at 40 frames a second
   * that read as motion. Over 200 ms the same jitter counts an eighth as much. */
  motionBaseMs: 200,
  /** Camera tilt from straight-on, degrees. In simulation a 2% focal error costs
   * up to 0.05 mm at 25° and 0.11 mm at 40° (docs/open-items.md, item 1): block the shot
   * above 25°, and nudge toward straighter above 12°. */
  tiltDeg: { enter: 25, keep: 30 } as Limits,
  straighterDeg: 12,
  /** The arrow showing which way to move the phone (CoachState.arrow). Its
   * direction is a running average of the readings' unit vectors, each
   * counting `arrowWeight`; the average's length is how well they agree (1 all
   * the same, near 0 scattered). It shows above `arrowDeg` of tilt with the
   * readings agreeing to `arrowAgree`, and stays down to the looser limits, so
   * it does not blink on the borderline.
   * One reading's direction is sound on a curved chain from about 12 degrees, on
   * a near-straight one from about 20: a straight line says little about a tilt
   * round itself (experiments/live_view/tilt_direction.py, pin centres off by
   * 0.014 pitch; without noise it is right to 2 degrees at any tilt). Through
   * these limits, 8 readings a pose, from 12 degrees of tilt: an arrow in
   * 54-88% of readings (never on the first), more than 45 degrees off in at
   * most 3% of those (10% on the near-straight chain with twice the noise),
   * typically 1-8 degrees off. Below 12 degrees it shows in 5-38% of
   * readings and may point anywhere near straight-on, where the tilt costs
   * nothing (2026-11-19). */
  arrowDeg: { enter: 12, keep: 9 } as Limits,
  arrowAgree: { enter: 0.6, keep: 0.4 } as Limits,
  /** A reading's weight in the arrow's average, per 100 ms since the last
   * (time-based since 2026-11-21: at ~40 readings a second a fixed weight per
   * reading turned it four times as fast as at the ~10 it was set for). */
  arrowWeight: 0.4,
  /** The chain's extent (its pins') as a fraction of the frame's shorter side.
   * Recalibrated 2026-11-19 for the snap-fit chain, now that the live view
   * reads the region round the chain at its own scale (region.ts) rather than
   * the whole frame at 800 px (open-items.md, live view resolution). With
   * the pins' extent plus a rim as "span": the live view finds 13 of 13 rims
   * from a cold start down to 0.22 and tracks them to ~0.18
   * (experiments/live_view/regions.ts; the 800 px frame lost them below 0.40),
   * and the still measures its worst pin to 0.003-0.004 mm from 0.40 to 0.26,
   * 0.007 at 0.22, and loses a pin at 0.18 (still_by_span.py, clean renders).
   * So the shot needs 0.26 and keeps going to 0.22. The experiments' span is
   * the pins' greatest distance plus a rim; this one is their bounding box,
   * never more: a chain at 0.22 here spans at least 0.23 there (more when it
   * lies diagonally). (Was 0.40 / 0.33, the 800 px frame's edge; before that 0.45 / 0.38.) The dot chain's dots are drawn at 12 work pixels the same
   * way, as large as at 0.55 on the old 640 px frame. */
  minSpan: { enter: 0.26, keep: 0.22 } as Limits,
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
  private arrow: ArrowTrack = NO_ARROW;
  /** Holds broken, by the problem that broke them (for the camera view's diagnostics). */
  readonly breaks: Partial<Record<Problem, number>> = {};

  get current(): CoachState {
    return this.state;
  }

  /** After the still has been dealt with (measured, or it failed): coach again. */
  reset(): void {
    this.done = false;
    this.goodSince = null;
    this.lastT = null;
    this.since.clear();
    this.arrow = NO_ARROW;
    this.state = blank(this.state.pinsExpected);
  }

  update(r: FrameReading): CoachState {
    if (this.done) return (this.state = { ...this.state, pinsFound: r.chain.length, pinsExpected: r.pinsExpected, phase: "captured", arrow: null });
    this.arrow = nextArrow(this.arrow, r);
    const base = { pinsFound: r.chain.length, pinsExpected: r.pinsExpected, arrow: this.arrow.shown };

    const holding = this.goodSince !== null;
    const found = check(r, holding);
    const stale = this.lastT !== null && r.t - this.lastT > COACH.STALE_MS;
    this.lastT = r.t;

    // Track how long each problem has lasted.
    for (const p of [...this.since.keys()]) if (!found.includes(p)) this.since.delete(p);
    for (const p of found) if (!this.since.has(p)) this.since.set(p, r.t);
    const shown = PRIORITY.filter((p) => found.includes(p) && r.t - this.since.get(p)! >= COACH.PROBLEM_MS);

    const straighter = r.tiltDeg !== null && r.tiltDeg > COACH.straighterDeg;
    // A hold rides out problems that have not lasted: it stays steady, and
    // goes on counting, but does not end on such a reading.
    if (holding && !stale && found.length > 0 && shown.length === 0) {
      const hold = Math.min(0.99, (r.t - this.goodSince!) / COACH.HOLD_MS);
      return (this.state = { ...base, phase: "steady", problems: [], hold, straighter });
    }
    if (found.length > 0 || stale) {
      if (holding && shown[0]) this.breaks[shown[0]] = (this.breaks[shown[0]] ?? 0) + 1;
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
      return (this.state = { ...base, phase: "capture", problems: [], hold: 1, straighter, arrow: null });
    }
    return (this.state = { ...base, phase: "steady", problems: [], hold, straighter });
  }
}

/** The arrow's running state: the average of the readings' directions
 * (not normalised: its length is their agreement) and what is shown. */
export interface ArrowTrack {
  mean: Point;
  /** The last reading's time, ms (null: none yet). */
  t: number | null;
  shown: CoachState["arrow"];
}

export const NO_ARROW: ArrowTrack = { mean: [0, 0], t: null, shown: null };

/** The arrow after reading `r` (COACH.arrowDeg, arrowAgree, arrowWeight).
 * Gone with the chain or the tilt; a reading straight on (no direction) leaves
 * the average as it was. */
export function nextArrow(prev: ArrowTrack, r: FrameReading): ArrowTrack {
  if (r.chain.length < 3 || r.tiltDeg === null) return NO_ARROW;
  if (r.tiltDir === null) return { mean: prev.mean, t: r.t, shown: null };
  // The weight for the time since the last reading: arrowWeight per 100 ms, the first reading arrowWeight.
  const w = prev.t === null ? COACH.arrowWeight : 1 - (1 - COACH.arrowWeight) ** (Math.max(0, r.t - prev.t) / 100);
  const mean: Point = [(1 - w) * prev.mean[0] + w * r.tiltDir[0], (1 - w) * prev.mean[1] + w * r.tiltDir[1]];
  const agree = Math.hypot(mean[0], mean[1]);
  const lim = (l: Limits) => (prev.shown ? l.keep : l.enter);
  const show = r.tiltDeg > lim(COACH.arrowDeg) && agree > lim(COACH.arrowAgree);
  return { mean, t: r.t, shown: show ? { dir: [mean[0] / agree, mean[1] / agree], tiltDeg: r.tiltDeg } : null };
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
  return { phase: "searching", problems: [], hold: 0, pinsFound: 0, pinsExpected, straighter: false, arrow: null };
}
