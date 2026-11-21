// What the camera view animates over the video, apart from the coach's state:
// the arrow eased to where the coach says (turning, sliding and growing
// smoothly, fading in and out instead of popping) and the lock-on rings, one
// round each pin, closing in on it as the hold fills (coach.ts HOLD_MS) until
// they meet it as the still is taken. Pure: the page calls these once a frame
// with the time, and draws what they return.

import type { Point } from "./types";

export const ANIM = {
  /** The arrow's direction, distance and length follow the coach's with this time constant, ms. */
  arrowTauMs: 160,
  /** The arrow fades in or out over this long, ms. */
  arrowFadeMs: 220,
  /** A lock-on ring starts at this many times its pin's ring, and closes to 1. */
  lockScale: 2.2,
  /** A hold that falls back (a problem broke it) is shown opening out again with this time constant, ms. */
  lockReleaseTauMs: 120,
};

/** Where the arrow is drawn: its direction (a unit vector), how far its centre
 * is from the chain's centre, its length (all in canvas pixels), and its opacity. */
export interface ArrowPose {
  dir: Point;
  along: number;
  length: number;
  alpha: number;
}

export class ArrowAnimator {
  private pose: ArrowPose | null = null;
  private t: number | null = null;

  /** The pose to draw at `now` (ms), easing toward `target` (null: the coach
   * shows none, so it fades out where it is); null once faded out. */
  update(target: Omit<ArrowPose, "alpha"> | null, now: number): ArrowPose | null {
    const dt = this.t === null ? 0 : Math.max(0, now - this.t);
    this.t = now;
    if (!this.pose) {
      if (!target) return null;
      this.pose = { ...target, alpha: 0 };                     // appears where it should be, fading in
    }
    const p = this.pose;
    if (target) {
      const k = 1 - Math.exp(-dt / ANIM.arrowTauMs);
      // Turn the short way round.
      const from = Math.atan2(p.dir[1], p.dir[0]), to = Math.atan2(target.dir[1], target.dir[0]);
      let turn = to - from;
      turn -= 2 * Math.PI * Math.round(turn / (2 * Math.PI));
      const a = from + k * turn;
      p.dir = [Math.cos(a), Math.sin(a)];
      p.along += k * (target.along - p.along);
      p.length += k * (target.length - p.length);
    }
    const fade = ANIM.arrowFadeMs > 0 ? dt / ANIM.arrowFadeMs : 1;
    p.alpha = Math.min(1, Math.max(0, p.alpha + (target ? fade : -fade)));
    if (!target && p.alpha <= 0) {
      this.pose = null;
      return null;
    }
    return { ...p, dir: [p.dir[0], p.dir[1]] };
  }

  reset(): void {
    this.pose = null;
    this.t = null;
  }
}

/** The hold as the lock-on rings show it: following the coach's as it fills
 * (it fills steadily), easing back out when a hold is broken rather than jumping. */
export class HoldAnimator {
  private shown = 0;
  private t: number | null = null;

  update(hold: number, now: number): number {
    const dt = this.t === null ? 0 : Math.max(0, now - this.t);
    this.t = now;
    const h = Math.min(1, Math.max(0, hold));
    this.shown = h >= this.shown ? h : this.shown + (h - this.shown) * (1 - Math.exp(-dt / ANIM.lockReleaseTauMs));
    if (this.shown < 0.002) this.shown = 0;
    return this.shown;
  }

  reset(): void {
    this.shown = 0;
    this.t = null;
  }
}

/** A lock-on ring's radius for a pin's ring of radius `r` at `hold` (0 to 1):
 * ANIM.lockScale × r at the start, closing fast and then settling, r at 1. */
export function lockRadius(r: number, hold: number): number {
  const h = Math.min(1, Math.max(0, hold));
  return r * (1 + (ANIM.lockScale - 1) * (1 - h) ** 1.5);
}

/** A lock-on ring's opacity at `hold`: none at 0, plain at once while wide (it must be seen), firm as it closes. */
export function lockAlpha(hold: number): number {
  const h = Math.min(1, Math.max(0, hold));
  return Math.min(1, 10 * h) * (0.55 + 0.4 * h);
}
