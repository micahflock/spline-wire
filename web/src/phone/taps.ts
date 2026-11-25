// Taps on the expanded photo (for UI review/phone/BRIEF.md): a double tap fits
// the chain to the view, so a single tap (remove a pin, add one) waits out the
// double-tap window before it edits. The editor (src/photo/view.ts) is shared
// with the desktop page, which has no double tap, so this sits in front of its
// tap callback on the phone only.

import type { Pt } from "../editor/viewport";

export const DOUBLE_TAP_MS = 300;

export interface Timer {
  set(fn: () => void, ms: number): unknown;
  clear(handle: unknown): void;
}

const realTimer: Timer = { set: (fn, ms) => setTimeout(fn, ms), clear: (h) => clearTimeout(h as ReturnType<typeof setTimeout>) };

export class TapFilter<K> {
  private last: { at: Pt; slack: number; kind: K; handle: unknown } | null = null;

  constructor(
    private readonly single: (at: Pt, slack: number, kind: K) => void,
    private readonly double: () => void,
    private readonly timer: Timer = realTimer,
    private readonly windowMs = DOUBLE_TAP_MS,
  ) {}

  /** A tap at `at` (photo px) with the pointer's target size `slack` (photo px). A second tap
   * within the window and near the first is a double tap; any other tap ends the wait at once. */
  tap(at: Pt, slack: number, kind: K): void {
    const prev = this.last;
    if (prev) {
      this.timer.clear(prev.handle);
      this.last = null;
      if (Math.hypot(at.x - prev.at.x, at.y - prev.at.y) <= 2 * Math.max(slack, prev.slack)) {
        this.double();
        return;
      }
      this.single(prev.at, prev.slack, prev.kind);
    }
    const handle = this.timer.set(() => {
      if (this.last?.handle !== handle) return;
      this.last = null;
      this.single(at, slack, kind);
    }, this.windowMs);
    this.last = { at, slack, kind, handle };
  }

  /** Forget a waiting tap (the view closed, another photo). */
  cancel(): void {
    if (this.last) this.timer.clear(this.last.handle);
    this.last = null;
  }
}
