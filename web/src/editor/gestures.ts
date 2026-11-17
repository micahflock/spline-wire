// Pointer events to editor intents: tap, drag to pan, pinch to zoom, wheel.
//
// One state machine for mouse, pen and touch (Pointer Events). It knows
// nothing of the DOM: the page forwards each pointer event's id, type and
// position (CSS px relative to the editor element) and acts on what comes
// back. The rules:
//   - A tap is a press and release that moved less than the slop (bigger for
//     touch, whose contact point wanders) and did not last long enough to be
//     a deliberate hold. It reports the point where the press began.
//   - A drag with one pointer pans.
//   - A second finger turns the gesture into a pinch, and the gesture can no
//     longer end in a tap, even after a finger lifts: lifting the first of two
//     fingers must not add a pin.
//   - After a pinch, the remaining finger pans from where it is, without a jump.

import { pan, pinch, type Pt, type Size, type ViewBox, type ZoomLimits } from "./viewport";

export const SLOP_PX = { mouse: 4, pen: 6, touch: 10 };
/** A press held longer than this is not a tap (the person was reading the screen). */
export const TAP_MAX_MS = 600;

export type PointerKind = keyof typeof SLOP_PX;

export interface PointerInput {
  id: number;
  kind: PointerKind;
  /** CSS px relative to the editor element. */
  x: number;
  y: number;
  /** ms */
  t: number;
}

export type Intent =
  | { type: "none" }
  | { type: "view"; box: ViewBox }
  | { type: "tap"; at: Pt; kind: PointerKind };

interface Down {
  start: Pt;
  now: Pt;
}

export class Gestures {
  private readonly down = new Map<number, Down>();
  private startBox: ViewBox | null = null;
  private pinchFrom: [Pt, Pt] | null = null;
  private tapCandidate: { id: number; at: Pt; t: number; kind: PointerKind } | null = null;
  private moved = false;

  /** `view()` is the current view box; `el()` the element's size; `limits()` the zoom range. */
  constructor(
    private readonly view: () => ViewBox,
    private readonly el: () => Size,
    private readonly limits: () => ZoomLimits,
  ) {}

  get active(): boolean {
    return this.down.size > 0;
  }

  pointerDown(p: PointerInput): Intent {
    this.down.set(p.id, { start: { x: p.x, y: p.y }, now: { x: p.x, y: p.y } });
    if (this.down.size === 1) {
      this.tapCandidate = { id: p.id, at: { x: p.x, y: p.y }, t: p.t, kind: p.kind };
      this.moved = false;
    } else {
      this.tapCandidate = null;           // two fingers never tap
    }
    this.restart();
    return { type: "none" };
  }

  pointerMove(p: PointerInput): Intent {
    const d = this.down.get(p.id);
    if (!d) return { type: "none" };
    d.now = { x: p.x, y: p.y };
    if (!this.moved) {
      const tc = this.tapCandidate;
      const slop = SLOP_PX[p.kind];
      if (tc && Math.hypot(p.x - tc.at.x, p.y - tc.at.y) <= slop) return { type: "none" };
      this.moved = true;
      this.tapCandidate = null;
      // Pan from here, not from the press, so the view does not jump by the slop.
      this.restart();
      return { type: "none" };
    }
    return { type: "view", box: this.current() };
  }

  pointerUp(p: PointerInput): Intent {
    const tc = this.tapCandidate;
    const was = this.down.delete(p.id);
    if (!was) return { type: "none" };
    if (this.down.size > 0) {
      this.restart();                     // the other finger pans on from where it is
      return { type: "none" };
    }
    this.tapCandidate = null;
    this.startBox = null;
    if (tc && tc.id === p.id && !this.moved && p.t - tc.t <= TAP_MAX_MS) {
      return { type: "tap", at: tc.at, kind: tc.kind };
    }
    return { type: "none" };
  }

  /** pointercancel (the browser took the gesture over, a call came in): forget everything. */
  cancel(): void {
    this.down.clear();
    this.tapCandidate = null;
    this.startBox = null;
    this.pinchFrom = null;
  }

  private restart(): void {
    this.startBox = this.view();
    const pts = [...this.down.values()];
    for (const d of pts) d.start = { ...d.now };
    this.pinchFrom = pts.length >= 2 ? [pts[0]!.now, pts[1]!.now] : null;
  }

  private current(): ViewBox {
    const start = this.startBox ?? this.view();
    const pts = [...this.down.values()];
    if (pts.length >= 2 && this.pinchFrom) {
      return pinch(start, this.el(), this.pinchFrom, [pts[0]!.now, pts[1]!.now], this.limits());
    }
    const d = pts[0]!;
    return pan(start, this.el(), d.now.x - d.start.x, d.now.y - d.start.y);
  }
}

/** A wheel or trackpad event's zoom factor (>1 zooms out). Trackpad pinches
 * arrive as wheel events with ctrlKey and small deltas; scale them up so a
 * pinch feels like a pinch. deltaMode 1 is lines, 2 pages. */
export function wheelFactor(deltaY: number, deltaMode: number, ctrlKey: boolean): number {
  const px = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY;
  return Math.exp(px * (ctrlKey ? 0.01 : 0.0015));
}
