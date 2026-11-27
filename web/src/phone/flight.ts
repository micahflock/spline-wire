// The working curve's photo flying between the card and the expanded view
// (main.ts): its rectangle on screen and the part of the photo it shows move
// together, so it leaves the card showing the card's crop and lands showing
// the editor's framing (and back). Pure: main.ts runs the frames.

import type { ViewBox } from "../editor/viewport";

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** CSS's cubic-bezier(x1, y1, x2, y2) as a function of time 0..1 (Newton's method, then bisection). */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number): (t: number) => number {
  const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
  const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
  const x = (s: number) => ((ax * s + bx) * s + cx) * s;
  const dx = (s: number) => (3 * ax * s + 2 * bx) * s + cx;
  const y = (s: number) => ((ay * s + by) * s + cy) * s;
  return (t) => {
    if (t <= 0) return 0;
    if (t >= 1) return 1;
    let s = t;
    for (let i = 0; i < 8; i++) {
      const e = x(s) - t, d = dx(s);
      if (Math.abs(e) < 1e-9) return y(s);
      if (Math.abs(d) < 1e-6) break;
      s -= e / d;
    }
    let lo = 0, hi = 1;
    s = t;
    for (let i = 0; i < 40; i++) {
      if (x(s) < t) lo = s; else hi = s;
      s = (lo + hi) / 2;
    }
    return y(s);
  };
}

/** The expanded view's easing: a gentle start (under 1% in the first 60 Hz frame,
 * so the picture never jumps off the card), half way by a quarter of the time,
 * then a long settle. (0.2, 0.8, 0.2, 1) moved 21% in the first frame: a jump on a phone. */
export const EASE = cubicBezier(0.3, 0, 0, 1);
export const FLIGHT_MS = 380;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
/** Sizes grow and shrink geometrically: a zoom looks even all the way. */
const glerp = (a: number, b: number, t: number) => (a > 0 && b > 0 ? a * (b / a) ** t : lerp(a, b, t));

export function lerpRect(a: Rect, b: Rect, t: number): Rect {
  return { left: lerp(a.left, b.left, t), top: lerp(a.top, b.top, t), width: lerp(a.width, b.width, t), height: lerp(a.height, b.height, t) };
}

/** The part of the photo shown at `t`: the centre moves straight, the size by a steady zoom. */
export function lerpBox(a: ViewBox, b: ViewBox, t: number): ViewBox {
  const w = glerp(a.w, b.w, t), h = glerp(a.h, b.h, t);
  const cx = lerp(a.x + a.w / 2, b.x + b.w / 2, t), cy = lerp(a.y + a.h / 2, b.y + b.h / 2, t);
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}
