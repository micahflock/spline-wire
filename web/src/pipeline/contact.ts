// splinewire/contact.py: from pins to the points where the target touches
// the chain. See that file for the geometry; this follows it line for line.

import { DEG, clip, norm2, type Vec2 } from "./numeric";
import type { Side } from "./types";

// Bends smaller than this (radians) count as straight.
const STRAIGHT_TOL = 0.5 * DEG;

/** Signed turn at each interior pin, counter-clockwise positive. */
export function turns(pins: readonly Vec2[]): number[] {
  const out: number[] = [];
  for (let i = 1; i + 1 < pins.length; i++) {
    const ax = pins[i]![0] - pins[i - 1]![0], ay = pins[i]![1] - pins[i - 1]![1];
    const bx = pins[i + 1]![0] - pins[i]![0], by = pins[i + 1]![1] - pins[i]![1];
    out.push(Math.atan2(ax * by - ay * bx, ax * bx + ay * by));
  }
  return out;
}

/** Which side of the pin line the object is on: +1 left, -1 right. */
export function objectSign(pins: readonly Vec2[], side: Side): number {
  let net = 0;
  for (const t of turns(pins)) net += t;       // np.sum adds in another order: only the sign is used
  const toward = net >= 0 ? 1 : -1;
  return side === "inside" ? toward : -toward;
}

/** The side, from a point picked in the photo on the object's side. */
export function signFromPhotoPoint(pinsPx: readonly Vec2[], p: Vec2): number {
  let best = -1, bestD = Infinity, cross = 0;
  for (let k = 0; k + 1 < pinsPx.length; k++) {
    const a = pinsPx[k]!, b = pinsPx[k + 1]!;
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const t = clip(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / Math.max(dx * dx + dy * dy, 1e-12), 0, 1);
    const d = norm2(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    if (d < bestD) {
      bestD = d;
      best = k;
      cross = dx * (p[1] - a[1]) - dy * (p[0] - a[0]);
    }
  }
  if (best < 0) return 1;
  return cross >= 0 ? -1 : 1;
}

export function sideName(pins: readonly Vec2[], sign: number): Side {
  return objectSign(pins, "inside") === sign ? "inside" : "outside";
}

/** Points where the target curve touches the chain, in chain order. */
export function contactPoints(
  pins: readonly Vec2[], halfWidthMm: number, sign: number, roundContact = false, gaps: Iterable<number> = [],
): Vec2[] {
  const gapSet = new Set(gaps);
  const n = pins.length;
  const left: Vec2[] = [];
  for (let i = 0; i + 1 < n; i++) {
    const dx = pins[i + 1]![0] - pins[i]![0], dy = pins[i + 1]![1] - pins[i]![1];
    const len = norm2(dx, dy);
    left.push([-dy / len, dx / len]);
  }
  const offset = sign * halfWidthMm;
  const t = turns(pins);
  const away = [false, ...t.map((v) => v * sign < -STRAIGHT_TOL), false];
  if (away.length > 2) {
    away[0] = away[1]!;
    away[away.length - 1] = away[away.length - 2]!;
  }
  let endTurn: [number, number] = [0, 0];
  if (roundContact) {
    away.fill(true);
    if (n > 2) endTurn = [t[0]! / 2, t[t.length - 1]! / 2];
  }
  const out: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const p = pins[i]!;
    if (away[i]) {
      let normal: Vec2;
      if (i === 0) normal = rotated(left[0]!, -endTurn[0]);
      else if (i === n - 1) normal = rotated(left[n - 2]!, endTurn[1]);
      else {
        const nx = left[i - 1]![0] + left[i]![0], ny = left[i - 1]![1] + left[i]![1];
        const len = norm2(nx, ny);
        normal = [nx / len, ny / len];
      }
      out.push([p[0] + offset * normal[0], p[1] + offset * normal[1]]);
    }
    if (i < n - 1 && !gapSet.has(i) && !(away[i] && away[i + 1])) {
      const q = pins[i + 1]!;
      out.push([(p[0] + q[0]) / 2 + offset * left[i]![0], (p[1] + q[1]) / 2 + offset * left[i]![1]]);
    }
  }
  return out;
}

function rotated(v: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
}
