// pipeline.compare_to_truth: measured pins against known ones, after the best
// rigid fit (and after a uniform scale too), trying both directions along the
// chain. No reflections: a mirrored result is an error.

import { PipelineError } from "./errors";
import type { Vec2 } from "./numeric";

export interface TruthComparison {
  max_error_mm: number;
  rms_error_mm: number;
  max_error_scaled_mm: number;
  scale: number;
}

export function compareToTruth(pins: readonly Vec2[], truth: readonly Vec2[]): TruthComparison {
  if (pins.length !== truth.length) {
    throw new PipelineError(`measured ${pins.length} pins but truth has ${truth.length}`);
  }
  let best: { err: number[]; scaled: number[]; scale: number } | null = null;
  for (const t of [truth, truth.slice().reverse()]) {
    const err = fit(pins, t, false).errors;
    const s = fit(pins, t, true);
    if (best === null || Math.max(...err) < Math.max(...best.err)) best = { err, scaled: s.errors, scale: 1 / s.scale };
  }
  const { err, scaled, scale } = best!;
  return {
    max_error_mm: Math.max(...err),
    rms_error_mm: Math.sqrt(err.reduce((a, e) => a + e * e, 0) / err.length),
    max_error_scaled_mm: Math.max(...scaled),
    scale,
  };
}

/** Rotate (and with withScale, scale) a onto b about their centroids: the
 * rotation maximising trace, which for 2D is an angle in closed form. */
function fit(a: readonly Vec2[], b: readonly Vec2[], withScale: boolean): { errors: number[]; scale: number } {
  const n = a.length;
  const ca = mean(a), cb = mean(b);
  const a0 = a.map(([x, y]) => [x - ca[0], y - ca[1]] as Vec2);
  const b0 = b.map(([x, y]) => [x - cb[0], y - cb[1]] as Vec2);
  // M = a0^T b0
  let m00 = 0, m01 = 0, m10 = 0, m11 = 0, sa = 0;
  for (let i = 0; i < n; i++) {
    m00 += a0[i]![0] * b0[i]![0];
    m01 += a0[i]![0] * b0[i]![1];
    m10 += a0[i]![1] * b0[i]![0];
    m11 += a0[i]![1] * b0[i]![1];
    sa += a0[i]![0] ** 2 + a0[i]![1] ** 2;
  }
  // p @ R with R = [[c, s], [-s, c]] maximises c (m00 + m11) + s (m01 - m10).
  const th = Math.atan2(m01 - m10, m00 + m11);
  const c = Math.cos(th), s = Math.sin(th);
  const scale = withScale ? Math.hypot(m00 + m11, m01 - m10) / sa : 1;
  const errors = a0.map(([x, y], i) => Math.hypot(scale * (x * c - y * s) - b0[i]![0], scale * (x * s + y * c) - b0[i]![1]));
  return { errors, scale };
}

function mean(p: readonly Vec2[]): Vec2 {
  let x = 0, y = 0;
  for (const q of p) {
    x += q[0];
    y += q[1];
  }
  return [x / p.length, y / p.length];
}
