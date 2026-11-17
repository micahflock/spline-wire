// splinewire/rectify.py: place the pins on the chain plane from the pitch
// alone. Each pin's viewing ray meets the plane {X : m . X = 1} at
// X = ray / (m . ray); m (tilt and distance) is chosen so linked pins are one
// pitch apart, by least squares (lm.ts, as SciPy's least_squares "lm").
// Without a focal length it is a fourth unknown. Radial distortion
// (estimate_distortion) is not ported: the pipeline never asks for it.

import { PipelineError } from "./errors";
import { leastSquaresLM, NotFinite } from "./lm";
import { clip, median, norm2, type Vec2 } from "./numeric";
import type { Rectification } from "./types";

/** Fewest links rectifyChain accepts: two more than its unknowns. */
export function minLinks(focalKnown = true): number {
  return 3 + (focalKnown ? 0 : 1) + 2;
}

export function rectifyChain(
  pointsPx: readonly Vec2[],
  links: readonly (readonly [number, number])[],
  pitchMm: number,
  imageSize: readonly [number, number],
  focalPx: number | null = null,
): Rectification {
  const solveF = focalPx === null;
  const need = minLinks(!solveF);
  if (links.length < need) throw new PipelineError(`need at least ${need} links to rectify, got ${links.length}`);
  if (focalPx !== null && !(Number.isFinite(focalPx) && focalPx > 0)) {
    throw new PipelineError(`the focal length must be a positive number of pixels, got ${focalPx}`);
  }
  const [w, h] = imageSize;
  const diag = Math.hypot(w, h);
  const guesses = focalPx !== null ? [focalPx] : [0.6 * diag, 0.85 * diag, 1.2 * diag];
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  const pxPitch = median(links.map(([a, b]) =>
    norm2(pointsPx[b]![0] - pointsPx[a]![0], pointsPx[b]![1] - pointsPx[a]![1])));

  const focalOf = (p: readonly number[], f0: number) => (solveF ? focal(f0, p[3]!) : f0);
  const lift = (m: readonly number[], f: number): [number, number, number][] =>
    pointsPx.map(([u, v]) => {
      const x = (u - cx) / f, y = (v - cy) / f;
      const s = x * m[0]! + y * m[1]! + m[2]!;
      return [x / s, y / s, 1 / s];
    });

  let best: { x: number[]; cost: number; f0: number; d0: number } | null = null;
  for (const f0 of guesses) {
    const d0 = (f0 * pitchMm) / pxPitch;        // the distance if the photo were straight on
    const residuals = (p: readonly number[]): number[] => {
      const X = lift([p[0]! / d0, p[1]! / d0, p[2]! / d0], focalOf(p, f0));
      return links.map(([a, b]) => dist3(X[b]!, X[a]!) - pitchMm);
    };
    for (const q0 of planeStarts()) {
      const p0 = solveF ? [...q0, 0] : [...q0];
      let sol;
      try {
        sol = leastSquaresLM(residuals, p0, { maxNfev: 400 });
      } catch (err) {
        if (err instanceof NotFinite) continue;
        throw err;
      }
      if (best === null || sol.cost < best.cost) best = { x: sol.x, cost: sol.cost, f0, d0 };
    }
  }
  if (best === null) {
    throw new PipelineError("could not fit a plane through the pins; check the detections and the focal length");
  }
  let m = [best.x[0]! / best.d0, best.x[1]! / best.d0, best.x[2]! / best.d0];
  const f = focalOf(best.x, best.f0);
  let X = lift(m, f);
  if (X.some((p) => p[2] <= 0)) {           // m and -m fit alike: keep the plane in front of the camera
    m = m.map((v) => -v);
    X = X.map(([a, b, c]) => [-a, -b, -c]);
  }
  const mn = Math.sqrt(m[0]! * m[0]! + m[1]! * m[1]! + m[2]! * m[2]!);
  const n = m.map((v) => v / mn);           // plane normal, away from the camera
  // Image +x projected into the plane, and image "up" (-n x e1).
  let e1 = [1 - n[0]! * n[0]!, -n[0]! * n[1]!, -n[0]! * n[2]!];
  const e1n = Math.sqrt(e1[0]! ** 2 + e1[1]! ** 2 + e1[2]! ** 2);
  e1 = e1.map((v) => v / e1n);
  const e2 = cross([-n[0]!, -n[1]!, -n[2]!], e1);
  const raw = X.map((p) => [dot(p, e1), dot(p, e2)] as const);
  const pins: Vec2[] = raw.map(([a, b]) => [a - raw[0]![0], b - raw[0]![1]]);
  const res = links.map(([a, b]) => norm2(pins[b]![0] - pins[a]![0], pins[b]![1] - pins[a]![1]) - pitchMm);
  let sq = 0, worst = 0;
  for (const r of res) {
    sq += r * r;
    worst = Math.max(worst, Math.abs(r));
  }
  return {
    pins_mm: pins,
    focal_px: f,
    focal_estimated: focalPx === null,
    tilt_deg: (Math.acos(Math.min(1, Math.abs(n[2]!))) * 180) / Math.PI,
    residual_rms_mm: Math.sqrt(sq / res.length),
    residual_max_mm: worst,
    link_residuals_mm: res,
    depth_mm: X.map((p) => p[2]),
  };
}

// Keep the focal-length search within a plausible range for phone cameras.
function focal(f0: number, logRatio: number): number {
  return f0 * Math.exp(clip(logRatio, -1, 1));
}

/** Straight on, then tilts of 20, 40 and 60 degrees in 8 directions: from a
 * single start the solver can settle in a wrong minimum for arcs. */
function planeStarts(): number[][] {
  const starts = [[0, 0, 1]];
  for (const tilt of [20, 40, 60]) {
    const t = (tilt * Math.PI) / 180;
    for (let k = 0; k < 8; k++) {
      const a = (k * Math.PI) / 4;
      starts.push([Math.sin(t) * Math.cos(a), Math.sin(t) * Math.sin(a), Math.cos(t)]);
    }
  }
  return starts;
}

function dist3(a: readonly number[], b: readonly number[]): number {
  const dx = a[0]! - b[0]!, dy = a[1]! - b[1]!, dz = a[2]! - b[2]!;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function dot(a: readonly number[], b: readonly number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

function cross(a: readonly number[], b: readonly number[]): number[] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
}
