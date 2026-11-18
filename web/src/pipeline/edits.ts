// splinewire/edits.py: a person's corrections to the detections. A point on
// a detection removes it; a point on a pin adds it, snapped to the fiducial
// under the click (snapFiducial), or the click itself where there is none to
// see. Added pins then join the chain like any other, and a clicked one is
// fitted to the pitch of its links (pinToPitch).

import type { Gray, GrayF } from "./cv";
import { snapFiducial } from "./detect";
import { PipelineError } from "./errors";
import { leastSquaresLM } from "./lm";
import { fixed, median, norm2, type Vec2 } from "./numeric";
import type { ChainSpec, Fiducial } from "./types";

export const MAX_EDITS = 300;

export interface Edits {
  add_px: Vec2[];
  remove_px: Vec2[];
}

/** A pin the person added, as it ended up in the measurement. */
export interface ManualPin {
  fiducial: number;    // index into the measurement's fiducials
  edit: number;        // index into Edits.add_px
  snapped: boolean;    // false: no fiducial was found there, the click itself is the pin
}

export function hasEdits(e: Edits | null | undefined): e is Edits {
  return !!e && (e.add_px.length > 0 || e.remove_px.length > 0);
}

/** Edits from the page's JSON, checked (Edits.from_json). size: the photo's
 * (width, height); points outside it are refused. */
export function editsFromJson(doc: unknown, size?: readonly [number, number]): Edits {
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
    throw new PipelineError("edits must be an object with add and remove lists");
  }
  const d = doc as Record<string, unknown>;
  return { add_px: points(d.add ?? [], "add", size), remove_px: points(d.remove ?? [], "remove", size) };
}

function points(raw: unknown, name: string, size?: readonly [number, number]): Vec2[] {
  if (!Array.isArray(raw) || raw.length > MAX_EDITS) {
    throw new PipelineError(`${name} must be a list of at most ${MAX_EDITS} points`);
  }
  return raw.map((p) => {
    if (!Array.isArray(p) || p.length !== 2 || !p.every((v) => typeof v === "number" || typeof v === "string")) {
      throw new PipelineError(`${name}: each point is [x, y] in pixels`);
    }
    const [x, y] = p.map(Number) as [number, number];
    if (Number.isNaN(x) || Number.isNaN(y)) throw new PipelineError(`${name}: each point is [x, y] in pixels`);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new PipelineError(`${name}: points must be finite numbers`);
    if (size && !(-1 <= x && x <= size[0] + 1 && -1 <= y && y <= size[1] + 1)) {
      throw new PipelineError(`${name}: a point lies outside the photo`);
    }
    return [Number(fixed(x, 2)), Number(fixed(y, 2))];       // Python's round(x, 2)
  });
}

/** The detections left after dropping the one under each remove point. */
export function removeDetections(fiducials: readonly Fiducial[], edits: Edits): Fiducial[] {
  const drop = new Set<number>();
  for (const p of edits.remove_px) {
    const left = fiducials.map((_, i) => i).filter((i) => !drop.has(i));    // one detection per point
    const i = detectionAt(p, left.map((k) => fiducials[k]!));
    if (i !== null) drop.add(left[i]!);
  }
  return fiducials.filter((_, i) => !drop.has(i));
}

/** `fiducials` with the added pins in place of the detections under them
 * (edits.add_pins). chain: indices of the pins that already form a chain,
 * whose nearest tell an added pin its size and look. Returns the detections
 * kept first, then the added pins in the order of edits.add_px. */
export function addPins(
  image: Gray, fiducials: readonly Fiducial[], spec: ChainSpec, edits: Edits, chain: readonly number[], grayf?: GrayF,
): [Fiducial[], ManualPin[]] {
  if (!edits.add_px.length) return [fiducials.slice(), []];
  const refs = (chain.length ? chain : fiducials.map((_, i) => i)).map((i) => fiducials[i]!);
  const fallback = refs.length ? null : fromSpacing(edits.add_px, spec);
  const made: [number, Fiducial, boolean][] = [];
  edits.add_px.forEach((p, j) => {
    const like = refs.length ? likeAt(p, refs) : fallback!;
    // A detection already under the click, else a fiducial found there, else the click.
    const near = detectionAt(p, fiducials);
    let center: Vec2, snapped: boolean;
    if (near !== null) {
      center = fiducials[near]!.center_px;
      snapped = true;
    } else {
      const found = snapFiducial(image, p, spec, like, grayf);
      [center, snapped] = found ? [found.center_px, true] : [[p[0], p[1]], false];
    }
    const pin: Fiducial = { ...like, center_px: center };
    // Two clicks on one pin make one pin.
    if (made.some(([, q]) => Math.hypot(pin.center_px[0] - q.center_px[0], pin.center_px[1] - q.center_px[1])
      < 0.5 * q.outer_axes_px[1])) return;
    made.push([j, pin, snapped]);
  });
  const kept = fiducials.filter((f) => !made.some(([, q]) =>
    Math.hypot(f.center_px[0] - q.center_px[0], f.center_px[1] - q.center_px[1])
      < 0.5 * Math.min(f.outer_axes_px[1], q.outer_axes_px[1])));
  const manual = made.map(([j, , snapped], n) => ({ fiducial: kept.length + n, edit: j, snapped }));
  return [[...kept, ...made.map(([, q]) => q)], manual];
}

/** The detection a point is on, if any. */
function detectionAt(p: Vec2, fiducials: readonly Fiducial[]): number | null {
  if (!fiducials.length) return null;
  let best = 0, bestD = Infinity;
  fiducials.forEach((f, i) => {
    const d = norm2(f.center_px[0] - p[0], f.center_px[1] - p[1]);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return bestD <= Math.max(0.6 * fiducials[best]!.outer_axes_px[0], 4.0) ? best : null;
}

/** A fiducial standing in for one at p: the nearest chain pin's shape, with
 * the median size, contrast and surround of the nearest few (edits._like). */
function likeAt(p: Vec2, refs: readonly Fiducial[]): Fiducial {
  const dist = refs.map((f) => norm2(f.center_px[0] - p[0], f.center_px[1] - p[1]));
  // np.argsort's default sort is not stable; ties between equal distances are not expected.
  const near = dist.map((d, i) => [d, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, 3);
  const group = near.map(([, i]) => refs[i]!);
  const first = group[0]!;
  return {
    center_px: [p[0], p[1]],
    outer_axes_px: [median(group.map((f) => f.outer_axes_px[0])), median(group.map((f) => f.outer_axes_px[1]))],
    inner_axes_px: first.inner_axes_px,
    angle_deg: first.angle_deg,
    residual_px: 0,
    contrast: median(group.map((f) => f.contrast)),
    polarity: first.polarity,
    surround: median(group.map((f) => f.surround)),
  };
}

/** With no detections, a fiducial as big as the spacing of the added pins
 * says (edits._from_spacing). */
function fromSpacing(pts: readonly Vec2[], spec: ChainSpec): Fiducial {
  if (pts.length < 2) {
    throw new PipelineError("no fiducial was detected to compare with; click at least two pins to say how big they are");
  }
  const nearest = pts.map((p, i) => Math.min(...pts.map((q, j) => (i === j ? Infinity : norm2(p[0] - q[0], p[1] - q[1])))));
  const d = (median(nearest) * spec.fiducial_mm) / spec.pitch_mm;
  const inner = spec.fiducial === "ring" ? (d * spec.ring_inner_mm) / spec.fiducial_mm : 0;
  return {
    center_px: [0, 0], outer_axes_px: [d, d], inner_axes_px: [inner, inner], angle_deg: 0, residual_px: 0,
    contrast: 40, polarity: 1, surround: 40,
  };
}

/** Pins in `free` moved, as little as they must, to make the links around
 * them one pitch long (edits.pin_to_pitch). */
export function pinToPitch(
  pinsMm: readonly Vec2[], links: readonly (readonly [number, number])[], free: readonly number[], pitchMm: number,
  sigmaClickMm: number, sigmaLinkMm = 0.05,
): Vec2[] {
  const pins = pinsMm.map((p) => [p[0], p[1]] as Vec2);
  if (!free.length) return pins;
  const index = new Map(free.map((k, n) => [k, n]));
  const start = free.flatMap((k) => [pins[k]![0], pins[k]![1]]);
  const touching = links.filter(([a, b]) => index.has(a) || index.has(b));
  const put = (x: readonly number[]): Vec2[] => {
    const q = pins.slice();
    free.forEach((k, n) => (q[k] = [x[2 * n]!, x[2 * n + 1]!]));
    return q;
  };
  const residuals = (x: readonly number[]): number[] => {
    const q = put(x);
    const link = touching.map(([a, b]) => (norm2(q[a]![0] - q[b]![0], q[a]![1] - q[b]![1]) - pitchMm) / sigmaLinkMm);
    return [...link, ...x.map((v, i) => (v - start[i]!) / sigmaClickMm)];
  };
  return put(leastSquaresLM(residuals, start, { maxNfev: 200 }).x);
}
