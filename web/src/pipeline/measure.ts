// splinewire/pipeline.py measure(), from the detections on: order the pins
// (each polarity on its own), drop pins that turn out not to belong and solve
// again, deskew, warn, and offset to the contact points. Hand edits
// (splinewire/edits.py) are not ported yet: they need detection to snap a
// click to a dot.

import { contactPoints, objectSign, sideName, signFromPhotoPoint } from "./contact";
import { PipelineError } from "./errors";
import { fixed, median, norm2, type Vec2 } from "./numeric";
import { orderChain } from "./order";
import { minLinks, rectifyChain } from "./rectify";
import type { ChainOrder, ChainSpec, Fiducial, Rectification, Side } from "./types";

const MISFIT_ROUNDS = 4;
// More detections than this is a busy background (pipeline._BUSY).
const BUSY = 600;

export interface Measurement {
  fiducials: Fiducial[];
  order: ChainOrder;
  rectification: Rectification;
  object_side: Side;
  object_sign: number;
  side_px: Vec2 | null;
  contacts_mm: Vec2[];
  warnings: string[];
  notes: string[];
  pins_px: Vec2[];
}

/** Measure the chain from a photo's detections. side: "inside" or "outside"
 * of the bend, or a photo point on the object's side. */
export function measureDetections(
  fiducials: readonly Fiducial[],
  imageSize: readonly [number, number],
  spec: ChainSpec,
  focalPx: number | null,
  side: Side | Vec2 = "inside",
): Measurement {
  const warnings: string[] = [];
  const notes: string[] = [];
  const fids = fiducials.slice();
  if (fids.length < 3) throw new PipelineError(`found ${fids.length} fiducials; need the whole chain in view`);
  const busy = fids.length > BUSY
    ? `the background has ${fids.length} dot-like spots (fabric, terrazzo, print?) that can pass `
      + "for pins: lay the chain on a plain surface"
    : "";
  const centers = fids.map((f) => f.center_px);
  const excluded = new Set<number>();
  let order!: ChainOrder;
  let rect!: Rectification;
  for (let attempt = 0; attempt <= MISFIT_ROUNDS; attempt++) {
    order = orderFiducials(fids, spec, excluded);
    const links = order.links;              // no pins placed by a click yet: every link is measured
    const need = minLinks(focalPx !== null);
    if (links.length < need) {
      throw new PipelineError(
        `found ${order.indices.length} pins in a row, joined by ${links.length} links; taking the `
        + `perspective out needs at least ${need} links (${need + 1} pins)`
        + (focalPx !== null ? "" : " when the focal length is unknown")
        + ". Is the whole chain in the photo?" + (busy ? ` Also, ${busy}.` : ""));
    }
    rect = rectifyChain(order.indices.map((i) => centers[i]!), links, spec.pitch_mm, imageSize, focalPx);
    if (attempt === MISFIT_ROUNDS) break;
    const bad = misfits(fids, order, rect, spec);
    if (bad.size === 0) break;
    for (const b of bad) excluded.add(b);
  }
  const nOnChain = order.indices.length;
  const skipped = order.rejected.length;
  if (skipped) notes.push(`ignored ${skipped} look-alike detection(s) not on the chain`);
  if (order.gaps.length) warnings.push(`${order.gaps.length} pin(s) not detected; the curve bridges those gaps`);
  const expected = spec.n_pins - order.gaps.length;
  const beyond = pastTheEnds(fids, order, rect.pins_mm, spec, new Set(order.rejected));
  if (beyond && nOnChain === expected) {
    warnings.push(
      `the chain seems to go on past its end (${beyond} more pin(s) about one pitch on); `
      + `is the pin count (${spec.n_pins}) right?`);
  }
  if (nOnChain !== expected) {
    warnings.push(
      `found ${nOnChain} pins on the chain, which is set to have ${spec.n_pins}`
      + (order.gaps.length ? ` (${order.gaps.length} more missed in between)` : "")
      + "; is the whole chain in the photo, free of glare (a lamp's reflection on the links), "
      + "and is the pin count right?");
  }
  if (focalPx === null) {
    warnings.push(
      "no focal length (none in the photo's EXIF, and none set); estimating it from the chain, "
      + "which is unreliable for nearly straight chains");
  }
  const offPitch = rect.residual_max_mm > 0.05 * spec.pitch_mm;
  if (offPitch) {
    warnings.push(
      `link lengths deviate from the pitch by up to ${fixed(rect.residual_max_mm, 2)} mm; `
      + "check the detections in the preview image");
  }
  if (busy && (nOnChain !== expected || order.gaps.length || offPitch)) warnings.push(busy);

  const pinsPx = order.indices.map((i) => centers[i]!);
  let sign: number;
  let sidePx: Vec2 | null = null;
  if (typeof side === "string") {
    sign = objectSign(rect.pins_mm, side);
  } else {
    sidePx = [side[0], side[1]];
    sign = signFromPhotoPoint(pinsPx, sidePx);
  }
  const contacts = contactPoints(rect.pins_mm, spec.half_width_mm, sign, spec.contact === "round", order.gaps);
  return {
    fiducials: fids, order, rectification: rect, object_side: sideName(rect.pins_mm, sign), object_sign: sign,
    side_px: sidePx, contacts_mm: contacts, warnings, notes, pins_px: pinsPx,
  };
}

/** Chain order over the fiducials, each polarity on its own; the longer chain
 * wins, light-on-dark on a tie (pipeline._order). */
export function orderFiducials(fids: readonly Fiducial[], spec: ChainSpec, excluded: ReadonlySet<number>): ChainOrder {
  let best: ChainOrder | null = null;
  for (const polarity of [1, -1]) {
    const idx: number[] = [];
    fids.forEach((f, i) => {
      if (f.polarity === polarity && !excluded.has(i)) idx.push(i);
    });
    if (idx.length < 3) continue;
    const o = orderChain(idx.map((i) => fids[i]!.center_px), {
      axesPx: idx.map((i) => fids[i]!.outer_axes_px),
      pitchPerDiameter: spec.pitch_mm / spec.fiducial_mm,
      nPins: spec.n_pins,
      maxTurnDeg: spec.max_bend_deg,
      strength: idx.map((i) => fids[i]!.contrast),
    });
    const indices = o.indices.map((i) => idx[i]!);
    const on = new Set(indices);
    const rejected: number[] = [];
    for (let i = 0; i < fids.length; i++) if (!on.has(i)) rejected.push(i);
    const mapped: ChainOrder = { indices, links: o.links, gaps: o.gaps, rejected };
    if (best === null || mapped.indices.length > best.indices.length) best = mapped;
  }
  if (best === null) throw new PipelineError(`found ${fids.length} fiducials; need the whole chain in view`);
  return best;
}

/** Pins that do not belong on the chain, judged after deskewing (pipeline._misfits). */
function misfits(fids: readonly Fiducial[], order: ChainOrder, rect: Rectification, spec: ChainSpec): Set<number> {
  const idx = order.indices;
  const bad = new Set<number>();
  if (idx.length < 6) return bad;
  const sizeMm = idx.map((i, k) => (fids[i]!.outer_axes_px[0] * rect.depth_mm[k]!) / rect.focal_px);
  for (let k = 0; k < idx.length; k++) {
    const nb: number[] = [];
    for (let j = Math.max(0, k - 2); j < Math.min(idx.length, k + 3); j++) if (j !== k) nb.push(sizeMm[j]!);
    if (Math.abs(sizeMm[k]! / median(nb) - 1.0) > 0.12) bad.add(idx[k]!);
  }
  if (bad.size === 0 && idx.length + order.gaps.length > spec.n_pins) {
    // More pins than the chain has: drop the end that stands out more by what
    // surrounds it, and by size, compared with its neighbours only.
    const surround = idx.map((i) => fids[i]!.surround);
    const spread = surroundSpread(surround);
    const oddness = (k: number): number => {
      const nb = k === 0 ? [1, 2] : [idx.length - 2, idx.length - 3];
      return Math.abs(surround[k]! - median(nb.map((j) => surround[j]!))) / spread
        + Math.abs(sizeMm[k]! / median(nb.map((j) => sizeMm[j]!)) - 1.0) / 0.03;
    };
    const ends = [0, idx.length - 1];
    // max() keeps the first of equals, as Python's.
    const worst = oddness(ends[1]!) > oddness(ends[0]!) ? ends[1]! : ends[0]!;
    bad.add(idx[worst]!);
    return bad;
  }
  const res = rect.link_residuals_mm;
  const links = order.links;
  if (links.length >= 6) {
    let worst = 0;
    for (const r of res) worst = Math.max(worst, Math.abs(r));
    for (const [endLink, endPin] of [[0, 0], [links.length - 1, idx.length - 1]] as const) {
      const [a, b] = links[endLink]!;
      if (endPin !== a && endPin !== b) continue;      // the end is across a gap: leave it
      if (Math.abs(res[endLink]!) > 0.08 * spec.pitch_mm && Math.abs(res[endLink]!) >= 0.999 * worst) {
        bad.add(idx[endPin]!);
      }
    }
  }
  return bad;
}

/** How much what surrounds a pin varies along the chain: 1.4826 times the
 * median absolute step, at least 5 gray levels. */
function surroundSpread(surround: readonly number[]): number {
  const steps: number[] = [];
  for (let k = 1; k < surround.length; k++) steps.push(Math.abs(surround[k]! - surround[k - 1]!));
  return Math.max(5.0, 1.4826 * median(steps));
}

/** How many chain ends have a detection left off the chain just past them,
 * looking like the next pin (pipeline._past_the_ends). */
function pastTheEnds(
  fids: readonly Fiducial[], order: ChainOrder, pinsMm: readonly Vec2[], spec: ChainSpec, candidates: ReadonlySet<number>,
): number {
  const idx = order.indices;
  if (idx.length < 3 || candidates.size === 0) return 0;
  const spread = surroundSpread(idx.map((i) => fids[i]!.surround));
  let found = 0;
  for (const [end, prev] of [[0, 1], [idx.length - 1, idx.length - 2]] as const) {
    const e = fids[idx[end]!]!, q = fids[idx[prev]!]!;
    // The end link maps photo to plane: a scale and rotation, as complex numbers
    // (photo y flipped to point up like the plane's).
    const dPx: Vec2 = [e.center_px[0] - q.center_px[0], q.center_px[1] - e.center_px[1]];
    const dMm: Vec2 = [pinsMm[end]![0] - pinsMm[prev]![0], pinsMm[end]![1] - pinsMm[prev]![1]];
    if (norm2(...dPx) < 1e-9 || norm2(...dMm) < 1e-9) continue;
    for (const c of [...candidates].sort((a, b) => a - b)) {
      const f = fids[c]!;
      if (f.polarity !== e.polarity || Math.abs(f.outer_axes_px[0] / e.outer_axes_px[0] - 1.0) > 0.25) continue;
      if (Math.abs(f.surround - (e.surround + q.surround) / 2) > 3.0 * spread) continue;
      const v: Vec2 = [f.center_px[0] - e.center_px[0], e.center_px[1] - f.center_px[1]];
      const step = cdiv(cmul(v, dMm), dPx);
      const rel = cdiv(step, dMm);
      const turn = Math.abs((Math.atan2(rel[1], rel[0]) * 180) / Math.PI);
      if (Math.abs(norm2(...step) / spec.pitch_mm - 1.0) < 0.15 && turn <= spec.max_bend_deg + 10.0) {
        found++;
        break;
      }
    }
  }
  return found;
}

function cmul(a: Vec2, b: Vec2): Vec2 {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

function cdiv(a: Vec2, b: Vec2): Vec2 {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
}
