// splinewire/live.py: order and rectify the blobs a video frame detector
// found, for the live camera overlay. Nothing here is a measurement: the
// numbers coach the person, and the still goes through the full pipeline.
// Returns what the worker's "analyse" command returns (core/protocol.ts
// FrameAnalysis), so the overlay can run it in the page with no worker.

import { fixed, type Vec2 } from "./numeric";
import { orderChain } from "./order";
import { minLinks, rectifyChain } from "./rectify";
import type { ChainSpec } from "./types";

// More candidates than this is a busy background, not a chain worth ordering at video rate.
export const MAX_POINTS = 120;

export interface LiveAnalysis {
  order: number[];
  gaps: number[];
  tilt_deg: number | null;
  link_residual_max_mm: number | null;
}

/** The chain among a frame's blobs. focalPx: in this frame's pixels, or null
 * to estimate it from the chain. Never throws for a frame without a chain. */
export function analyse(
  points: readonly Vec2[], diameters: readonly number[], size: readonly [number, number], spec: ChainSpec,
  focalPx: number | null = null,
): LiveAnalysis {
  const none: LiveAnalysis = { order: [], gaps: [], tilt_deg: null, link_residual_max_mm: null };
  if (!(points.length >= 3 && points.length <= MAX_POINTS) || diameters.length !== points.length) return none;
  let order;
  try {
    order = orderChain(points, {
      axesPx: diameters.map((d) => [d, d] as Vec2), pitchPerDiameter: spec.pitch_mm / spec.fiducial_mm,
      nPins: spec.n_pins, maxTurnDeg: spec.max_bend_deg,
    });
  } catch {
    return none;
  }
  const out: LiveAnalysis = { order: order.indices, gaps: order.gaps, tilt_deg: null, link_residual_max_mm: null };
  if (order.links.length >= minLinks(focalPx !== null)) {
    try {
      const rect = rectifyChain(order.indices.map((i) => points[i]!), order.links, spec.pitch_mm, size, focalPx);
      out.tilt_deg = Number(fixed(rect.tilt_deg, 1));
      out.link_residual_max_mm = Number(fixed(rect.residual_max_mm, 3));
    } catch {
      return out;
    }
  }
  return out;
}
