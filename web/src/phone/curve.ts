// From a measurement to the message the phone sends (src/relay/messages.ts Curve).

import type { MeasurementDocument } from "../core/types";
import type { PackedFiles } from "../relay/files";
import { CURVE_VERSION, newCurveId, type Curve } from "../relay/messages";

export interface CurveContext {
  /** The chain preset's name, as the person sees it in settings. */
  chainName: string;
  pitch_mm: number;
  half_width_mm: number;
  n_pins: number;
  now?: Date;
}

/** A Curve for a measurement that succeeded; null if it did not (there is nothing to send). */
export function curveFrom(doc: MeasurementDocument, ctx: CurveContext, files?: PackedFiles): Curve | null {
  const r = doc.result;
  if (!doc.ok || !r || !doc.curve) return null;
  const now = ctx.now ?? new Date();
  const curve: Curve = {
    v: CURVE_VERSION,
    id: newCurveId(now.getTime()),
    measured_at: now.toISOString(),
    photo: doc.name.slice(0, 200),
    chain: { name: ctx.chainName.slice(0, 200), pitch_mm: ctx.pitch_mm, half_width_mm: ctx.half_width_mm, n_pins: ctx.n_pins },
    points_mm: doc.curve.points,
    pins_mm: doc.curve.pins,
    object_side: r.object_side,
    warnings: r.warnings,
    notes: r.notes,
    diagnostics: {
      focal_source: r.focal_source,
      focal_35mm: r.focal_35mm,
      tilt_deg: r.tilt_deg,
      missing_pins: r.missing_pins,
      link_residual_rms_mm: r.link_residual_rms_mm,
      link_residual_max_mm: r.link_residual_max_mm,
    },
  };
  if (files && Object.keys(files).length) (curve as Curve & { files: PackedFiles }).files = files;
  return curve;
}
