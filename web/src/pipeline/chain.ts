// splinewire/chain.py: the chain's physical description, checked as Python
// checks it (a ValueError there is a PipelineError here).

import { PipelineError } from "./errors";
import type { ChainSpec } from "./types";

const FIDUCIALS = ["dot", "ring"] as const;
const CONTACTS = ["edge", "round"] as const;

/** ChainSpec from chain.yaml's keys or saved settings (spec_from_dict). Older
 * files describe a ring as ring_outer_mm + ring_inner_mm, with no `fiducial`. */
export function specFromDict(raw: Record<string, unknown>): ChainSpec {
  const num = (k: string): number => {
    if (!(k in raw)) throw new PipelineError(`the chain settings have no ${k}`);
    return Number(raw[k]);
  };
  let size: number, kind: string;
  if ("fiducial_mm" in raw || !("ring_outer_mm" in raw)) {
    size = num("fiducial_mm");
    kind = String(raw.fiducial ?? "dot");
  } else {
    size = num("ring_outer_mm");
    kind = String(raw.fiducial ?? "ring");
  }
  return makeSpec({
    pitch_mm: num("pitch_mm"),
    half_width_mm: num("half_width_mm"),
    fiducial_mm: size,
    n_pins: Math.trunc(num("n_pins")),
    fiducial: kind as ChainSpec["fiducial"],
    ring_inner_mm: kind === "ring" ? Number(raw.ring_inner_mm || 0) : 0,
    max_bend_deg: Number(raw.max_bend_deg || 80),
    contact: String(raw.contact || "edge") as ChainSpec["contact"],
  });
}

/** A ChainSpec, checked as ChainSpec.__post_init__ checks it. */
export function makeSpec(s: ChainSpec): ChainSpec {
  if (!(s.pitch_mm > 0)) throw new PipelineError(`pitch_mm must be positive, got ${s.pitch_mm}`);
  if (!(s.half_width_mm >= 0)) throw new PipelineError(`half_width_mm must be >= 0, got ${s.half_width_mm}`);
  if (!FIDUCIALS.includes(s.fiducial)) throw new PipelineError(`fiducial must be one of dot, ring, got ${s.fiducial}`);
  if (!CONTACTS.includes(s.contact)) throw new PipelineError(`contact must be one of edge, round, got ${s.contact}`);
  if (!(s.fiducial_mm > 0)) throw new PipelineError(`fiducial_mm must be positive, got ${s.fiducial_mm}`);
  if (s.fiducial === "ring" && !(0 < s.ring_inner_mm && s.ring_inner_mm < s.fiducial_mm)) {
    throw new PipelineError(
      `need 0 < ring_inner_mm (${s.ring_inner_mm}) < fiducial_mm (${s.fiducial_mm}) for a ring`);
  }
  if (s.fiducial_mm >= s.pitch_mm) {
    throw new PipelineError(
      `fiducial_mm (${s.fiducial_mm}) must be smaller than pitch_mm (${s.pitch_mm}) or neighbouring fiducials touch`);
  }
  if (!(s.n_pins >= 3)) throw new PipelineError(`n_pins must be >= 3, got ${s.n_pins}`);
  if (!(s.max_bend_deg >= 30 && s.max_bend_deg <= 150)) {
    throw new PipelineError(`max_bend_deg must be between 30 and 150, got ${s.max_bend_deg}`);
  }
  return { ...s };
}
