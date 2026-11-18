// The pipeline's data, as splinewire's dataclasses hold it. Points are
// [x, y] pairs; photo pixels run y-down, plane millimetres y-up.

import type { Vec2 } from "./numeric";

/** splinewire.chain.ChainSpec */
export interface ChainSpec {
  pitch_mm: number;
  half_width_mm: number;
  fiducial_mm: number;
  n_pins: number;
  fiducial: "dot" | "ring";
  ring_inner_mm: number;
  max_bend_deg: number;
  contact: "edge" | "round";
}

/** splinewire.detect.Fiducial */
export interface Fiducial {
  center_px: Vec2;
  outer_axes_px: Vec2;       // ellipse (major, minor) diameters
  inner_axes_px: Vec2;       // a ring's hole; (0, 0) for a dot
  angle_deg: number;
  residual_px: number;
  contrast: number;
  polarity: number;          // +1 light on dark, -1 dark on light
  surround: number;
}

/** splinewire.order.ChainOrder */
export interface ChainOrder {
  indices: number[];         // detection indices in chain order
  links: [number, number][]; // positions in `indices` exactly one pitch apart
  gaps: number[];            // k where indices[k] -> indices[k+1] skips one missing pin
  rejected: number[];        // detections not on the chain
}

/** splinewire.rectify.Rectification */
export interface Rectification {
  pins_mm: Vec2[];
  focal_px: number;
  focal_estimated: boolean;
  tilt_deg: number;
  residual_rms_mm: number;
  residual_max_mm: number;
  link_residuals_mm: number[];
  depth_mm: number[];
}

export type Side = "inside" | "outside";
