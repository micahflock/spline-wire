// The documents the Python core hands to the page. They mirror
// splinewire/session.py (SCHEMA "spline-wire/measurement@1") field for field:
// change both together. tests/test_session.py pins the Python side.

export type Point = [number, number];
export type Side = "inside" | "outside";
export type FocalSource = "exif" | "override" | "default" | "estimated";

/** data/chain*.yaml, as JSON. */
export interface ChainSpec {
  pitch_mm: number;
  half_width_mm: number;
  fiducial_mm: number;
  n_pins: number;
  fiducial?: "dot" | "ring";
  ring_inner_mm?: number;
  max_bend_deg?: number;
  contact?: "edge" | "round";
  /** The fiducials lighter than the links ("light", the default) or darker. */
  fiducial_shade?: "light" | "dark";
}

/** Pins a person removed or added by tapping the photo, in photo pixels. */
export interface Edits {
  add: Point[];
  remove: Point[];
}

/** session.Options.from_json */
export interface MeasureOptions {
  chain: ChainSpec;
  side?: Side | Point;
  focal_override_35mm?: number | null;
  saved_focal_35mm?: number | null;
  edits?: Edits;
  truth?: { name: string; pin_points: Point[] } | null;
  compare_truth?: boolean | null;
  /** Where the chain is likely to be, photo pixels [x0, y0, x1, y1]: searched first (session.Options.search_px). */
  search_px?: [number, number, number, number] | null;
}

/** session.photo_info */
export interface PhotoInfo {
  format: string | null;
  width: number;
  height: number;
  make: string | null;
  model: string | null;
  focal_mm: number | null;
  focal_35mm: number | null;
}

export interface MeasurementResult {
  pins: number;
  missing_pins: number;
  rejected: number;
  curve_points: number;
  tilt_deg: number;
  focal_35mm: number;
  focal_source: FocalSource;
  link_residual_rms_mm: number;
  link_residual_max_mm: number;
  object_side: Side;
  /** Things that may make the measurement wrong: shown in orange. */
  warnings: string[];
  notes: string[];
  truth: Record<string, number> | null;
  truth_status: "compared" | "off" | "pins" | "shape" | null;
}

/** A marker on the photo, in photo pixels. */
export interface Marker {
  x: number;
  y: number;
  r: number;
  /** Position along the chain, for pins on it. */
  k?: number;
  /** Index into Edits.add, for pins added by hand. */
  edit?: number;
  /** For added pins: whether a fiducial was found under the tap. */
  snapped?: boolean;
}

/** What the pin editor draws over the photo (session.PhotoSession.view). */
export interface PhotoView {
  width: number;
  height: number;
  /** A typical marker radius, for drawing added pins and hit-testing taps. */
  r: number;
  pins: Marker[];
  /** k where the chain bridges a missing pin between pins k and k+1. */
  gaps: number[];
  /** Detections not on the chain. */
  unused: Marker[];
  /** The contact edge, drawn on the photo. */
  curve: Point[];
  side_px: Point | null;
  removed: { x: number; y: number; edit: number }[];
  edits: Edits;
}

export interface MeasurementDocument {
  schema: "spline-wire/measurement@1";
  name: string;
  info: PhotoInfo;
  ok: boolean;
  /** For the person, as is. With ok false, `view` may still let them fix it by hand. */
  error: string | null;
  result: MeasurementResult | null;
  curve: { points: Point[]; pins: Point[]; spline: Point[] } | null;
  view: PhotoView | null;
}

export type OutputKind = "json" | "csv" | "svg" | "dxf" | "fusion_csv" | "tsv";
