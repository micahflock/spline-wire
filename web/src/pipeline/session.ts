// splinewire/session.py in TypeScript: one decoded photo, measured in
// memory, described as the documents the pages already read
// (core/types.ts, schema "spline-wire/measurement@1"). The session keeps
// what is slow to get (the photo, its detections), so measuring again after
// an edit, a pin count or an object side only orders and solves.
//
// Decoding the photo is the caller's (decode.ts in a page or worker; PNG in
// the tests): this takes the gray image, the focal length in pixels from its
// EXIF if any, and its PhotoInfo.

import type { MeasureOptions, MeasurementDocument, OutputKind, PhotoInfo, PhotoView } from "../core/types";
import { splineSamples } from "../desktop/spline";
import { specFromDict } from "./chain";
import type { Gray, GrayF } from "./cv";
import { detectFiducials, searchBox, type Box } from "./detect";
import { dxfText } from "./dxf";
import { toPhotoPx } from "./drawing";
import { editsFromJson, removeDetections, type Edits } from "./edits";
import { PipelineError } from "./errors";
import { measureDetections, type Measurement } from "./measure";
import { fixed, median, type Vec2 } from "./numeric";
import { csvText, fusionCsvText, jsonText, pointsTsv, pyRound, svgText } from "./output";
import { pyFloat } from "./pyjson";
import { compareToTruth, type TruthComparison } from "./truth";
import type { ChainSpec, Fiducial, Side } from "./types";

export const SCHEMA = "spline-wire/measurement@1";
// Pins this far off the truth file after fitting position, angle and scale
// mean a different shape: not the test part (session.NOT_THE_TEST_PART_PITCHES).
const NOT_THE_TEST_PART_PITCHES = 0.5;
const FULL_FRAME_DIAGONAL_MM = Math.hypot(36, 24);

export interface Truth {
  name: string;
  pins_mm: Vec2[];
}

/** session.Options: everything a person can set for one measurement. */
export interface Options {
  spec: ChainSpec;
  side: Side | Vec2;
  focal_override_35mm: number | null;
  saved_focal_35mm: number | null;
  edits: Edits;
  truth: Truth | null;
  compare_truth: boolean | null;
  /** Where the chain is likely to be (x0, y0, x1, y1, photo pixels): searched
   * first, the whole photo only if that does not give every pin (Options.search_px). */
  search_px: Box | null;
}

/** Options from the page's JSON, checked as Options.from_json checks them. */
export function optionsFromJson(doc: unknown, size?: readonly [number, number]): Options {
  if (typeof doc !== "object" || doc === null || Array.isArray(doc)) throw new PipelineError("options must be an object with a chain");
  const d = doc as Record<string, unknown>;
  if (typeof d.chain !== "object" || d.chain === null || Array.isArray(d.chain)) {
    throw new PipelineError("options must be an object with a chain");
  }
  const spec = specFromDict(d.chain as Record<string, unknown>);
  let side: Side | Vec2;
  const raw = d.side ?? "inside";
  if (typeof raw === "string") {
    if (raw !== "inside" && raw !== "outside") throw new PipelineError("side must be inside, outside or a photo point [x, y]");
    side = raw;
  } else {
    const p = Array.isArray(raw) ? raw.slice(0, 2).map(Number) : [];
    if (p.length < 2 || p.some(Number.isNaN)) throw new PipelineError("side must be inside, outside or a photo point [x, y]");
    side = [p[0]!, p[1]!];
  }
  const truth = d.truth;
  let search: Box | null = null;
  if (d.search_px !== null && d.search_px !== undefined) {
    const b = Array.isArray(d.search_px) ? d.search_px.map((v) => (typeof v === "number" || typeof v === "string" ? Number(v) : NaN)) : [];
    if (b.length !== 4 || !b.every((v) => Number.isFinite(v))) throw new PipelineError("search_px must be a box [x0, y0, x1, y1] in photo pixels");
    search = b as Box;
  }
  return {
    spec, side,
    focal_override_35mm: focal(d.focal_override_35mm, "focal_override_35mm"),
    saved_focal_35mm: focal(d.saved_focal_35mm, "saved_focal_35mm"),
    edits: editsFromJson(d.edits || {}, size),
    truth: typeof truth === "object" && truth !== null && !Array.isArray(truth)
      ? truthFromJson(truth, String((truth as Record<string, unknown>).name || "the truth file")) : null,
    compare_truth: (d.compare_truth ?? null) as boolean | null,
    search_px: search,
  };
}

function focal(value: unknown, name: string): number | null {
  if (value === null || value === undefined || value === "") return null;
  const f = typeof value === "number" || typeof value === "string" ? Number(value) : NaN;
  if (Number.isNaN(f)) throw new PipelineError(`${name} must be a number of millimetres`);
  if (!(f >= 5.0 && f <= 500.0)) {
    throw new PipelineError(`${name} must be a 35 mm-equivalent focal length between 5 and 500 mm, got ${formatG(f)}`);
  }
  return f;
}

/** Python's f"{x:g}". */
function formatG(x: number): string {
  if (!Number.isFinite(x)) return String(x).replace("Infinity", "inf");
  const e = x === 0 ? 0 : Math.floor(Math.log10(Math.abs(x)));
  if (e < -4 || e >= 6) {
    const [m, ex] = x.toExponential(5).split("e") as [string, string];
    const exp = Number(ex);
    return `${m.replace(/\.?0+$/, "")}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return String(Number(x.toPrecision(6)));
}

function truthFromJson(doc: unknown, name: string): Truth {
  const pins = (doc as Record<string, unknown>).pin_points;
  if (!Array.isArray(pins)) throw new PipelineError(`${name} is not a truth file: it has no pin_points list`);
  if (!pins.every((p) => Array.isArray(p) && p.length === 2 && p.every((v) => Number.isFinite(Number(v))))) {
    throw new PipelineError(`${name} is not a truth file: pin_points must be [x, y] pairs`);
  }
  return { name, pins_mm: pins.map((p) => [Number(p[0]), Number(p[1])] as Vec2) };
}

type FocalSource = "exif" | "override" | "default" | "estimated";

interface Measured {
  measurement: Measurement;
  options: Options;
  focal_source: FocalSource;
  truth_comparison: TruthComparison | null;
  truth_status: "compared" | "off" | "pins" | "shape" | null;
}

export function focalPxFrom35mm(f35: number, size: readonly [number, number]): number {
  return (f35 * Math.hypot(size[0], size[1])) / FULL_FRAME_DIAGONAL_MM;
}

export class PhotoSession {
  private detected: Fiducial[] | null = null;
  private detectedFor: string | null = null;
  /** The fiducials in a search box (Options.search_px), and for which kind and box. */
  private regionDetected: Fiducial[] | null = null;
  private regionFor: string | null = null;
  private grayf: GrayF | undefined;
  last: Measured | null = null;
  lastError: string | null = null;
  private edits: Edits = { add_px: [], remove_px: [] };

  constructor(
    readonly name: string,
    readonly image: Gray,
    readonly exifFocalPx: number | null,
    readonly info: PhotoInfo,
  ) {}

  get size(): [number, number] {
    return [this.image.width, this.image.height];
  }

  /** The photo's fiducials, found once per kind of fiducial. */
  detections(spec: ChainSpec): Fiducial[] {
    const key = `${spec.fiducial}:${spec.fiducial === "ring" ? spec.ring_inner_mm / spec.fiducial_mm : 0}:${spec.fiducial_shade}`;
    if (this.detected === null || (this.detectedFor !== null && this.detectedFor !== key)) {
      this.detected = detectFiducials(this.image, spec);
    }
    this.detectedFor = key;
    return this.detected;
  }

  /** The fiducials in the search box round `region` (detect.searchBox), found
   * once per kind and box; null when that box is the whole photo. */
  regionDetections(spec: ChainSpec, region: readonly number[]): Fiducial[] | null {
    const box = searchBox(this.image.width, this.image.height, region);
    if (!box) return null;
    const key = `${spec.fiducial}:${spec.fiducial === "ring" ? spec.ring_inner_mm / spec.fiducial_mm : 0}:${spec.fiducial_shade}:${box.join(",")}`;
    if (this.regionFor !== key) {
      this.regionDetected = detectFiducials(this.image, spec, box);
      this.regionFor = key;
    }
    return this.regionDetected;
  }

  /** Measure with these options; throws PipelineError with a message for the person. */
  run(options: Options): Measured {
    const size = this.size;
    let focalPx = this.exifFocalPx;
    let source: FocalSource = focalPx !== null ? "exif" : "estimated";
    if (options.focal_override_35mm !== null) {
      focalPx = focalPxFrom35mm(options.focal_override_35mm, size);
      source = "override";
    } else if (focalPx === null && options.saved_focal_35mm !== null) {
      focalPx = focalPxFrom35mm(options.saved_focal_35mm, size);
      source = "default";
    }
    this.edits = options.edits;
    this.last = null;
    const needsPhoto = options.edits.add_px.length > 0;
    if (needsPhoto && !this.grayf) this.grayf = { width: size[0], height: size[1], data: Float32Array.from(this.image.data) };
    // The search box first, if there is one: kept only if it gives every pin.
    let m: Measurement | null = null;
    const found = options.search_px ? this.regionDetections(options.spec, options.search_px) : null;
    if (found) {
      try {
        m = measureDetections(found, size, options.spec, focalPx, options.side, options.edits, this.image, this.grayf);
      } catch (err) {
        if (!(err instanceof PipelineError)) throw err;
        m = null;
      }
      if (m && m.order.indices.length < options.spec.n_pins) m = null;
    }
    m ??= measureDetections(this.detections(options.spec), size, options.spec, focalPx, options.side, options.edits, this.image, this.grayf);
    let comparison: TruthComparison | null = null;
    let status: Measured["truth_status"] = null;
    if (options.truth !== null) {
      const c = compare(m, options.truth, options.spec, options.compare_truth);
      comparison = c.t;
      status = c.status;
      if (c.why && options.compare_truth) m = { ...m, warnings: [...m.warnings, c.why] };
      else if (c.why) m = { ...m, notes: [...m.notes, c.why] };
    }
    this.last = { measurement: m, options, focal_source: source, truth_comparison: comparison, truth_status: status };
    return this.last;
  }

  /** Measure and describe the outcome as one document (session.measure). A
   * photo with no chain is a document with ok false, not an exception. */
  measure(options: MeasureOptions | Options): MeasurementDocument {
    try {
      const opts = "spec" in options ? options : optionsFromJson(options, this.size);
      this.run(opts);
      this.lastError = null;
    } catch (err) {
      if (err instanceof PipelineError) this.lastError = err.message;
      else if (err instanceof Error) this.lastError = `${err.name}: ${err.message}`;     // a bug, shown with its type
      else this.lastError = String(err);
    }
    return this.document();
  }

  document(): MeasurementDocument {
    const doc: MeasurementDocument = {
      schema: SCHEMA, name: this.name, info: this.info, ok: this.last !== null, error: this.lastError,
      result: null, curve: null, view: this.view(),
    };
    if (this.last !== null) {
      const m = this.last.measurement;
      doc.result = this.result();
      doc.curve = {
        points: rounded(m.contacts_mm, 4),
        pins: rounded(m.rectification.pins_mm, 4),
        spline: rounded(splineSamples(m.contacts_mm as [number, number][]) as Vec2[], 4),
      };
    }
    return doc;
  }

  private result(): NonNullable<MeasurementDocument["result"]> {
    const r = this.last!;
    const m = r.measurement, rect = m.rectification;
    const size = this.size;
    return {
      pins: m.order.indices.length,
      missing_pins: m.order.gaps.length,
      rejected: m.order.rejected.length,
      curve_points: m.contacts_mm.length,
      tilt_deg: pyRound(rect.tilt_deg, 1),
      focal_35mm: pyRound((rect.focal_px * FULL_FRAME_DIAGONAL_MM) / Math.hypot(size[0], size[1]), 1),
      focal_source: r.focal_source,
      link_residual_rms_mm: pyRound(rect.residual_rms_mm, 3),
      link_residual_max_mm: pyRound(rect.residual_max_mm, 3),
      object_side: m.object_side,
      warnings: [...m.warnings],
      notes: [...m.notes],
      truth: r.truth_comparison ? { ...r.truth_comparison } : null,
      truth_status: r.truth_status,
    };
  }

  /** What the pin editor draws over the photo, in photo pixels (session.view). */
  view(): PhotoView | null {
    if (this.detected === null && this.last === null) return null;
    const edits = this.edits, r = this.last;
    // The fiducial's outline as detected: an ellipse (the photo's perspective),
    // r and r2 its semi-axes, angle its major axis's direction in degrees.
    const marker = (f: Fiducial, extra: Record<string, unknown> = {}) => ({
      x: pyRound(f.center_px[0], 2), y: pyRound(f.center_px[1], 2), r: pyRound(f.outer_axes_px[0] / 2, 1),
      r2: pyRound(f.outer_axes_px[1] / 2, 1), angle: pyRound(((f.angle_deg % 180) + 180) % 180, 1), ...extra,
    });
    let pins: PhotoView["pins"], unused: PhotoView["unused"], gaps: number[], curve: [number, number][];
    let sidePx: [number, number] | null = null;
    if (r !== null) {
      const m = r.measurement;
      const byHand = new Map(m.manual.map((mp) => [mp.fiducial, mp]));
      const mark = (i: number, extra: Record<string, unknown> = {}) => {
        const mp = byHand.get(i);
        return marker(m.fiducials[i]!, { ...extra, ...(mp ? { edit: mp.edit, snapped: mp.snapped } : {}) });
      };
      pins = m.order.indices.map((i, k) => mark(i, { k }));
      unused = m.order.rejected.map((i) => mark(i));
      gaps = [...m.order.gaps];
      const samples = splineSamples(m.contacts_mm as [number, number][], 6) as Vec2[];
      curve = rounded(toPhotoPx(samples, m.rectification.pins_mm, m.pins_px), 1);
      sidePx = m.side_px ? [m.side_px[0], m.side_px[1]] : null;
    } else {
      pins = [];
      gaps = [];
      curve = [];
      unused = removeDetections(this.detected ?? [], edits).map((f) => marker(f));
    }
    const radii = (pins.length ? pins : unused).map((p) => p.r);
    const rDefault = radii.length ? median(radii) : 12.0;
    if (r === null) {
      unused = [...unused, ...edits.add_px.map(([x, y], j) => ({ x, y, r: rDefault, edit: j, snapped: false }))];
    }
    const [w, h] = this.size;
    return {
      width: w, height: h, r: rDefault, pins, gaps, unused, curve, side_px: sidePx,
      removed: edits.remove_px.map(([x, y], j) => ({ x, y, edit: j })),
      edits: { add: edits.add_px.map(([x, y]) => [x, y]), remove: edits.remove_px.map(([x, y]) => [x, y]) },
    };
  }

  /** What the JSON file says beyond the measurement (session.json_extra), in pyDumps form. */
  private jsonExtra(): Record<string, unknown> {
    const r = this.need();
    const m = r.measurement;
    const extra: Record<string, unknown> = { photo: this.name };
    const e = r.options.edits;
    if (e.add_px.length || e.remove_px.length) {
      extra.edits = { add: e.add_px.map((p) => p.map(pyFloat)), remove: e.remove_px.map((p) => p.map(pyFloat)) };
    }
    if (m.side_px !== null) extra.object_side_picked_px = m.side_px.map((v) => pyFloat(pyRound(v, 1)));
    if (r.truth_comparison !== null) {
      extra.truth_comparison = Object.fromEntries(Object.entries(r.truth_comparison).map(([k, v]) => [k, pyFloat(v)]));
    }
    return extra;
  }

  /** The latest measurement's files, as text (session.outputs). */
  outputs(): Record<OutputKind, string> {
    const m = this.need().measurement;
    return {
      json: jsonText(m, this.jsonExtra()),
      csv: csvText(m.contacts_mm),
      svg: svgText(m),
      dxf: dxfText(m.contacts_mm),
      fusion_csv: fusionCsvText(m.contacts_mm),
      tsv: pointsTsv(m.contacts_mm),
    };
  }

  private need(): Measured {
    if (this.last === null) throw new PipelineError("this photo has no measurement yet");
    return this.last;
  }
}

function compare(m: Measurement, truth: Truth, spec: ChainSpec, compareTruth: boolean | null):
  { t: TruthComparison | null; status: "compared" | "off" | "pins" | "shape"; why: string | null } {
  if (compareTruth === false) return { t: null, status: "off", why: null };
  const pins = m.rectification.pins_mm;
  if (truth.pins_mm.length !== pins.length) {
    return { t: null, status: "pins", why: `not compared to ${truth.name}: it has ${truth.pins_mm.length} pins and this chain ${pins.length}` };
  }
  const t = compareToTruth(pins, truth.pins_mm);
  if ((compareTruth === null || compareTruth === undefined) && t.max_error_scaled_mm > NOT_THE_TEST_PART_PITCHES * spec.pitch_mm) {
    return {
      t: null, status: "shape",
      why: `not compared to ${truth.name}: a different shape (pins up to ${fixed(t.max_error_scaled_mm, 1)} mm apart), so not the test part`,
    };
  }
  return { t, status: "compared", why: null };
}

function rounded(pts: readonly Vec2[], digits: number): [number, number][] {
  return pts.map(([x, y]) => [pyRound(x, digits), pyRound(y, digits)]);
}

