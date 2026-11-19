// The test page's logic (spikes/accuracy/, docs/real-world-tests.md): one
// test photo measured and scored against its target, the record kept for
// it, and the report sent back. Kept apart from the browser, so it is tested
// in Node (bench.test.ts); the page wires it to the camera, storage and DOM.
//
// The report is what `splinewire score --report` reads (scoring.REPORT_SCHEMA):
// each test's photo file, chain, pin count and target, so the saved photos
// can be scored again by Python, the reference, with any code version.

import type { Gray } from "../pipeline/cv";
import { detectFiducials } from "../pipeline/detect";
import { PipelineError } from "../pipeline/errors";
import { measureDetections, type Measurement } from "../pipeline/measure";
import { headline, score, type Score } from "../pipeline/scoring";
import { focalPxFrom35mm } from "../pipeline/session";
import { gaugeByName, isPlaque, PLAQUES } from "../pipeline/targets";
import type { ChainSpec, Side } from "../pipeline/types";

export const REPORT_SCHEMA = "spline-wire/test-report@1";

/** The proposed pass marks (docs/real-world-tests.md): a plaque's worst pin,
 * a gauge's worst contact point without the ends and its radius. */
export const PASS = { plaqueMm: 0.3, gaugeMm: 0.5, radiusMm: 0.5 };

export type Colourway = "light" | "dark";
export const COLOURWAYS: Record<Colourway, string> = {
  light: "White rims on black links",
  dark: "Orange rims on white links",
};

export type FocalSource = "override" | "exif" | "remembered" | "saved" | "estimated" | "fitted";

export interface FocalRun {
  source: FocalSource;
  focal_35mm: number | null;
  score?: Score;
  error?: string;
}

export interface Results {
  pins_found: number;
  tilt_deg: number;
  warnings: string[];
  focal: { main: FocalRun; estimated?: FocalRun; fitted?: FocalRun };
  ms: number;
}

export interface TestRecord {
  id: string;
  photo: string;
  created: string;
  phone: string;
  chain: "snap";
  shade: Colourway;
  pins: number;
  /** The pitch the photo was measured with: the plaque's own for a plaque, the
   * chain's (as measured with calipers, else the preset's) otherwise. */
  pitch_mm: number;
  target: string;
  route: "camera" | "frame" | "library";
  conditions: { tilt: string; surface: string; light: string; frame: string };
  exif: { make: string | null; model: string | null; focal_mm: number | null; focal_35mm: number | null };
  saved_focal_35mm: number | null;
  /** The override set on the page, used instead of the EXIF; null if none. */
  focal_override_35mm?: number | null;
  size: [number, number];
  results?: Results;
  error?: string;
}

/** The object's side the target implies (scoring.target_side). */
export function targetSide(target: string): Side | null {
  const g = gaugeByName(target);
  if (!g || g.concave === undefined) return null;
  return g.concave ? "outside" : "inside";
}

/** The pitch a chain's own pitch setting may take, mm: a typing slip outside it is ignored. */
export const PITCH_RANGE = { min: 12, max: 14 };

/** The chain a test is measured with. A plaque was printed from the link's
 * design, so it is measured at its own pitch whatever the chain's is; the
 * chain on a gauge or an object at the chain's pitch as measured with calipers
 * (docs/real-world-tests.md, phase 0), else the preset's. */
export function specFor(base: ChainSpec, target: string, links: number, shade: Colourway, chainPitch: number | null): ChainSpec {
  const plaque = PLAQUES.find((p) => p.name === target);
  const measured = chainPitch !== null && chainPitch >= PITCH_RANGE.min && chainPitch <= PITCH_RANGE.max ? chainPitch : null;
  const pitch = plaque ? plaque.pitch_mm : measured ?? base.pitch_mm;
  return { ...base, n_pins: links, fiducial_shade: shade, pitch_mm: pitch };
}

/** The focal length a photo is measured with: an override set on the page
 * (for a camera whose EXIF is off, as the iPhone's in-page still was by ~5%
 * in phase 1), else its EXIF, else the lens the
 * phone page remembered for it, else the saved setting, else none. */
export function chooseFocal(exif35: number | null, remembered35: number | null, saved35: number | null,
  override35: number | null = null): { source: FocalSource; f35: number | null } {
  if (override35) return { source: "override", f35: override35 };
  if (exif35) return { source: "exif", f35: exif35 };
  if (remembered35) return { source: "remembered", f35: remembered35 };
  if (saved35) return { source: "saved", f35: saved35 };
  return { source: "estimated", f35: null };
}

/** Measure the photo and score it: with the focal length chooseFocal gives,
 * for a plaque also with the focal length that fits it best (fitFocal), and
 * (when the focal length is known) with it estimated from the chain, to show
 * what an estimate costs. Detection runs once. */
export function measureAndScore(gray: Gray, spec: ChainSpec, target: string,
  focal: { source: FocalSource; f35: number | null }, now: () => number = () => Date.now()): Results {
  const t0 = now();
  const size: [number, number] = [gray.width, gray.height];
  const fids = detectFiducials(gray, spec);
  const side = targetSide(target) ?? "inside";
  const run = (source: FocalSource, f35: number | null): [FocalRun, Measurement | null] => {
    try {
      const m = measureDetections(fids, size, spec, f35 ? focalPxFrom35mm(f35, size) : null, side);
      const used35 = (m.rectification.focal_px * Math.hypot(36, 24)) / Math.hypot(size[0], size[1]);
      const entry: FocalRun = { source, focal_35mm: Math.round(used35 * 100) / 100 };
      if (target !== "object") {
        const sign = targetSide(target) ? m.object_sign : null;
        entry.score = score(target, m.rectification.pins_mm, spec, sign, m.order.gaps);
      }
      return [entry, m];
    } catch (e) {
      if (!(e instanceof PipelineError)) throw e;
      return [{ source, focal_35mm: f35, error: e.message }, null];
    }
  };
  const [main, m] = run(focal.source, focal.f35);
  if (!m) throw new PipelineError(main.error ?? "no chain found");
  const results: Results = {
    pins_found: m.order.indices.length,
    tilt_deg: Math.round(m.rectification.tilt_deg * 10) / 10,
    warnings: m.warnings,
    focal: { main },
    ms: 0,
  };
  if (focal.f35) results.focal.estimated = run("estimated", null)[0];
  if (isPlaque(target)) {
    const start = focal.f35 ?? main.focal_35mm ?? 26;
    const best = fitFocal((f35) => {
      const r = run("fitted", f35)[0];
      return r.score?.kind === "plaque" ? r.score.rms_error_mm : Infinity;
    }, 0.6 * start, 1.6 * start);
    results.focal.fitted = run("fitted", best)[0];
  }
  results.ms = now() - t0;
  return results;
}

/** The focal length (35 mm equivalent) that minimises cost(f35) between lo
 * and hi, to 0.01 mm: a coarse scan (the cost can have several dips when a
 * focal length is far off), then golden-section search around the best. As
 * scoring._fit_focal, which uses SciPy's bounded Brent search instead. */
export function fitFocal(cost: (f35: number) => number, lo: number, hi: number): number {
  const steps = 24;
  let bestI = 0, bestC = Infinity;
  const xs = Array.from({ length: steps + 1 }, (_, i) => lo + (hi - lo) * i / steps);
  xs.forEach((x, i) => {
    const c = cost(x);
    if (c < bestC) {
      bestC = c;
      bestI = i;
    }
  });
  let a = xs[Math.max(0, bestI - 1)]!, b = xs[Math.min(steps, bestI + 1)]!;
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a);
  let fc = cost(c), fd = cost(d);
  while (b - a > 0.01) {
    if (fc < fd) {
      b = d; d = c; fd = fc; c = b - g * (b - a); fc = cost(c);
    } else {
      a = c; c = d; fc = fd; d = a + g * (b - a); fd = cost(d);
    }
  }
  return Math.round(((a + b) / 2) * 100) / 100;
}

/** The plaque photos' best-fit focal lengths, for the page's summary: only
 * photos tilted 20° or more (straight on, the focal length hardly changes the
 * result, so the fit says nothing) that read the whole chain. */
export function focalSummary(tests: readonly TestRecord[]): { n: number; median: number; min: number; max: number } | null {
  const fs = tests.filter((t) => t.results && t.results.tilt_deg >= 20 && t.results.pins_found === t.pins)
    .map((t) => t.results!.focal.fitted?.focal_35mm)
    .filter((f): f is number => typeof f === "number")
    .sort((a, b) => a - b);
  if (!fs.length) return null;
  const mid = fs.length >> 1;
  const median = fs.length % 2 ? fs[mid]! : (fs[mid - 1]! + fs[mid]!) / 2;
  return { n: fs.length, median: Math.round(median * 100) / 100, min: fs[0]!, max: fs[fs.length - 1]! };
}

export interface Verdict {
  ok: boolean;
  text: string;
}

/** One line on how a run scored, against the pass marks. */
export function verdict(run: FocalRun, pinsFound: number, pins: number): Verdict {
  if (run.error) return { ok: false, text: `failed: ${run.error}` };
  const s = run.score;
  const found = pinsFound === pins ? "" : ` (${pinsFound} of ${pins} pins)`;
  if (!s) return { ok: pinsFound === pins, text: `measured${found}` };
  if (s.kind === "plaque") {
    const ok = s.max_error_mm < PASS.plaqueMm && pinsFound === pins;
    return { ok, text: `worst pin ${mm(s.max_error_mm)}, rms ${mm(s.rms_error_mm)}, scale ${s.scale.toFixed(4)}${found}` };
  }
  const i = s.inner;
  const radius = i.radius_error_mm !== undefined ? `, radius ${i.radius_error_mm >= 0 ? "+" : ""}${mm(i.radius_error_mm)}` : "";
  const ok = i.max_mm < PASS.gaugeMm && (i.radius_error_mm === undefined || Math.abs(i.radius_error_mm) < PASS.radiusMm)
    && pinsFound === pins;
  return { ok, text: `worst point ${mm(i.max_mm)} (ends left out), rms ${mm(i.rms_mm)}${radius}${found}` };
}

function mm(v: number): string {
  return `${v.toFixed(3)} mm`;
}

/** A short name for the phone in file names: "iPhone 16 Pro" -> "iphone16pro". */
export function phoneSlug(phone: string): string {
  return phone.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 16) || "phone";
}

export function testId(phone: string, n: number): string {
  return `${phoneSlug(phone)}-${String(n).padStart(3, "0")}`;
}

export function photoName(id: string, target: string, type: string): string {
  const ext = /heic|heif/i.test(type) ? "heic" : /png/i.test(type) ? "png" : "jpg";
  return `${id}-${target}.${ext}`;
}

export interface Report {
  schema: string;
  page: string;
  version: string;
  user_agent: string;
  created: string;
  tests: TestRecord[];
}

export function report(tests: readonly TestRecord[], version: string, userAgent: string, created: string): Report {
  return { schema: REPORT_SCHEMA, page: "accuracy", version, user_agent: userAgent, created, tests: [...tests] };
}

/** The report as text to paste into a message: a line per test, then the JSON. */
export function reportText(r: Report): string {
  const lines = [`Spline Wire test page ${r.version}, ${r.tests.length} test(s), ${r.created}`, r.user_agent, ""];
  for (const t of r.tests) {
    const res = t.results;
    if (!res) {
      lines.push(`${t.id} ${t.target}: ${t.error ?? "not measured"}`);
      continue;
    }
    const main = res.focal.main, est = res.focal.estimated, fit = res.focal.fitted;
    const head = (run: FocalRun | undefined) => (!run ? "-" : run.error ? "failed" : run.score ? headline(run.score).toFixed(3) : "ok");
    lines.push(`${t.id} ${t.target} ${COLOURWAYS[t.shade]}, ${t.pins} links at ${t.pitch_mm} mm, tilt ${res.tilt_deg}° (planned ${t.conditions.tilt}), `
      + `${t.conditions.surface}, ${t.conditions.light}, ${t.route}: ${res.pins_found} pins; `
      + `${main.source} ${main.focal_35mm ?? "?"} mm -> ${head(main)} mm; estimated -> ${head(est)} mm`
      + (fit ? `; best fit ${fit.focal_35mm ?? "?"} mm -> ${head(fit)} mm` : "")
      + (res.warnings.length ? ` [${res.warnings.length} warning(s)]` : ""));
  }
  return `${lines.join("\n")}\n\n${JSON.stringify(r)}\n`;
}

export function isTarget(name: string): boolean {
  return name === "object" || isPlaque(name) || gaugeByName(name) !== undefined;
}
