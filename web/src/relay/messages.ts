// What paired pages say to each other, inside sealed frames (crypto.ts).
//
// The phone sends curves; the desktop stores them. Each side also says what it
// holds in a `state` message: the revision of each curve, its star, and the
// curves deleted (sync.ts has the rules). The relay keeps nothing, so both
// pages send their whole state whenever the other may have appeared, and a
// change at once while both are open; a curve stays in the phone's outbox
// until a state from the desktop shows its revision, so the relay may drop anything.
//
// Versioning: `v` on a curve is bumped when a field changes meaning or goes
// away. A page must ignore fields it does not know and message types it does
// not know, so that a newer phone and an older desktop tab keep working.
//
// Everything here is checked on arrival (parseMessage): the sender is
// authenticated by the pairing key, but may be an older or newer build.

import type { Role } from "./crypto";
import { parseFiles, type PackedFiles } from "./files";

export const CURVE_VERSION = 1;

/** Sanity limits for a curve off the network. */
export const LIMITS = { points: 2000, texts: 50, textChars: 500, coordMm: 1e5 };

export type Point = [number, number];

/** One measured curve: decision D5, the curve and its diagnostics, never the photo. */
export interface Curve {
  v: typeof CURVE_VERSION;
  /** Made by the phone, unique for ever, one per photo: an edited photo's curve keeps it. */
  id: string;
  /** Its revision: 1 when first sent, one more each time the photo's curve changes
   * (a pin edit, the object side). The desktop keeps the highest. Absent: 1 (older phones). */
  rev?: number;
  /** When it was measured, ISO 8601. */
  measured_at: string;
  /** The photo's file name, for the person to tell curves apart. */
  photo: string;
  chain: { name: string; pitch_mm: number; half_width_mm: number; n_pins: number };
  /** Points on the target curve, mm, in chain order. What Copy points copies. */
  points_mm: Point[];
  /** The chain's pin centres, mm, for drawing. */
  pins_mm: Point[];
  object_side: "inside" | "outside";
  /** Things that may make the measurement wrong. Shown in orange. */
  warnings: string[];
  notes: string[];
  /** The output files, gzipped (D14, files.ts). Absent when they would not fit in one message. */
  files?: PackedFiles;
  diagnostics: {
    focal_source: "exif" | "override" | "default" | "estimated";
    focal_35mm: number;
    tilt_deg: number;
    missing_pins: number;
    link_residual_rms_mm: number;
    link_residual_max_mm: number;
  };
}

export type Message =
  /** Sent on connecting and whenever a peer joins: who is there, for the status line. */
  | { type: "hello"; role: Role; device: string; app: string }
  | { type: "curve"; curve: Curve }
  /** The desktop has stored these curves. Idempotent. Kept for phones from before `state`. */
  | { type: "ack"; ids: string[] }
  /** What the sender holds (sync.ts). `full`: all of it, else only what just changed. */
  | { type: "state"; full: boolean; curves: CurveState[]; deleted: string[] };

/** One curve as a page holds it: the revision it has and the star. */
export interface CurveState {
  id: string;
  rev: number;
  /** Starred, and when that was last set (ms since 1970): the later setting wins. */
  star: boolean;
  star_at: number;
  /** A person's name for it ("": the photo's file name) and when it was set; the later wins.
   * Absent from pages before names: no name, never set. */
  name?: string;
  name_at?: number;
}

/** Entries a state message may carry (a full state of a hundred curves and the
 * last deletions is ~15 KB, well inside the relay's 64 K frame). */
export const STATE_LIMITS = { curves: 500, deleted: 500 };
/** The longest name a state may carry (sync.ts NAME_CHARS keeps names shorter). */
const NAME_LIMIT = 200;

/** A new curve id: time-ordered, so a list sorted by id is in order of measuring. */
export function newCurveId(now: number = Date.now()): string {
  const rand = crypto.getRandomValues(new Uint8Array(8));
  return now.toString(36).padStart(9, "0") + "-" + Array.from(rand, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** A message off the network, checked. null for anything this build does not
 * know or that breaks the limits: ignore it, do not fail. */
export function parseMessage(body: unknown): Message | null {
  if (!isObject(body)) return null;
  switch (body.type) {
    case "hello":
      if ((body.role === "phone" || body.role === "desktop") && isText(body.device) && isText(body.app)) {
        return { type: "hello", role: body.role, device: body.device, app: body.app };
      }
      return null;
    case "ack":
      if (Array.isArray(body.ids) && body.ids.length <= 1000 && body.ids.every(isId)) {
        return { type: "ack", ids: body.ids as string[] };
      }
      return null;
    case "state": {
      const { curves, deleted } = body;
      if (!Array.isArray(curves) || curves.length > STATE_LIMITS.curves) return null;
      if (!Array.isArray(deleted) || deleted.length > STATE_LIMITS.deleted || !deleted.every(isId)) return null;
      const out: CurveState[] = [];
      for (const c of curves) {
        if (!isObject(c) || !isId(c.id) || !isRev(c.rev) || typeof c.star !== "boolean" || !isNum(c.star_at)) return null;
        const named = c.name !== undefined || c.name_at !== undefined;
        if (named && (typeof c.name !== "string" || c.name.length > NAME_LIMIT || !isNum(c.name_at))) return null;
        out.push({ id: c.id, rev: c.rev, star: c.star, star_at: c.star_at, ...(named ? { name: c.name as string, name_at: c.name_at as number } : {}) });
      }
      return { type: "state", full: body.full === true, curves: out, deleted: deleted as string[] };
    }
    case "curve": {
      const curve = parseCurve(body.curve);
      return curve && { type: "curve", curve };
    }
    default:
      return null;
  }
}

export function parseCurve(c: unknown): Curve | null {
  if (!isObject(c) || c.v !== CURVE_VERSION || !isId(c.id)) return null;
  const chain = c.chain;
  const d = c.diagnostics;
  if (!isObject(chain) || !isObject(d)) return null;
  const points = parsePoints(c.points_mm);
  const pins = parsePoints(c.pins_mm);
  const warnings = parseTexts(c.warnings);
  const notes = parseTexts(c.notes);
  if (!points || points.length < 2 || !pins || !warnings || !notes) return null;
  if (!isText(c.measured_at) || Number.isNaN(Date.parse(c.measured_at)) || !isText(c.photo)) return null;
  if (c.object_side !== "inside" && c.object_side !== "outside") return null;
  if (!isText(chain.name) || !isNum(chain.pitch_mm) || !isNum(chain.half_width_mm) || !isNum(chain.n_pins)) return null;
  const sources = ["exif", "override", "default", "estimated"] as const;
  const source = sources.find((s) => s === d.focal_source);
  const nums = [d.focal_35mm, d.tilt_deg, d.missing_pins, d.link_residual_rms_mm, d.link_residual_max_mm];
  if (!source || !nums.every(isNum)) return null;
  const files = c.files === undefined ? undefined : parseFiles(c.files);
  if (files === null) return null;
  if (c.rev !== undefined && !isRev(c.rev)) return null;
  // Built field by field: unknown fields from a newer build are dropped, not stored.
  return {
    v: CURVE_VERSION,
    id: c.id,
    rev: (c.rev as number | undefined) ?? 1,
    measured_at: c.measured_at,
    photo: c.photo,
    chain: { name: chain.name, pitch_mm: chain.pitch_mm, half_width_mm: chain.half_width_mm, n_pins: chain.n_pins },
    points_mm: points,
    pins_mm: pins,
    object_side: c.object_side,
    warnings,
    notes,
    diagnostics: {
      focal_source: source,
      focal_35mm: d.focal_35mm as number,
      tilt_deg: d.tilt_deg as number,
      missing_pins: d.missing_pins as number,
      link_residual_rms_mm: d.link_residual_rms_mm as number,
      link_residual_max_mm: d.link_residual_max_mm as number,
    },
    ...(files ? { files } : {}),
  };
}

/** The clipboard table the Fusion add-in's Paste points reads (decision D6).
 * Must stay byte for byte what splinewire/output.py points_tsv writes. */
export function pointsTsv(points_mm: readonly Point[]): string {
  return ["x_mm\ty_mm", ...points_mm.map(([x, y]) => `${fixed4(x)}\t${fixed4(y)}`)].join("\n") + "\n";
}

// Python's f"{x:.4f}" and toFixed(4) both round the exact binary value
// (JavaScript's toFixed is exact up to 1e21), so they agree; -0 is the one
// difference: Python writes "-0.0000", toFixed "0.0000" for -0 but "-0.0000"
// for a tiny negative. Match Python.
function fixed4(x: number): string {
  const s = x.toFixed(4);
  return Object.is(x, -0) ? "-" + s : s;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}
function isText(v: unknown): v is string {
  return typeof v === "string" && v.length <= LIMITS.textChars;
}
function isRev(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= 1e9;
}
function isId(v: unknown): v is string {
  return typeof v === "string" && /^[A-Za-z0-9_-]{8,64}$/.test(v);
}
function parsePoints(v: unknown): Point[] | null {
  if (!Array.isArray(v) || v.length > LIMITS.points) return null;
  const out: Point[] = [];
  for (const p of v) {
    if (!Array.isArray(p) || p.length !== 2 || !isNum(p[0]) || !isNum(p[1])) return null;
    if (Math.abs(p[0]) > LIMITS.coordMm || Math.abs(p[1]) > LIMITS.coordMm) return null;
    out.push([p[0], p[1]]);
  }
  return out;
}
function parseTexts(v: unknown): string[] | null {
  return Array.isArray(v) && v.length <= LIMITS.texts && v.every(isText) ? (v as string[]) : null;
}
