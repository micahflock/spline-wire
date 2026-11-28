// The messages between a page and the core worker (src/core/engine.ts), which
// runs the TypeScript pipeline (decision D18; it began as the Python package
// under Pyodide, D2, removed on 2026-11-25, D29).
//
// One request, one response, matched by id; `progress` events may come in
// between. Photos are named by an id the page chooses (its IndexedDB key), so
// the worker can drop a photo to save memory and the page can open it again.
//
// Memory is the constraint this protocol is shaped by (S1: a 12-megapixel
// photo decoded is 48 MB, and detection needs some 235 MB more). So:
//   - the worker keeps at most MAX_OPEN_PHOTOS sessions and closes the least
//     recently used; a request for a closed photo fails with kind "closed",
//     and CoreClient opens it again from the page's copy and retries;
//   - photo bytes are transferred, not copied.

import type { MeasureOptions, MeasurementDocument, OutputKind, PhotoInfo, Point } from "./types";

export const MAX_OPEN_PHOTOS = 2;

export interface Commands {
  /** Get the engine ready (the TypeScript engine has nothing to load). Idempotent. */
  load: { args: Record<string, never>; result: { versions: Record<string, string>; ms: number } };
  /** Decode a photo. Fails (kind "person") if it is not one. */
  /** With `previews`, also the copies the pages draw, for a photo the browser could
   * not decode (HEIC outside Safari); null for any other photo, which the page draws itself. */
  /** `channel`: what of its colour to read (ChainSpec gray_channel, "luma" by default);
   * a measure whose chain reads another decodes the photo again (the worker keeps its bytes). */
  open: {
    args: { photo: string; name: string; data: ArrayBuffer; previews?: boolean; channel?: "luma" | "r" | "g" | "b" };
    result: { info: PhotoInfo; previews?: EnginePreviews | null };
  };
  /** Measure an open photo. A photo with no chain in it is a document with ok false, not a failure. */
  measure: { args: { photo: string; options: MeasureOptions }; result: MeasurementDocument };
  /** The latest measurement's files, as text, with their download names. */
  outputs: { args: { photo: string }; result: { files: Record<OutputKind, string>; names: Record<string, string> } };
  close: { args: { photo: string }; result: null };
  /** For the live overlay: order and rectify points a frame detector found (pipeline/live.ts, as splinewire/live.py). */
  analyse: { args: FramePoints; result: FrameAnalysis };
  /** What the worker holds: a wasm heap's size (null: the TypeScript engine has none) and
   * the open photos, least recently used first. For deciding when to restart(). */
  stats: { args: Record<string, never>; result: { heap_bytes: number | null; open: string[] } };
}

/** What photo/preview.ts's previews() makes, made by the engine instead (JPEG blobs). */
export interface EnginePreviews {
  /** The photo's size, upright, as the core reads it. */
  pixels: { width: number; height: number };
  previewSize: { width: number; height: number };
  preview: Blob;
  thumb: Blob;
}

export type Command = keyof Commands;

export type Request = { [C in Command]: { id: number; cmd: C; args: Commands[C]["args"] } }[Command];

export interface Failure {
  /** person: the message is for the person, shown as is (a Python ValueError).
   *  closed: the photo is not open in the worker; open it again and retry.
   *  bug: anything else, shown with its type so it can be reported. */
  kind: "person" | "closed" | "bug";
  message: string;
  /** The exception's type, for bugs. */
  type?: string;
}

export type Response =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: Failure }
  | { event: "progress"; progress: Progress };

/** How far the engine has got loading (the TypeScript engine: ready at once). */
export interface Progress {
  stage: "ready";
  /** 0 to 1 over the whole load. */
  fraction: number;
}

/** Fiducial candidates a frame detector found, in frame pixels. */
export interface FramePoints {
  points: Point[];
  /** Each blob's diameter in px, same order. */
  diameters: number[];
  width: number;
  height: number;
  chain: MeasureOptions["chain"];
  /** In px of this frame; null to estimate from the chain. */
  focal_px: number | null;
}

export interface FrameAnalysis {
  /** Indices into `points`, in chain order. Empty if no chain was found. */
  order: number[];
  gaps: number[];
  /** null when the chain cannot be rectified (too few pins). */
  tilt_deg: number | null;
  link_residual_max_mm: number | null;
  /** A unit vector in the frame's pixel axes (x right, y down) toward the
   * chain's far side: the way to move the phone, still aimed at the chain, to
   * look straight down on it. null with tilt_deg, or straight on. */
  tilt_dir: [number, number] | null;
}

export class CoreError extends Error {
  constructor(readonly failure: Failure) {
    super(failure.kind === "bug" && failure.type ? `${failure.type}: ${failure.message}` : failure.message);
  }
}
