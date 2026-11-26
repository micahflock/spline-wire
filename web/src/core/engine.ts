/// <reference types="vite/client" />
// The core worker (D18): answers protocol.ts's requests with the TypeScript
// port of the pipeline in src/pipeline/, so there is nothing to download and
// nothing to load. (It replaced the Python package under Pyodide, removed on
// 2026-11-25, D29.)
//
// A HEIC photo the browser cannot decode (every browser but Safari) is
// decoded with libheif in WebAssembly (src/pipeline/heif.ts, D28), loaded
// only then.

import { PhotoSession } from "../pipeline/session";
import { isHeif, photoInfo } from "../pipeline/photo-info";
import { decodeToGray } from "../pipeline/decode";
import { PipelineError } from "../pipeline/errors";
import { analyse } from "../pipeline/live";
import { specFromDict } from "../pipeline/chain";
import heifWasmUrl from "libheif-js/libheif-wasm/libheif.wasm?url";
import { MAX_OPEN_PHOTOS, type Command, type Commands, type EnginePreviews, type Failure, type Request, type Response } from "./protocol";
import type { OutputKind } from "./types";
import type { LibheifFactory } from "../pipeline/heif";
import { boxDownscale, type Rgba } from "../photo/downscale";
import { PREVIEW_QUALITY, THUMB_QUALITY, previewSizeOf, thumbSizeOf } from "../photo/preview";

declare const self: DedicatedWorkerGlobalScope;

// session.OUTPUT_NAMES: a download's name after the photo's stem.
const OUTPUT_NAMES: Record<Exclude<OutputKind, "tsv">, string> = {
  json: "{stem}.json", csv: "{stem}-curve.csv", svg: "{stem}-curve.svg", dxf: "{stem}-curve.dxf",
  fusion_csv: "{stem}-fusion-cm.csv",
};

class Failed extends Error {
  constructor(readonly failure: Failure) {
    super(failure.message);
  }
}

const open = new Map<string, PhotoSession>();
// Every open photo, least recently used first: at most MAX_OPEN_PHOTOS, as the protocol says.
const recent: string[] = [];

function touch(photo: string): void {
  const i = recent.indexOf(photo);
  if (i >= 0) recent.splice(i, 1);
  recent.push(photo);
}

function forget(photo: string): void {
  const i = recent.indexOf(photo);
  if (i >= 0) recent.splice(i, 1);
  open.delete(photo);
}

function session(photo: string): PhotoSession {
  const s = open.get(photo);
  if (!s) throw new Failed({ kind: "closed", message: `photo '${photo}' is not open` });
  touch(photo);
  return s;
}

function stem(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return (dot > 0 ? base.slice(0, dot) : base) || "photo";
}

// ---- HEIC photos the browser cannot decode ----

let heifWasm: Promise<ArrayBuffer> | null = null;

async function decodeHeif(bytes: Uint8Array, name: string, withPreview: boolean) {
  heifWasm ??= fetch(heifWasmUrl).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  let parts;
  try {
    parts = await Promise.all([import("../pipeline/heif"), import("libheif-js/libheif-wasm/libheif.js"), heifWasm]);
  } catch (err) {
    heifWasm = null;                                    // try again with the next photo
    throw new Failed({ kind: "person", message: `could not load the HEIC decoder (${(err as Error)?.message ?? err}); check the connection and try again` });
  }
  const [{ decodeHeif: decode }, lib, wasm] = parts;
  return decode(bytes, name, lib.default as unknown as LibheifFactory, wasm, withPreview ? previewSizeOf : undefined);
}

/** The pages' preview and thumbnail (photo/preview.ts), from the preview's pixels. */
async function encodePreviews(pixels: { width: number; height: number }, preview: Rgba): Promise<EnginePreviews> {
  const thumb = thumbSizeOf(pixels);
  const small = boxDownscale(preview.data, preview.width, preview.height, preview.width * 4, 4, thumb.width, thumb.height);
  const [previewBlob, thumbBlob] = await Promise.all([jpeg(preview, PREVIEW_QUALITY), jpeg(small, THUMB_QUALITY)]);
  return { pixels, previewSize: { width: preview.width, height: preview.height }, preview: previewBlob, thumb: thumbBlob };
}

function jpeg(img: Rgba, quality: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas to make the photo's preview");
  ctx.putImageData(new ImageData(img.data as Uint8ClampedArray<ArrayBuffer>, img.width, img.height), 0, 0);
  return canvas.convertToBlob({ type: "image/jpeg", quality });
}

// ---- the commands ----

type Handlers = { [C in Command]: (args: Commands[C]["args"]) => Promise<Commands[C]["result"]> };

const handlers: Handlers = {
  async load() {
    self.postMessage({ event: "progress", progress: { stage: "ready", fraction: 1 } } satisfies Response);
    return { versions: { splinewire: import.meta.env.APP_VERSION ?? "dev", engine: "typescript" }, ms: 0 };
  },

  async open({ photo, name, data, previews: wanted }) {
    const bytes = new Uint8Array(data);
    forget(photo);                                    // opened again: the new bytes win
    while (recent.length >= MAX_OPEN_PHOTOS) forget(recent[0]!);
    const heif = isHeif(bytes);
    let gray;
    let previews: EnginePreviews | null = null;
    try {
      gray = await decodeToGray(new Blob([bytes]));
    } catch {
      if (!heif) throw new Failed({ kind: "person", message: `${name} is not a photo this program can read (JPEG, HEIC, PNG, TIFF, BMP or WebP)` });
      // The browser cannot decode it, so the page cannot draw it either: make its previews here.
      const decoded = await decodeHeif(bytes, name, !!wanted);
      gray = decoded.gray;
      if (decoded.preview) {
        try {
          previews = await encodePreviews({ width: gray.width, height: gray.height }, decoded.preview);
        } catch {
          previews = null;                              // no OffscreenCanvas here: measured, not drawn
        }
      }
    }
    const { info, focalPx } = photoInfo(bytes, gray);
    open.set(photo, new PhotoSession(name, gray, focalPx, info));
    touch(photo);
    return wanted ? { info, previews } : { info };
  },

  async measure({ photo, options }) {
    return session(photo).measure(options);
  },

  async outputs({ photo }) {
    const s = session(photo);
    let files;
    try {
      files = s.outputs();
    } catch (err) {
      if (err instanceof PipelineError) throw new Failed({ kind: "person", message: err.message });
      throw err;
    }
    const st = stem(s.name);
    const names = Object.fromEntries(Object.entries(OUTPUT_NAMES).map(([k, v]) => [k, v.replace("{stem}", st)]));
    return { files, names };
  },

  async close({ photo }) {
    forget(photo);
    return null;
  },

  async analyse(frame) {
    if (typeof frame !== "object" || frame === null || typeof frame.chain !== "object" || frame.chain === null) {
      throw new Failed({ kind: "person", message: "a frame must be an object with points, diameters, width, height and a chain" });
    }
    const spec = specFromDict(frame.chain as unknown as Record<string, unknown>);
    return analyse(frame.points, frame.diameters, [frame.width, frame.height], spec, frame.focal_px);
  },

  async stats() {
    return { heap_bytes: null, open: [...recent] };
  },
};

function failure(err: unknown): Failure {
  if (err instanceof Failed) return err.failure;
  if (err instanceof PipelineError) return { kind: "person", message: err.message };
  const e = err as { name?: string; message?: string };
  return { kind: "bug", type: e?.name ?? "Error", message: String(e?.message ?? err) };
}

// One request at a time, in order: an `outputs` sent after a `measure` must see it.
let queue: Promise<void> = Promise.resolve();

self.onmessage = (ev: MessageEvent<Request>) => {
  const { id, cmd, args } = ev.data;
  queue = queue.then(async () => {
    let reply: Response;
    try {
      const handler = handlers[cmd] as (a: typeof args) => Promise<unknown>;
      if (!handler) throw new Failed({ kind: "bug", type: "TypeError", message: `unknown command ${String(cmd)}` });
      reply = { id, ok: true, result: await handler(args) };
    } catch (err) {
      reply = { id, ok: false, error: failure(err) };
    }
    self.postMessage(reply);
  });
};
