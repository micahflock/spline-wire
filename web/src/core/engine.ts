/// <reference types="vite/client" />
// The core worker in TypeScript (D18): answers protocol.ts's requests with
// the port in src/pipeline/ instead of Python under Pyodide, so there is
// nothing to download and nothing to load. Pages make it exactly as they made
// worker.ts, and the protocol is unchanged.
//
// One exception: a HEIC photo the browser cannot decode (every browser but
// Safari). Such a photo is handed, with all its later commands, to the
// Pyodide worker (worker.ts, with pillow-heif), started inside this one the
// first time it is needed.

import { focalPxFrom35mm, PhotoSession } from "../pipeline/session";
import { decodeToGray } from "../pipeline/decode";
import { PipelineError } from "../pipeline/errors";
import { analyse } from "../pipeline/live";
import { specFromDict } from "../pipeline/chain";
import { readExif } from "../phone/exif";
import { MAX_OPEN_PHOTOS, type Command, type Commands, type Failure, type Request, type Response } from "./protocol";
import type { OutputKind, PhotoInfo } from "./types";

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
const delegated = new Set<string>();                    // photos the Pyodide worker holds
// Every open photo, either engine's, least recently used first: at most
// MAX_OPEN_PHOTOS in all, as the protocol says.
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
  if (delegated.delete(photo)) void callPython("close", { photo }).catch(() => {});
}

function session(photo: string): PhotoSession {
  const s = open.get(photo);
  if (!s) throw new Failed({ kind: "closed", message: `photo '${photo}' is not open` });
  touch(photo);
  return s;
}

/** The format Pillow would name, from the file's first bytes. */
function formatOf(u: Uint8Array): string | null {
  const text = (a: number, b: number) => String.fromCharCode(...u.subarray(a, b));
  if (u[0] === 0xff && u[1] === 0xd8) return isMpo(u) ? "MPO" : "JPEG";
  if (text(1, 4) === "PNG") return "PNG";
  if (text(0, 4) === "RIFF" && text(8, 12) === "WEBP") return "WEBP";
  if (isHeif(u)) return "HEIF";
  if (text(0, 3) === "GIF") return "GIF";
  if (text(0, 2) === "BM") return "BMP";
  if (text(0, 4) === "II*\0" || text(0, 4) === "MM\0*") return "TIFF";
  return null;
}

/** A JPEG with a Multi-Picture (MPF) APP2 segment, which Pillow opens as "MPO" (many iPhone photos). */
function isMpo(u: Uint8Array): boolean {
  let off = 2;
  while (off + 8 <= u.length && u[off] === 0xff) {
    const m = u[off + 1]!;
    if (m === 0xda || m === 0xd9) return false;
    const len = (u[off + 2]! << 8) | u[off + 3]!;
    if (m === 0xe2 && String.fromCharCode(...u.subarray(off + 4, off + 7)) === "MPF") return true;
    off += 2 + len;
  }
  return false;
}

/** As worker.ts: an ISO base media file of a HEIF brand. */
function isHeif(u: Uint8Array): boolean {
  if (u.length < 12) return false;
  const text = (a: number, b: number) => String.fromCharCode(...u.subarray(a, b));
  return text(4, 8) === "ftyp" && ["heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs", "mif1", "msf1", "avif"].includes(text(8, 12));
}

function stem(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return (dot > 0 ? base.slice(0, dot) : base) || "photo";
}

// ---- the Pyodide worker, for HEIC photos the browser cannot decode ----

let python: Worker | null = null;
let pythonNext = 1;
const pythonWaiting = new Map<number, { resolve(v: unknown): void; reject(e: Failed): void }>();

function callPython<C extends Command>(cmd: C, args: Commands[C]["args"], transfer: Transferable[] = []): Promise<Commands[C]["result"]> {
  if (!python) {
    python = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    python.onmessage = (ev: MessageEvent<Response>) => {
      const msg = ev.data;
      if ("event" in msg) {
        self.postMessage(msg);          // its loading progress, for the page's bar
        return;
      }
      const w = pythonWaiting.get(msg.id);
      if (!w) return;
      pythonWaiting.delete(msg.id);
      if (msg.ok) w.resolve(msg.result);
      else w.reject(new Failed(msg.error));
    };
    python.onerror = (ev) => {
      for (const w of pythonWaiting.values()) {
        w.reject(new Failed({ kind: "bug", type: "WorkerError", message: ev.message || "the HEIC decoder stopped" }));
      }
      pythonWaiting.clear();
      python?.terminate();
      python = null;
      for (const photo of delegated) forget(photo);
    };
  }
  const id = pythonNext++;
  return new Promise((resolve, reject) => {
    pythonWaiting.set(id, { resolve: resolve as (v: unknown) => void, reject });
    python!.postMessage({ id, cmd, args } as Request, transfer);
  });
}

// ---- the commands ----

type Handlers = { [C in Command]: (args: Commands[C]["args"]) => Promise<Commands[C]["result"]> };

const handlers: Handlers = {
  async load() {
    self.postMessage({ event: "progress", progress: { stage: "ready", fraction: 1 } } satisfies Response);
    return { versions: { splinewire: import.meta.env.APP_VERSION ?? "dev", engine: "typescript" }, ms: 0 };
  },

  async open({ photo, name, data }) {
    const bytes = new Uint8Array(data);
    forget(photo);                                    // opened again: the new bytes win
    while (recent.length >= MAX_OPEN_PHOTOS) forget(recent[0]!);
    let gray;
    try {
      gray = await decodeToGray(new Blob([bytes]));
    } catch {
      if (isHeif(bytes)) {
        delegated.add(photo);
        touch(photo);
        return callPython("open", { photo, name, data: bytes.buffer as ArrayBuffer }, [bytes.buffer as ArrayBuffer]);
      }
      throw new Failed({ kind: "person", message: `${name} is not a photo this program can read (JPEG, HEIC, PNG, TIFF, BMP or WebP)` });
    }
    const exif = readExif(bytes);
    // The stored size, before turning upright (Pillow's im.size); the focal length
    // depends only on the diagonal, which turning keeps.
    const turned = (exif.orientation ?? 1) >= 5;
    const stored: [number, number] = turned ? [gray.height, gray.width] : [gray.width, gray.height];
    const info: PhotoInfo = {
      format: formatOf(bytes), width: stored[0], height: stored[1], make: exif.make, model: exif.model,
      focal_mm: exif.focalMm, focal_35mm: exif.focal35,
    };
    const focal = exif.focal35 ? focalPxFrom35mm(exif.focal35, stored) : null;
    open.set(photo, new PhotoSession(name, gray, focal, info));
    touch(photo);
    return { info };
  },

  async measure({ photo, options }) {
    if (delegated.has(photo)) {
      touch(photo);
      return callPython("measure", { photo, options });
    }
    return session(photo).measure(options);
  },

  async outputs({ photo }) {
    if (delegated.has(photo)) {
      touch(photo);
      return callPython("outputs", { photo });
    }
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
