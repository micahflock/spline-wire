/// <reference types="vite/client" />
// The core worker: runs the splinewire package under Pyodide (decision D2)
// and answers protocol.ts's requests with splinewire/bridge.py.
//
// Everything is loaded from the site's own origin, as public/core-manifest.json
// (written by `npm run assets`) lists it. Pyodide 314 refuses classic workers,
// so this must be started as a module worker.
//
// Load order, shaped by S1's timings: the runtime, then the packages the
// pipeline imports and the splinewire wheel (installed without its
// dependencies: the wheel asks for opencv-python-headless, which Pyodide ships
// as opencv-python), then `import splinewire.bridge`. The DXF needs no
// package of its own (splinewire/dxf.py).

import type { PyodideAPI } from "pyodide";
import type { Command, Commands, Failure, Progress, Request, Response } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

interface Manifest {
  pyodide: { version: string; python: string; index: string };
  boot: { packages: string[]; bytes: number };
  /** Optional: a manifest from before HEIF support has none, and HEIC photos then fail as before. */
  heif?: { packages: string[]; bytes: number };
  wheels: { splinewire: string };
}

// The site's root. Built, this file is assets/worker-<hash>.js; under the dev
// server it is /src/core/worker.ts and the root is /.
const ROOT = new URL(import.meta.env.DEV ? "/" : "../", import.meta.url);

type PyCall = ((cmd: string, args: string, data?: Uint8Array) => string) & { destroy(): void };

let loading: Promise<Commands["load"]["result"]> | null = null;
let heif: Promise<void> | null = null;
let py: PyodideAPI;
let manifest: Manifest;
let bridgeCall: PyCall;

function progress(stage: Progress["stage"], fraction: number): void {
  self.postMessage({ event: "progress", progress: { stage, fraction } } satisfies Response);
}

/** Thrown to answer with a ready-made Failure. */
class Failed extends Error {
  constructor(readonly failure: Failure) { super(failure.message); }
}

function load(): Promise<Commands["load"]["result"]> {
  loading ??= loadOnce().catch((err) => {
    loading = null;            // a failed load (network) may be tried again
    throw err;
  });
  return loading;
}

async function loadOnce(): Promise<Commands["load"]["result"]> {
  const t0 = performance.now();
  // Fractions follow the e2e test's timings in desktop Chromium: runtime 2.3-3.1 s,
  // packages 2.8-3.4 s, imports 2.3-3.1 s. Passing the packages to loadPyodide
  // (downloaded while it boots) was no faster there: 8.2-10.3 s against 7.7-9.1 s.
  progress("runtime", 0);
  const res = await fetch(new URL("core-manifest.json", ROOT), { cache: "no-cache" });
  if (!res.ok) throw new Error(`core-manifest.json: HTTP ${res.status} (run npm run assets)`);
  manifest = (await res.json()) as Manifest;
  const index = new URL(manifest.pyodide.index, ROOT).href;
  const { loadPyodide } = (await import(/* @vite-ignore */ index + "pyodide.mjs")) as typeof import("pyodide");
  py = await loadPyodide({ indexURL: index });

  progress("packages", 0.3);
  const wanted = [...manifest.boot.packages, new URL(manifest.wheels.splinewire, ROOT).href];
  await py.loadPackage(wanted, { messageCallback: () => {} });

  progress("starting", 0.65);
  // Kept for the worker's life: `call` is borrowed from the module's proxy.
  bridgeCall = py.pyimport("splinewire.bridge").call as PyCall;
  const versions = bridge("versions", {}) as Record<string, string>;
  progress("ready", 1);
  return { versions: { ...versions, pyodide: manifest.pyodide.version }, ms: Math.round(performance.now() - t0) };
}

/** HEIC/HEIF decoding, loaded the first time a photo needs it. Pillow learns
 * the format from pillow_heif's opener; the pipeline registers it at import
 * when pillow_heif is there, which in the browser it is not yet. */
function loadHeif(): Promise<void> {
  heif ??= (async () => {
    await load();
    if (!manifest.heif?.packages.length) return;
    await py.loadPackage(manifest.heif.packages, { messageCallback: () => {} });
    py.runPython("from pillow_heif import register_heif_opener\nregister_heif_opener()");
  })().catch((err) => {
    heif = null;
    throw err;
  });
  return heif;
}

/** Whether the bytes are an ISO base media file of a HEIF brand (an iPhone
 * photo is "heic"). Box 1 is "ftyp" with the major brand at bytes 8-11. */
export function isHeif(data: Uint8Array): boolean {
  if (data.length < 12) return false;
  const text = (a: number, b: number) => String.fromCharCode(...data.subarray(a, b));
  return text(4, 8) === "ftyp" && HEIF_BRANDS.has(text(8, 12));
}
const HEIF_BRANDS = new Set(["heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs", "mif1", "msf1", "avif"]);

function pyCall(cmd: string, args: object, data?: Uint8Array): string {
  return bridgeCall(cmd, JSON.stringify(args), data);
}

/** A bridge command; its {ok, result | error} answer unwrapped. */
function bridge(cmd: string, args: object, data?: Uint8Array): unknown {
  const out = JSON.parse(pyCall(cmd, args, data)) as { ok: true; result: unknown } | { ok: false; error: Failure };
  if (!out.ok) throw new Failed(out.error);
  return out.result;
}

function heapBytes(): number | null {
  const m = (py as unknown as { _module?: { HEAPU8?: Uint8Array } } | undefined)?._module;
  return m?.HEAPU8?.buffer.byteLength ?? null;
}

type Handlers = { [C in Command]: (args: Commands[C]["args"]) => Promise<Commands[C]["result"]> };

const handlers: Handlers = {
  load: () => load(),
  async open({ photo, name, data }) {
    await load();
    const bytes = new Uint8Array(data);
    if (isHeif(bytes)) await loadHeif();
    return bridge("open", { photo, name }, bytes) as Commands["open"]["result"];
  },
  async measure({ photo, options }) {
    await load();
    const doc = bridge("measure", { photo, options }) as Commands["measure"]["result"];
    return doc;
  },
  async outputs({ photo }) {
    await load();
    return bridge("outputs", { photo }) as Commands["outputs"]["result"];
  },
  async close({ photo }) {
    if (!loading) return null;       // nothing can be open yet
    await load();
    return bridge("close", { photo }) as null;
  },
  async analyse(frame) {
    await load();
    return bridge("analyse", frame) as Commands["analyse"]["result"];
  },
  async stats() {
    if (!loading) return { heap_bytes: null, open: [] };
    await load();
    const { open } = bridge("stats", {}) as { open: string[] };
    return { heap_bytes: heapBytes(), open };
  },
};

function failure(err: unknown): Failure {
  if (err instanceof Failed) return err.failure;
  // A PythonError from outside the bridge (it catches its own) carries the
  // Python type; anything else is a JavaScript error, most likely while loading.
  const e = err as { type?: string; name?: string; message?: string };
  return { kind: "bug", type: e?.type ?? e?.name ?? "Error", message: String(e?.message ?? err) };
}

// One request at a time, in the order they came: Python cannot run two, and
// an `outputs` sent after a `measure` must see that measurement.
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
