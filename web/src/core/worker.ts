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
// as opencv-python), then `import splinewire.bridge`. ezdxf and its
// dependencies (DXF output only; 1.4 s to import) are loaded after the first
// measurement, while the person looks at it, or on the first `outputs`.

import type { PyodideAPI } from "pyodide";
import type { Command, Commands, Failure, Progress, Request, Response } from "./protocol";

declare const self: DedicatedWorkerGlobalScope;

interface Manifest {
  pyodide: { version: string; python: string; index: string };
  boot: { packages: string[]; bytes: number };
  extras: { packages: string[]; bytes: number };
  wheels: { splinewire: string; ezdxf: string };
}

// The site's root. Built, this file is assets/worker-<hash>.js; under the dev
// server it is /src/core/worker.ts and the root is /.
const ROOT = new URL(import.meta.env.DEV ? "/" : "../", import.meta.url);

type PyCall = ((cmd: string, args: string, data?: Uint8Array) => string) & { destroy(): void };

let loading: Promise<Commands["load"]["result"]> | null = null;
let extras: Promise<void> | null = null;
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
  // Fractions follow S1's warm timings: runtime ~2 s, packages ~3.5 s, imports ~3 s.
  progress("runtime", 0);
  const res = await fetch(new URL("core-manifest.json", ROOT), { cache: "no-cache" });
  if (!res.ok) throw new Error(`core-manifest.json: HTTP ${res.status} (run npm run assets)`);
  manifest = (await res.json()) as Manifest;
  const index = new URL(manifest.pyodide.index, ROOT).href;
  const { loadPyodide } = (await import(/* @vite-ignore */ index + "pyodide.mjs")) as typeof import("pyodide");
  py = await loadPyodide({ indexURL: index });

  progress("packages", 0.2);
  const wanted = [...manifest.boot.packages, new URL(manifest.wheels.splinewire, ROOT).href];
  await py.loadPackage(wanted, { messageCallback: () => {} });

  progress("starting", 0.6);
  const mod = py.pyimport("splinewire.bridge");
  bridgeCall = mod.call as PyCall;
  mod.destroy();
  const versions = bridge("versions", {}) as Record<string, string>;
  progress("ready", 1);
  return { versions: { ...versions, pyodide: manifest.pyodide.version }, ms: Math.round(performance.now() - t0) };
}

function loadExtras(): Promise<void> {
  extras ??= (async () => {
    await load();
    await py.loadPackage([...manifest.extras.packages, new URL(manifest.wheels.ezdxf, ROOT).href],
                         { messageCallback: () => {} });
  })().catch((err) => {
    extras = null;
    throw err;
  });
  return extras;
}

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
    return bridge("open", { photo, name }, new Uint8Array(data)) as Commands["open"]["result"];
  },
  async measure({ photo, options }) {
    await load();
    const doc = bridge("measure", { photo, options }) as Commands["measure"]["result"];
    // The files are likely wanted next: get ezdxf in while the person looks at the result.
    if (doc.ok) setTimeout(() => void loadExtras().catch(() => {}), 0);
    return doc;
  },
  async outputs({ photo }) {
    await loadExtras();
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
