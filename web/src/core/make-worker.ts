// The measuring engine's worker, for both pages: the TypeScript port
// (engine.ts, D18) unless the page's address asks for ?engine=python, which
// keeps the Python core under Pyodide (worker.ts) to compare against. Each
// `new Worker(new URL(...))` is written out so the bundler finds both.

export type EngineKind = "typescript" | "python";

export function engineKind(search: string = globalThis.location?.search ?? ""): EngineKind {
  return new URLSearchParams(search).get("engine") === "python" ? "python" : "typescript";
}

export function makeCoreWorker(kind: EngineKind = engineKind()): Worker {
  return kind === "python"
    ? new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
    : new Worker(new URL("./engine.ts", import.meta.url), { type: "module" });
}
