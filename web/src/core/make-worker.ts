// The measuring engine's worker, for both pages: the TypeScript port of the
// pipeline (engine.ts, D18). `new Worker(new URL(...))` is written out so the
// bundler finds it.

export function makeCoreWorker(): Worker {
  return new Worker(new URL("./engine.ts", import.meta.url), { type: "module" });
}
