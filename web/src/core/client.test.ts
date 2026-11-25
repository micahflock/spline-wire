import { describe, expect, it } from "vitest";
import { CoreClient, type WorkerLike } from "./client";
import { CoreError, type Request, type Response } from "./protocol";
import type { MeasureOptions, MeasurementDocument } from "./types";

/** A worker that records requests and answers when told to. */
class FakeWorker implements WorkerLike {
  onmessage: WorkerLike["onmessage"] = null;
  onerror: WorkerLike["onerror"] = null;
  sent: { msg: Request; transfer: Transferable[] }[] = [];
  terminated = false;

  postMessage(message: unknown, transfer: Transferable[] = []): void {
    this.sent.push({ msg: message as Request, transfer });
  }
  terminate(): void { this.terminated = true; }

  reply(msg: Response): void { this.onmessage!({ data: msg } as MessageEvent); }
  ok(i: number, result: unknown): void { this.reply({ id: this.sent[i]!.msg.id, ok: true, result }); }
  fail(i: number, kind: "person" | "closed" | "bug", message = "no"): void {
    this.reply({ id: this.sent[i]!.msg.id, ok: false, error: { kind, message } });
  }
  crash(message?: string): void { this.onerror!({ message } as ErrorEvent); }
}

function setup() {
  const workers: FakeWorker[] = [];
  const core = new CoreClient(() => {
    const w = new FakeWorker();
    workers.push(w);
    return w;
  });
  return { core, workers };
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const options = { chain: { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 13 } } as MeasureOptions;
const doc = { schema: "spline-wire/measurement@1", ok: true } as MeasurementDocument;

describe("CoreClient", () => {
  it("matches responses to calls by id, whatever their order", async () => {
    const { core, workers } = setup();
    const a = core.call("close", { photo: "a" });
    const b = core.call("stats", {});
    const w = workers[0]!;
    expect(workers).toHaveLength(1);                     // one worker for all calls
    expect(w.sent.map((s) => s.msg.cmd)).toEqual(["close", "stats"]);
    expect(w.sent[0]!.msg.id).not.toBe(w.sent[1]!.msg.id);
    w.ok(1, { heap_bytes: 5, open: [] });
    w.ok(0, null);
    w.reply({ id: 999, ok: true, result: "nobody asked" });   // ignored
    await expect(b).resolves.toEqual({ heap_bytes: 5, open: [] });
    await expect(a).resolves.toBeNull();
  });

  it("passes progress events on and turns failures into CoreErrors", async () => {
    const { core, workers } = setup();
    const seen: unknown[] = [];
    core.onProgress = (p) => seen.push(p);
    const load = core.preload();
    const w = workers[0]!;
    w.reply({ event: "progress", progress: { stage: "ready", fraction: 0.5 } });
    w.reply({ event: "progress", progress: { stage: "ready", fraction: 1 } });
    w.fail(0, "person", "not a photo");
    expect(seen).toEqual([{ stage: "ready", fraction: 0.5 }, { stage: "ready", fraction: 1 }]);
    const err = await load.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CoreError);
    expect((err as CoreError).failure.kind).toBe("person");
    expect((err as CoreError).message).toBe("not a photo");
    const bug = core.call("stats", {});
    w.reply({ id: w.sent[1]!.msg.id, ok: false, error: { kind: "bug", type: "KeyError", message: "'x'" } });
    await expect(bug).rejects.toThrow("KeyError: 'x'");
  });

  it("opens a closed photo again from the page's copy and retries the measurement", async () => {
    const { core, workers } = setup();
    let fetched = 0;
    const bytes = new ArrayBuffer(8);
    const source = { name: "IMG_1.jpg", bytes: async () => { fetched++; return bytes; } };
    const result = core.measure("p1", options, source);
    const w = workers[0]!;
    w.fail(0, "closed");
    await flush();
    expect(fetched).toBe(1);
    expect(w.sent[1]!.msg).toMatchObject({ cmd: "open", args: { photo: "p1", name: "IMG_1.jpg", data: bytes } });
    expect(w.sent[1]!.transfer).toEqual([bytes]);          // transferred, not copied
    w.ok(1, { info: {} });
    await flush();
    expect(w.sent[2]!.msg).toMatchObject({ cmd: "measure", args: { photo: "p1", options } });
    w.ok(2, doc);
    await expect(result).resolves.toBe(doc);
  });

  it("does not reopen for other failures, nor when the photo is open", async () => {
    const { core, workers } = setup();
    const source = { name: "a.jpg", bytes: async () => { throw new Error("should not be read"); } };
    const open = core.measure("p", options, source);
    workers[0]!.ok(0, doc);
    await expect(open).resolves.toBe(doc);
    const failing = core.measure("p", options, source);
    workers[0]!.fail(1, "bug", "boom");
    await expect(failing).rejects.toThrow("boom");
    expect(workers[0]!.sent).toHaveLength(2);
  });

  it("restart throws the worker away, fails what was waiting, and starts afresh", async () => {
    const { core, workers } = setup();
    const waiting = core.call("stats", {});
    core.restart();
    expect(workers[0]!.terminated).toBe(true);
    await expect(waiting).rejects.toThrow("restarted");
    const next = core.call("stats", {});
    expect(workers).toHaveLength(2);
    workers[1]!.ok(0, { heap_bytes: null, open: [] });
    await expect(next).resolves.toEqual({ heap_bytes: null, open: [] });
  });

  it("a worker that dies fails its calls, and the next call starts a new one", async () => {
    const { core, workers } = setup();
    const a = core.call("stats", {});
    const b = core.call("close", { photo: "x" });
    workers[0]!.crash();
    expect(workers[0]!.terminated).toBe(true);
    for (const p of [a, b]) {
      const err = (await p.catch((e: unknown) => e)) as CoreError;
      expect(err.failure).toMatchObject({ kind: "bug", type: "WorkerError" });
      expect(err.message).toMatch(/out of memory/);
    }
    const c = core.call("stats", {});
    expect(workers).toHaveLength(2);
    workers[1]!.crash("RangeError: Out of memory");
    await expect(c).rejects.toThrow("WorkerError: RangeError: Out of memory");
  });

  it("latest says whether a newer request for the photo was made", () => {
    const { core } = setup();
    const first = core.latest("a");
    expect(first()).toBe(true);
    const other = core.latest("b");
    expect(first()).toBe(true);
    const second = core.latest("a");
    expect(first()).toBe(false);
    expect(second()).toBe(true);
    expect(other()).toBe(true);
  });
});
