import { describe, expect, it } from "vitest";
import type { PhotoSource } from "../core/client";
import { CoreError } from "../core/protocol";
import type { MeasureOptions, MeasurementDocument } from "../core/types";
import { Measurer, type CoreLike, type Update } from "./measurer";

/** A core whose measurements finish when the test says so. */
class FakeCore implements CoreLike {
  calls: { photo: string; options: MeasureOptions; finish(doc?: Partial<MeasurementDocument>, err?: Error): void }[] = [];
  measure(photo: string, options: MeasureOptions, _source: PhotoSource): Promise<MeasurementDocument> {
    return new Promise((resolve, reject) => {
      this.calls.push({
        photo, options,
        finish: (doc = {}, err) => (err ? reject(err) : resolve({ ok: true, name: photo, ...doc } as MeasurementDocument)),
      });
    });
  }
}

const tick = () => new Promise((r) => setTimeout(r, 0));

function setup() {
  const core = new FakeCore();
  const opts = new Map<string, MeasureOptions>();
  const log: [string, Update][] = [];
  const m = new Measurer({
    core,
    options: (id) => opts.get(id) ?? null,
    source: (id) => ({ name: id, bytes: async () => new ArrayBuffer(0) }),
    update: (id, u) => log.push([id, u]),
    now: () => 0,
  });
  const set = (id: string, side: "inside" | "outside" = "inside") => opts.set(id, { chain: { pitch_mm: 10, half_width_mm: 4, fiducial_mm: 5, n_pins: 13 }, side });
  const statuses = (id: string) => log.filter(([x]) => x === id).map(([, u]) => u.status);
  return { core, m, log, set, opts, statuses };
}

describe("Measurer", () => {
  it("holds photos taken while the core loads, then measures them in turn", async () => {
    const { core, m, set, statuses } = setup();
    set("a");
    set("b");
    m.request("a");
    m.request("b");
    expect(statuses("a")).toEqual(["loading"]);
    expect(core.calls).toHaveLength(0);
    m.coreReady();
    await tick();
    expect(core.calls.map((c) => c.photo)).toEqual(["a"]);
    expect(statuses("b")).toEqual(["loading", "queued"]);
    core.calls[0]!.finish();
    await tick();
    expect(statuses("a")).toEqual(["loading", "queued", "measuring", "done"]);
    expect(core.calls.map((c) => c.photo)).toEqual(["a", "b"]);
    core.calls[1]!.finish({ ok: false, error: "found 2 fiducials; need the whole chain in view" });
    await tick();
    expect(statuses("b").at(-1)).toBe("error");
  });

  it("measures once more with the newest options when they change mid-measurement, showing only that", async () => {
    const { core, m, set, log, statuses } = setup();
    m.coreReady();
    set("a");
    m.request("a");
    await tick();
    expect(m.busy("a")).toBe(true);
    set("a", "outside");
    m.request("a");
    m.request("a");                                   // twice: still one more measurement
    core.calls[0]!.finish({ name: "stale" });
    await tick();
    expect(core.calls).toHaveLength(2);
    expect(core.calls[1]!.options.side).toBe("outside");
    expect(log.some(([, u]) => u.doc?.name === "stale")).toBe(false);
    core.calls[1]!.finish({ name: "fresh" });
    await tick();
    expect(log.at(-1)![1].doc!.name).toBe("fresh");
    expect(statuses("a").filter((s) => s === "done")).toHaveLength(1);
    expect(m.busy("a")).toBe(false);
  });

  it("puts the photo being edited first", async () => {
    const { core, m, set } = setup();
    m.coreReady();
    for (const id of ["a", "b", "c"]) {
      set(id);
      m.request(id);
    }
    await tick();
    m.request("c", true);
    core.calls[0]!.finish();
    await tick();
    expect(core.calls.map((c) => c.photo)).toEqual(["a", "c"]);
  });

  it("reports failures: a person's message as is, anything else as a bug with its type", async () => {
    const { core, m, set, log } = setup();
    m.coreReady();
    set("a");
    set("b");
    m.request("a");
    m.request("b");
    await tick();
    core.calls[0]!.finish(undefined, new CoreError({ kind: "person", message: "notes.txt is not a photo this program can read" }));
    await tick();
    core.calls[1]!.finish(undefined, new TypeError("x is undefined"));
    await tick();
    const failures = log.filter(([, u]) => u.failure).map(([id, u]) => [id, u.failure]);
    expect(failures).toEqual([
      ["a", { kind: "person", message: "notes.txt is not a photo this program can read" }],
      ["b", { kind: "bug", type: "TypeError", message: "x is undefined" }],
    ]);
  });

  it("skips a deleted photo and waits again when the core is lost", async () => {
    const { core, m, set, opts, statuses } = setup();
    m.coreReady();
    set("a");
    set("b");
    set("c");
    m.request("a");
    m.request("b");
    m.request("c");
    m.forget("b");
    opts.delete("b");
    await tick();
    m.coreLoading();
    expect(statuses("c").at(-1)).toBe("loading");
    core.calls[0]!.finish();
    await tick();
    expect(core.calls.map((c) => c.photo)).toEqual(["a"]);     // c waits for the core
    m.coreReady();
    await tick();
    expect(core.calls.map((c) => c.photo)).toEqual(["a", "c"]);
  });
});
