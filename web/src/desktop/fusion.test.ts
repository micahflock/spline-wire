import { describe, expect, it, vi } from "vitest";
import { MemoryStore } from "../relay/store";
import {
  FUSION_PROTOCOL, FusionBridge, FusionError, InsertLog, autoInsert, findSend, fusionMode, markText, waitForSend,
  type Inserted, type Send,
} from "./fusion";

const CURVE = { id: "0mi4x2-ab12", photo: "IMG_0042.jpg", points_mm: [[0, 0], [10, 5]] as [number, number][] };

/** The add-in as palette.py answers, returning a Promise as Fusion's palette does (S6). */
function addin(answer: (action: string, msg: Record<string, unknown>) => unknown, promise = true) {
  const calls: Array<{ action: string; msg: Record<string, unknown> }> = [];
  const send: Send = (action, data) => {
    const msg = JSON.parse(data) as Record<string, unknown>;
    calls.push({ action, msg });
    const text = JSON.stringify(answer(action, msg));
    return promise ? Promise.resolve(text) : text;
  };
  return { send, calls };
}

describe("FusionBridge", () => {
  it("asks for the status with the protocol version", async () => {
    const { send, calls } = addin(() => ({ ok: true, v: 1, addin: "0.2.0", sketch: "Sketch3" }));
    const bridge = new FusionBridge(send, () => {});
    expect(await bridge.status()).toEqual({ addin: "0.2.0", sketch: "Sketch3" });
    expect(calls).toEqual([{ action: "status", msg: { v: FUSION_PROTOCOL } }]);
  });

  it("works with a fusionSendData that returns the text itself (older Fusion)", async () => {
    const { send } = addin(() => ({ ok: true, v: 1, addin: "0.2.0", sketch: null }), false);
    expect(await new FusionBridge(send, () => {}).status()).toEqual({ addin: "0.2.0", sketch: null });
  });

  it("sends an insert with the curve's id, name, points and how", async () => {
    const { send, calls } = addin((_a, m) => (m.how === "ask" ? { ok: true, queued: true } : { ok: true, queued: false, sketch: null }));
    const bridge = new FusionBridge(send, () => {});
    expect(await bridge.insert(CURVE, "auto")).toBe("no-sketch");
    expect(await bridge.insert(CURVE, "ask")).toBe("queued");
    expect(calls[1]).toEqual({ action: "insert", msg: { v: 1, id: CURVE.id, photo: CURVE.photo, points_mm: CURVE.points_mm, how: "ask" } });
  });

  it("gives the add-in's refusal as a message for the person", async () => {
    const { send } = addin(() => ({ ok: false, error: "The curve's points must be 2 to 2000 pairs" }));
    await expect(new FusionBridge(send, () => {}).insert(CURVE, "ask")).rejects.toThrow(/2 to 2000/);
    await expect(new FusionBridge(send, () => {}).insert(CURVE, "ask")).rejects.toBeInstanceOf(FusionError);
  });

  it("says so when Fusion answers nothing, garbage, throws, or speaks another version", async () => {
    for (const send of [() => Promise.resolve(""), () => "not json", () => Promise.resolve(undefined)] as Send[]) {
      await expect(new FusionBridge(send, () => {}).status()).rejects.toThrow(/is the Spline Wire add-in running/);
    }
    await expect(new FusionBridge(() => Promise.reject(new Error("gone")), () => {}).status()).rejects.toThrow(/didn't answer \(gone\)/);
    const { send } = addin(() => ({ ok: true, v: 2 }));
    await expect(new FusionBridge(send, () => {}).status()).rejects.toThrow(/download it again/);
  });

  it("passes on what the add-in pushes about an insert, and ignores anything else", () => {
    const got: Inserted[] = [];
    const bridge = new FusionBridge(() => "", (r) => got.push(r));
    expect(bridge.handle("inserted", JSON.stringify({ v: 1, id: "a", ok: true, sketch: "Sketch3", points: 13 }))).toBe("OK");
    bridge.handle("inserted", JSON.stringify({ v: 1, id: "b", ok: false, error: "Not inserted: the dialog was cancelled.", cancelled: true }));
    for (const [action, data] of [["inserted", "{"], ["inserted", '{"v":2,"id":"c","ok":true}'], ["other", "{}"], ["inserted", '{"v":1}']]) {
      expect(bridge.handle(action!, data!)).toBe("OK");          // never empty: Fusion wants a string back
    }
    expect(got).toEqual([
      { id: "a", ok: true, sketch: "Sketch3", points: 13 },
      { id: "b", ok: false, error: "Not inserted: the dialog was cancelled.", cancelled: true },
    ]);
  });
});

describe("finding adsk", () => {
  it("binds fusionSendData to adsk", () => {
    const adsk = { fusionSendData: vi.fn(function (this: unknown) { return this; }) };
    expect(findSend({ adsk })!("status", "{}")).toBe(adsk);
    expect(findSend({})).toBeNull();
    expect(findSend({ adsk: {} })).toBeNull();
  });

  it("waits for adsk to appear, or gives up", async () => {
    const w: { adsk?: { fusionSendData?: unknown } } = {};
    let slept = 0;
    const sleep = async (ms: number) => {
      slept += ms;
      if (slept >= 100) w.adsk = { fusionSendData: () => "{}" };
    };
    expect(await waitForSend(w, 1000, sleep)).not.toBeNull();
    expect(slept).toBe(100);
    expect(await waitForSend({}, 200, async () => {})).toBeNull();
  });

  it("knows the ?fusion mode", () => {
    expect(fusionMode("?fusion")).toBe(true);
    expect(fusionMode("?relay=ws://x&fusion")).toBe(true);
    expect(fusionMode("?relay=ws://x")).toBe(false);
  });
});

describe("InsertLog", () => {
  it("keeps each curve's mark, drops an insert under way on reload, and tidies to the list", () => {
    const store = new MemoryStore();
    const log = new InsertLog(store);
    log.set("a", { state: "inserted", sketch: "Sketch3", at: "2026-11-19T10:00:00Z" });
    log.set("b", { state: "inserting" });
    log.set("c", { state: "waiting" });
    const again = new InsertLog(store);
    expect(again.get("a")).toEqual({ state: "inserted", sketch: "Sketch3", at: "2026-11-19T10:00:00Z" });
    expect(again.get("b")).toBeUndefined();
    again.keepOnly(["a"]);
    expect(new InsertLog(store).get("c")).toBeUndefined();
    store.set("inserted", "[1, 2]");
    expect(new InsertLog(store).get("0")).toBeUndefined();
  });

  it("puts marks in words", () => {
    expect(markText({ state: "inserted", sketch: "Sketch3", at: "" })).toBe("Inserted into Sketch3");
    expect(markText({ state: "failed", error: "Open a design.", cancelled: false })).toBe("Not inserted: Open a design.");
    expect(markText({ state: "failed", error: "x", cancelled: true })).toBe("Not inserted: cancelled");
    expect(markText({ state: "waiting" })).toMatch(/no sketch was open/);
    expect(markText(undefined)).toBe("");
  });

  it("inserts arriving curves unless turned off", () => {
    const store = new MemoryStore();
    expect(autoInsert(store)).toBe(true);
    store.set("auto-insert", "off");
    expect(autoInsert(store)).toBe(false);
  });
});
