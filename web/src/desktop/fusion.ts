// The desktop page's ?fusion mode (docs/fusion-palette.md): the page inside the
// Spline Wire add-in's palette in Fusion, where curves from the phone go
// straight into a sketch. This file is the bridge to the add-in and the
// contract with it; fusion/SplineWire/palette.py is the other side. Change both
// together, and FUSION_PROTOCOL with them (tests/test_fusion_addin.py checks the
// names agree).
//
//   page -> add-in   adsk.fusionSendData(action, JSON) -> the reply's JSON text, in a Promise
//                    (S6: a Promise in Fusion 2705; older Fusions returned the text itself)
//     "status" {v}                              -> {ok, v, addin, sketch: name | null}
//     "insert" {v, id, photo, points_mm, how}   -> {ok, queued: true}; for how "auto" with no
//                                                  sketch being edited, {ok, queued: false, sketch: null}
//     refused                                   -> {ok: false, error: <a message for a person>}
//   add-in -> page   palette.sendInfoToHTML("inserted", JSON) -> window.fusionJavaScriptHandler.handle
//     {v, id, ok: true, sketch, points} | {v, id, ok: false, error, cancelled}
//
// "auto" is a curve that has just arrived: it goes into the sketch being
// edited, or waits. "ask" is the person's Insert: with no sketch open, Fusion
// asks for a plane for a new one. Python checks everything it is sent (it is
// the boundary: this page's JavaScript runs inside Fusion).
//
// window.adsk appears a little after the page's first script (S6: 41 ms), so
// the page waits for it.

import type { Curve } from "../relay/messages";
import { readJson, type KeyValueStore } from "../relay/store";

export const FUSION_PROTOCOL = 1;

export type Send = (action: string, data: string) => unknown;
export type How = "auto" | "ask";

export interface FusionStatus {
  /** The add-in's version. */
  addin: string;
  /** The sketch being edited, or null. */
  sketch: string | null;
}

export type Inserted =
  | { id: string; ok: true; sketch: string; points: number }
  | { id: string; ok: false; error: string; cancelled: boolean };

/** A failure to tell the person, in their words. */
export class FusionError extends Error {}

/** Whether the page was opened in its Fusion mode (`?fusion` in its address). */
export function fusionMode(search: string): boolean {
  return new URLSearchParams(search).has("fusion");
}

interface WithAdsk {
  adsk?: { fusionSendData?: unknown };
}

/** adsk.fusionSendData, bound, if the page is in a Fusion palette. */
export function findSend(w: WithAdsk): Send | null {
  const adsk = w.adsk;
  const fn = adsk?.fusionSendData;
  return typeof fn === "function" ? (action, data) => (fn as Send).call(adsk, action, data) : null;
}

/** adsk.fusionSendData once it appears, or null after `timeoutMs` (not in Fusion). */
export async function waitForSend(
  w: WithAdsk,
  timeoutMs: number,
  sleep: (ms: number) => Promise<unknown> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<Send | null> {
  const step = 25;
  for (let waited = 0; ; waited += step) {
    const send = findSend(w);
    if (send || waited >= timeoutMs) return send;
    await sleep(step);
  }
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export class FusionBridge {
  constructor(
    private readonly send: Send,
    /** How an insert ended, pushed by the add-in. */
    private readonly inserted: (r: Inserted) => void,
  ) {}

  async status(): Promise<FusionStatus> {
    const r = await this.call("status", {});
    if (r.v !== FUSION_PROTOCOL) {
      throw new FusionError("The Spline Wire add-in doesn't match this page: download it again and reload the palette.");
    }
    return { addin: typeof r.addin === "string" ? r.addin : "?", sketch: typeof r.sketch === "string" ? r.sketch : null };
  }

  /** "queued": the add-in will insert it and say how it went (`inserted`);
   * "no-sketch": an arrived curve with no sketch being edited, which waits. */
  async insert(curve: Pick<Curve, "id" | "photo" | "points_mm">, how: How): Promise<"queued" | "no-sketch"> {
    const r = await this.call("insert", { id: curve.id, photo: curve.photo, points_mm: curve.points_mm, how });
    return r.queued === true ? "queued" : "no-sketch";
  }

  /** window.fusionJavaScriptHandler.handle: what the add-in pushes. It must return a non-empty string. */
  handle(action: string, data: string): string {
    if (action !== "inserted") return "OK";
    let r: unknown;
    try {
      r = JSON.parse(data);
    } catch {
      return "OK";
    }
    if (!isObject(r) || r.v !== FUSION_PROTOCOL || typeof r.id !== "string") return "OK";
    if (r.ok === true) {
      this.inserted({ id: r.id, ok: true, sketch: String(r.sketch ?? "the sketch"), points: Number(r.points) || 0 });
    } else {
      this.inserted({ id: r.id, ok: false, error: String(r.error ?? "Not inserted."), cancelled: r.cancelled === true });
    }
    return "OK";
  }

  private async call(action: string, body: object): Promise<Record<string, unknown>> {
    let text: unknown;
    try {
      text = await this.send(action, JSON.stringify({ v: FUSION_PROTOCOL, ...body }));
    } catch (e) {
      throw new FusionError(`Fusion didn't answer (${(e as Error)?.message ?? e}).`);
    }
    let reply: unknown = null;
    try {
      reply = typeof text === "string" && text ? JSON.parse(text) : null;
    } catch {
      // reported below
    }
    if (!isObject(reply)) throw new FusionError("Fusion gave no answer: is the Spline Wire add-in running?");
    if (reply.ok !== true) throw new FusionError(typeof reply.error === "string" ? reply.error : "The Spline Wire add-in refused.");
    return reply;
  }
}

// ---- what became of each curve -------------------------------------------------

export type InsertMark =
  | { state: "inserting" }
  | { state: "inserted"; sketch: string; at: string }
  /** It arrived with no sketch being edited. */
  | { state: "waiting" }
  | { state: "failed"; error: string; cancelled: boolean };

const LOG_KEY = "inserted";

/** Each curve's mark, kept in the page's store so a reload still says where it went. */
export class InsertLog {
  private marks: Record<string, InsertMark>;

  constructor(private readonly store: KeyValueStore) {
    const all = readJson<Record<string, InsertMark>>(store, LOG_KEY, {});
    // An insert under way when the page closed may or may not have happened: say nothing.
    this.marks = Object.fromEntries(Object.entries(isObject(all) ? all : {}).filter(([, m]) => isObject(m) && m.state !== "inserting"));
  }

  get(id: string): InsertMark | undefined {
    return this.marks[id];
  }

  set(id: string, mark: InsertMark): void {
    this.marks[id] = mark;
    this.save();
  }

  /** Forget where a curve went: a newer revision of it is not in any sketch yet. */
  forget(id: string): void {
    if (!(id in this.marks)) return;
    delete this.marks[id];
    this.save();
  }

  /** Drop the marks of curves no longer in the list. */
  keepOnly(ids: readonly string[]): void {
    const keep = new Set(ids);
    for (const id of Object.keys(this.marks)) if (!keep.has(id)) delete this.marks[id];
    this.save();
  }

  private save(): void {
    this.store.set(LOG_KEY, JSON.stringify(this.marks));
  }
}

/** A mark in words, for the list and the curve view. */
export function markText(mark: InsertMark | undefined): string {
  switch (mark?.state) {
    case "inserting": return "Inserting…";
    case "inserted": return `Inserted into ${mark.sketch}`;
    case "waiting": return "Not inserted: no sketch was open";
    case "failed": return mark.cancelled ? "Not inserted: cancelled" : `Not inserted: ${mark.error}`;
    default: return "";
  }
}

/** Whether new curves go straight into the open sketch: on unless the person turned it off. */
export const AUTO_KEY = "auto-insert";
export function autoInsert(store: KeyValueStore): boolean {
  return store.get(AUTO_KEY) !== "off";
}
