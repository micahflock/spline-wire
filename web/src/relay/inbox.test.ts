// The desktop's list of curves: a curve measured on the desktop (D9) beside
// those a phone sent.
import { describe, expect, it } from "vitest";
import { INBOX_CAP, Inbox } from "./inbox";
import { newCurveId, type Curve, type Message } from "./messages";
import { MemoryStore } from "./store";

function curve(id = newCurveId()): Curve {
  return {
    v: 1, id, rev: 1, measured_at: "2026-11-18T08:00:00Z", photo: "IMG_0001.jpg",
    chain: { name: "O-ring chain", pitch_mm: 10, half_width_mm: 4, n_pins: 13 },
    points_mm: [[0, 0], [9.5, 1.25]], pins_mm: [[0, 4], [10, 5]], object_side: "inside", warnings: [], notes: [],
    diagnostics: { focal_source: "exif", focal_35mm: 26, tilt_deg: 12.5, missing_pins: 0,
      link_residual_rms_mm: 0.03, link_residual_max_mm: 0.06 },
  };
}

function inbox(store = new MemoryStore()) {
  const sent: Message[] = [];
  const arrived: Curve[] = [];
  const box = new Inbox(store, { send: async (m) => (sent.push(m), true) }, { arrived: (c) => arrived.push(c) });
  return { box, sent, arrived, store };
}

describe("a curve measured on this computer", () => {
  it("joins the list at the top, marked as made here, with nothing acknowledged and no arrival", () => {
    const { box, sent, arrived, store } = inbox();
    const phone = curve();
    box.onMessage({ type: "curve", curve: phone });
    const here = curve();
    box.addLocal(here);
    expect(box.curves.map((c) => c.id)).toEqual([here.id, phone.id]);
    expect(box.isLocal(here.id)).toBe(true);
    expect(box.isLocal(phone.id)).toBe(false);
    expect(sent).toEqual([{ type: "ack", ids: [phone.id] },
      { type: "state", full: false, curves: [{ id: phone.id, rev: 1, star: false, star_at: 0, name: "", name_at: 0 }], deleted: [] }]);
    expect(arrived).toEqual([phone]);
    // A new page reads the same list and marks.
    const again = new Inbox(store, { send: async () => true });
    expect(again.curves.map((c) => c.id)).toEqual([here.id, phone.id]);
    expect(again.isLocal(here.id)).toBe(true);
  });

  it("is added once, and its mark goes with it when deleted or pushed off the end", () => {
    const { box, store } = inbox();
    const here = curve();
    box.addLocal(here);
    box.addLocal(here);
    expect(box.curves).toHaveLength(1);
    box.remove(here.id);
    expect(box.isLocal(here.id)).toBe(false);
    expect(JSON.parse(store.get("inbox-here")!)).toEqual([]);
    const first = curve();
    box.addLocal(first);
    for (let i = 0; i < INBOX_CAP; i++) box.addLocal(curve());
    expect(box.curves).toHaveLength(INBOX_CAP);
    expect(box.isLocal(first.id)).toBe(false);
    expect(JSON.parse(store.get("inbox-here")!)).toHaveLength(INBOX_CAP);
  });
});
