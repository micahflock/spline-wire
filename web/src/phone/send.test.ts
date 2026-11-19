import { describe, expect, it } from "vitest";
import type { MeasurementDocument, Point } from "../core/types";
import type { OutboxItem } from "../relay/outbox";
import type { Curve } from "../relay/messages";
import type { PhotoRecord } from "./photos";
import { applyState, curveKey, delivery, deliveryText, nextRevision, phoneState, sendOffer, setStar, starOf, type CurveSync } from "./send";

function doc(points: Point[], side: "inside" | "outside" = "inside", ok = true): MeasurementDocument {
  return {
    schema: "spline-wire/measurement@1", name: "a.jpg", ok, error: ok ? null : "nope", view: null,
    info: { format: "JPEG", width: 1, height: 1, make: null, model: null, focal_mm: null, focal_35mm: null },
    curve: ok ? { points, pins: points, spline: points } : null,
    result: ok ? { object_side: side } as MeasurementDocument["result"] : null,
  };
}

const record = (d: MeasurementDocument | null, more: Partial<PhotoRecord> = {}): PhotoRecord =>
  ({ id: "p1", name: "a.jpg", doc: d, ...more }) as PhotoRecord;
const item = (id: string, state: OutboxItem["state"]): OutboxItem => ({ curve: { id } as Curve, state });
const ROOM = "room-a";

describe("the send rule", () => {
  const first = doc([[0, 0], [10, 0]]);
  const fixed = doc([[0, 0], [10, 1]]);

  it("offers nothing unpaired or without a curve", () => {
    expect(sendOffer(null, record(first))).toEqual({ kind: "unpaired" });
    expect(sendOffer(ROOM, record(null))).toEqual({ kind: "nothing" });
    expect(sendOffer(ROOM, record(doc([], "inside", false)))).toEqual({ kind: "nothing" });
  });

  it("offers a button for a photo from before the pairing, and none once it is synced", () => {
    const r = record(first);
    expect(sendOffer(ROOM, r)).toEqual({ kind: "send" });
    r.sync = { id: "c1", room: ROOM, rev: 1, key: curveKey(first)!, remoteRev: 0 };
    expect(sendOffer(ROOM, r)).toEqual({ kind: "synced", sync: r.sync });
    expect(sendOffer("another room", r)).toEqual({ kind: "send" });
  });

  it("makes a new revision of the same curve for each change, and none for the same curve", () => {
    const r = record(first, { autoSend: true });
    const one = nextRevision(r, ROOM, () => "c1")!;
    expect(one).toEqual({ id: "c1", room: ROOM, rev: 1, key: curveKey(first), remoteRev: 0 });
    r.sync = one;
    expect(nextRevision(r, ROOM)).toBeNull();
    r.doc = fixed;
    expect(nextRevision(r, ROOM)).toMatchObject({ id: "c1", rev: 2, key: curveKey(fixed) });
    r.doc = doc([[0, 0], [10, 0]], "outside");                   // the other side is a change too
    expect(nextRevision(r, ROOM)!.rev).toBe(2);
    r.doc = doc([], "inside", false);                            // a failed measurement sends nothing
    expect(nextRevision(r, ROOM)).toBeNull();
  });

  it("keys a curve by its clipboard table, so rounding below 0.0001 mm is the same curve", () => {
    expect(curveKey(doc([[0, 0], [10.00001, 0]]))).toBe(curveKey(first));
    expect(curveKey(doc([], "inside", false))).toBeNull();
  });

  it("says where a curve is", () => {
    const s: CurveSync = { id: "c1", room: ROOM, rev: 1, key: "", remoteRev: 0 };
    const open = { state: "open" as const, peers: 0 };
    expect(delivery(s, [item("c1", "queued")], open)).toBe("queued-no-computer");
    expect(delivery(s, [item("c1", "queued")], { state: "waiting", peers: 0 })).toBe("queued-offline");
    expect(delivery(s, [item("c1", "sent")], { state: "open", peers: 1 })).toBe("sent");
    expect(delivery(s, [], open)).toBe("delivered");
    expect(deliveryText("queued-no-computer")).toMatch(/^Queued: the computer's page is not open. It goes/);
    expect(deliveryText("queued-no-computer", 2)).toMatch(/The change goes/);
  });
});

describe("the state a phone keeps in step", () => {
  const synced = (id: string, rev = 1): PhotoRecord =>
    record(doc([[0, 0], [1, 1]]), { id: `p-${id}`, sync: { id, room: ROOM, rev, key: "k", remoteRev: 0 } });

  it("says what it holds on this pairing only", () => {
    const a = synced("c1", 3), b = record(null, { id: "p2" }), c = synced("c3");
    c.sync!.room = "another room";
    setStar(a, true, 1000);
    expect(phoneState([a, b, c], ["c9"], ROOM, true)).toEqual({ type: "state", full: true, deleted: ["c9"],
      curves: [{ id: "c1", rev: 3, star: true, star_at: 1000 }] });
    expect(phoneState([a], [], ROOM, false, []).curves).toEqual([]);
  });

  it("takes the desktop's revisions, its later stars and its deletions", () => {
    const a = synced("c1", 2), b = synced("c2"), c = synced("c3");
    setStar(b, true, 5000);
    const { changed, deleted } = applyState([a, b, c], {
      type: "state", full: true, deleted: ["c3", "c-unknown"],
      curves: [{ id: "c1", rev: 2, star: true, star_at: 10 }, { id: "c2", rev: 1, star: false, star_at: 4000 },
        { id: "c-other-phone", rev: 1, star: true, star_at: 1 }],
    }, ROOM);
    expect(changed).toEqual([a, b]);
    expect(a.sync!.remoteRev).toBe(2);
    expect(starOf(a)).toEqual({ on: true, at: 10 });
    expect(starOf(b)).toEqual({ on: true, at: 5000 });            // set here later: kept
    expect(b.sync!.remoteRev).toBe(1);
    expect(deleted).toEqual([c]);
  });

  it("stamps a star later than the last, even if the clock is behind", () => {
    const r = synced("c1");
    setStar(r, true, 9000);
    setStar(r, false, 100);
    expect(starOf(r)).toEqual({ on: false, at: 9001 });
  });
});
