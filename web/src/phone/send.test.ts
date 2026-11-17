import { describe, expect, it } from "vitest";
import type { MeasurementDocument, Point } from "../core/types";
import type { OutboxItem } from "../relay/outbox";
import type { Curve } from "../relay/messages";
import type { SentCurve } from "./photos";
import { curveKey, delivery, deliveryText, sendOffer } from "./send";

function doc(points: Point[], side: "inside" | "outside" = "inside", ok = true): MeasurementDocument {
  return {
    schema: "spline-wire/measurement@1", name: "a.jpg", ok, error: ok ? null : "nope", view: null,
    info: { format: "JPEG", width: 1, height: 1, make: null, model: null, focal_mm: null, focal_35mm: null },
    curve: ok ? { points, pins: points, spline: points } : null,
    result: ok ? { object_side: side } as MeasurementDocument["result"] : null,
  };
}

const sent = (id: string, d: MeasurementDocument, delivered = false): SentCurve => ({ id, key: curveKey(d)!, at: "", delivered });
const item = (id: string, state: OutboxItem["state"]): OutboxItem => ({ curve: { id } as Curve, state });

describe("the send rule", () => {
  const first = doc([[0, 0], [10, 0]]);
  const fixed = doc([[0, 0], [10, 1]]);

  it("offers nothing unpaired or without a curve", () => {
    expect(sendOffer(false, first, [])).toEqual({ kind: "unpaired" });
    expect(sendOffer(true, null, [])).toEqual({ kind: "nothing" });
    expect(sendOffer(true, doc([], "inside", false), [])).toEqual({ kind: "nothing" });
  });

  it("offers a first send, then says it was sent", () => {
    expect(sendOffer(true, first, [])).toEqual({ kind: "send" });
    const s = sent("c1", first);
    expect(sendOffer(true, first, [s])).toEqual({ kind: "sent", curve: s });
  });

  it("offers a changed curve again as a new one, and recognises an undone change", () => {
    const s = sent("c1", first);
    expect(sendOffer(true, fixed, [s])).toEqual({ kind: "resend", previous: s });
    // The other side is another curve too.
    expect(sendOffer(true, doc([[0, 0], [10, 0]], "outside"), [s]).kind).toBe("resend");
    // Undo back to what was sent: nothing new to send.
    const s2 = sent("c2", fixed);
    expect(sendOffer(true, first, [s, s2])).toEqual({ kind: "sent", curve: s });
  });

  it("keys a curve by its clipboard table, so rounding below 0.0001 mm is the same curve", () => {
    expect(curveKey(doc([[0, 0], [10.00001, 0]]))).toBe(curveKey(first));
    expect(curveKey(doc([], "inside", false))).toBeNull();
  });

  it("says where a sent curve is", () => {
    const s = sent("c1", first);
    const open = { state: "open" as const, peers: 0 };
    expect(delivery(s, [item("c1", "queued")], open)).toBe("queued-no-computer");
    expect(delivery(s, [item("c1", "queued")], { state: "waiting", peers: 0 })).toBe("queued-offline");
    expect(delivery(s, [item("c1", "sent")], { state: "open", peers: 1 })).toBe("sent");
    expect(delivery(s, [], open)).toBe("delivered");
    expect(delivery({ ...s, delivered: true }, [item("c1", "queued")], open)).toBe("delivered");
    expect(deliveryText("queued-no-computer")).toMatch(/^Queued: the computer's page is not open/);
  });
});
