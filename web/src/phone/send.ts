// When a measured curve goes to the paired computer.
//
// The person sends it with a button; nothing goes out by itself. A curve
// sent straight after measuring would reach the computer before the person
// had looked at the pins, and the photo's first measurement is the one most
// likely to need a fix. A button costs one tap and keeps the computer's list
// to curves the person meant.
//
// After a curve is sent the person may still fix the photo. The curve then
// differs from what the computer has, and the button offers to send it again:
// as a new curve with a new id (the computer stores each id once, so reusing
// it would be dropped there), and the page says so. A re-measure that gives
// the same curve (an edit undone) is recognised by its key and not offered.

import type { LinkStatus } from "../relay/link";
import type { OutboxItem } from "../relay/outbox";
import { pointsTsv } from "../relay/messages";
import type { MeasurementDocument } from "../core/types";
import type { SentCurve } from "./photos";

/** What identifies a curve's content: the clipboard table and the side. */
export function curveKey(doc: MeasurementDocument): string | null {
  if (!doc.ok || !doc.curve || !doc.result) return null;
  return `${doc.result.object_side}\n${pointsTsv(doc.curve.points)}`;
}

export type SendOffer =
  /** No pairing: the page says how to pair. */
  | { kind: "unpaired" }
  /** Nothing to send (no successful measurement, or one under way). */
  | { kind: "nothing" }
  /** Never sent from this photo. */
  | { kind: "send" }
  /** This very curve was sent. */
  | { kind: "sent"; curve: SentCurve }
  /** An earlier curve of this photo was sent; this one differs. */
  | { kind: "resend"; previous: SentCurve };

export function sendOffer(paired: boolean, doc: MeasurementDocument | null, sent: readonly SentCurve[]): SendOffer {
  if (!paired) return { kind: "unpaired" };
  const key = doc && curveKey(doc);
  if (!key) return { kind: "nothing" };
  const same = sent.find((s) => s.key === key);
  if (same) return { kind: "sent", curve: same };
  const last = sent.at(-1);
  return last ? { kind: "resend", previous: last } : { kind: "send" };
}

export type Delivery =
  /** In the outbox; the link is open but no computer is in the room. */
  | "queued-no-computer"
  /** In the outbox; no connection to the relay at the moment. */
  | "queued-offline"
  /** Handed to a computer; no acknowledgement yet. */
  | "sent"
  | "delivered";

/** Where a sent curve is. A curve no longer in the outbox was acknowledged
 * (or dropped by the outbox's cap, which only happens after 50 more). */
export function delivery(curve: SentCurve, outbox: readonly OutboxItem[], link: LinkStatus): Delivery {
  if (curve.delivered) return "delivered";
  const item = outbox.find((i) => i.curve.id === curve.id);
  if (!item) return "delivered";
  if (item.state === "sent") return "sent";
  return link.state === "open" ? "queued-no-computer" : "queued-offline";
}

export function deliveryText(d: Delivery): string {
  switch (d) {
    case "queued-no-computer":
      return "Queued: the computer's page is not open. It goes when you open it.";
    case "queued-offline":
      return "Queued: can't reach the relay. It goes when the connection is back.";
    case "sent":
      return "Sent, waiting for the computer to confirm.";
    case "delivered":
      return "Delivered to the computer.";
  }
}
