// How a photo's curve reaches the paired computer and stays in step with it
// (relay/sync.ts has the rules both pages follow).
//
// A photo taken while paired sends its curve by itself once it is measured
// (D26): one curve per photo, with one id for good. Each change to it after
// that (a pin edit, the object side) is a new revision of the same curve, and
// the computer replaces what it showed. The person looks at the pins on the
// phone and the computer follows; nothing piles up there.
//
// A photo from before the pairing (or before D26) is sent with a button, once;
// from then on it follows the same way. The computer and the phone keep each
// curve's star and name in step, and a curve deleted on one goes on the other.

import type { LinkStatus } from "../relay/link";
import type { CurveState, Message } from "../relay/messages";
import { newCurveId, pointsTsv } from "../relay/messages";
import type { OutboxItem } from "../relay/outbox";
import { cleanName, laterName, laterStar, NO_NAME, NO_STAR, sameName, sameStar, stamp, type Name, type Star } from "../relay/sync";
import type { MeasurementDocument } from "../core/types";
import type { PhotoRecord } from "./photos";

/** A photo's curve as the computer of one pairing has it. */
export interface CurveSync {
  id: string;
  /** The pairing's room (relay/crypto.ts deriveKeys), not its secret: another pairing starts afresh. */
  room: string;
  /** The latest revision made here. */
  rev: number;
  /** curveKey of that revision. */
  key: string;
  /** The highest revision the computer is known to hold (0: none yet). */
  remoteRev: number;
}

/** What identifies a curve's content: the clipboard table and the side. */
export function curveKey(doc: MeasurementDocument): string | null {
  if (!doc.ok || !doc.curve || !doc.result) return null;
  return `${doc.result.object_side}\n${pointsTsv(doc.curve.points)}`;
}

/** The photo's curve for this pairing, if it has been sent on it. Records from
 * before revisions (`sent`, a list of separate curves) count their last as revision 1. */
export function syncOf(r: PhotoRecord, room: string): CurveSync | null {
  if (r.sync) return r.sync.room === room ? r.sync : null;
  const last = r.sent?.at(-1);
  if (!last) return null;
  r.sync = { id: last.id, room, rev: 1, key: last.key, remoteRev: last.delivered ? 1 : 0 };
  return r.sync;
}

/** After a measurement: the revision to send, or null when there is nothing new
 * to send (not to be sent, or the curve the computer already has). */
export function nextRevision(r: PhotoRecord, room: string, newId: () => string = newCurveId): CurveSync | null {
  const key = r.doc && curveKey(r.doc);
  if (!key) return null;
  const s = syncOf(r, room);
  if (s) return s.key === key ? null : { ...s, rev: s.rev + 1, key };
  return r.autoSend ? { id: newId(), room, rev: 1, key, remoteRev: 0 } : null;
}

export type SendOffer =
  /** No pairing: the page says how to pair. */
  | { kind: "unpaired" }
  /** Nothing to send (no successful measurement, or one under way). */
  | { kind: "nothing" }
  /** Not sent on this pairing (a photo from before it): a button sends it. */
  | { kind: "send" }
  /** On its way or there; `delivery` says which. */
  | { kind: "synced"; sync: CurveSync };

export function sendOffer(room: string | null, r: PhotoRecord): SendOffer {
  if (!room) return { kind: "unpaired" };
  const s = syncOf(r, room);
  if (s) return { kind: "synced", sync: s };
  return r.doc && curveKey(r.doc) ? { kind: "send" } : { kind: "nothing" };
}

export type Delivery =
  /** In the outbox; the link is open but no computer is in the room. */
  | "queued-no-computer"
  /** In the outbox; no connection to the relay at the moment. */
  | "queued-offline"
  /** Handed to a computer; not confirmed yet. */
  | "sent"
  | "delivered";

/** Where a photo's curve is. One no longer in the outbox was settled (or
 * dropped by the outbox's cap, which only happens after 50 more). */
export function delivery(sync: CurveSync, outbox: readonly OutboxItem[], link: LinkStatus): Delivery {
  const item = outbox.find((i) => i.curve.id === sync.id);
  if (!item) return "delivered";
  if (item.state === "sent") return "sent";
  return link.state === "open" ? "queued-no-computer" : "queued-offline";
}

export function deliveryText(d: Delivery, rev = 1): string {
  const it = rev > 1 ? "The change" : "It";
  switch (d) {
    case "queued-no-computer":
      return `Queued: the computer's page is not open. ${it} goes when you open it.`;
    case "queued-offline":
      return `Queued: can't reach the relay. ${it} goes when the connection is back.`;
    case "sent":
      return "Sent, waiting for the computer to confirm.";
    case "delivered":
      return rev > 1 ? "Delivered to the computer, with your changes." : "Delivered to the computer.";
  }
}

export const starOf = (r: PhotoRecord): Star => r.star ?? NO_STAR;
export const nameOf = (r: PhotoRecord): Name => r.title ?? NO_NAME;
/** What the photo is called: the person's name for it, else its file name. */
export const displayName = (r: PhotoRecord): string => nameOf(r).text || r.name;

/** What this phone holds of its curves on this pairing (all of them with `full`, else `only`). */
export function phoneState(records: Iterable<PhotoRecord>, deleted: string[], room: string, full: boolean, only?: readonly string[]):
  Extract<Message, { type: "state" }> {
  const curves: CurveState[] = [];
  for (const r of records) {
    const s = syncOf(r, room);
    if (!s || (only && !only.includes(s.id))) continue;
    const star = starOf(r), name = nameOf(r);
    curves.push({ id: s.id, rev: s.rev, star: star.on, star_at: star.at, name: name.text, name_at: name.at });
  }
  return { type: "state", full, curves, deleted };
}

/** A state from the computer, applied to the photos: the revisions it holds,
 * the stars that win, and the photos whose curves it deleted (to delete here). */
export function applyState(records: Iterable<PhotoRecord>, m: Extract<Message, { type: "state" }>, room: string):
  { changed: PhotoRecord[]; deleted: PhotoRecord[] } {
  const byId = new Map<string, PhotoRecord>();
  for (const r of records) {
    const s = syncOf(r, room);
    if (s) byId.set(s.id, r);
  }
  const changed = new Set<PhotoRecord>(), deleted: PhotoRecord[] = [];
  for (const id of m.deleted) {
    const r = byId.get(id);
    if (r) deleted.push(r);
  }
  for (const c of m.curves) {
    const r = byId.get(c.id);
    if (!r || deleted.includes(r)) continue;
    const s = r.sync!;
    if (c.rev > s.remoteRev) {
      s.remoteRev = c.rev;
      changed.add(r);
    }
    const won = laterStar(starOf(r), { on: c.star, at: c.star_at });
    if (!sameStar(won, starOf(r))) {
      r.star = won;
      changed.add(r);
    }
    const named = laterName(nameOf(r), { text: cleanName(c.name ?? ""), at: c.name_at ?? 0 });
    if (!sameName(named, nameOf(r))) {
      r.title = named;
      changed.add(r);
    }
  }
  return { changed: [...changed], deleted };
}

/** The person starred or unstarred a photo here. */
export function setStar(r: PhotoRecord, on: boolean, now = Date.now()): void {
  r.star = { on, at: stamp(starOf(r), now) };
}

/** The person renamed a photo's curve here ("" gives back the file name). False if nothing changed. */
export function setName(r: PhotoRecord, text: string, now = Date.now()): boolean {
  const name = cleanName(text);
  if (name === nameOf(r).text) return false;
  r.title = { text: name, at: stamp(nameOf(r), now) };
  return true;
}
