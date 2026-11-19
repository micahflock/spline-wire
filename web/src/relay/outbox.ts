// The phone's side of delivery: the latest revision of each curve stays here
// until a state from the desktop shows that revision (sync.ts).
//
// Rules, which make delivery at-least-once and the desktop's inbox makes it
// exactly-once (it keeps the highest revision of each curve):
//   - put(curve) stores first, then tries the link. A newer revision of a
//     curve replaces the older one here: only the latest is worth sending;
//   - everything unsettled is sent again whenever a desktop might have
//     appeared: the link opened, the room's peer count rose, a hello arrived;
//   - a state from the desktop settles each curve whose revision it holds
//     (an ack from an older desktop settles the curve as it is). Nothing else
//     removes a curve, except the cap below and the person (forget).
//
// The page shows `items` so the person knows what has and has not arrived
// ("queued: the computer's page is not open").

import type { RelayLink, LinkStatus } from "./link";
import type { Curve, Message } from "./messages";
import { readJson, type KeyValueStore } from "./store";

export interface OutboxItem {
  curve: Curve;
  /** queued: not yet handed to an open link with a peer in the room. sent: handed over, not settled yet. */
  state: "queued" | "sent";
}

/** Unsettled curves kept. Beyond it the oldest are dropped: a phone that
 * never meets its desktop must not fill its storage. */
export const OUTBOX_CAP = 50;
const KEY = "outbox";

export const revOf = (c: Curve): number => c.rev ?? 1;

export class Outbox {
  private list: OutboxItem[];
  /** Ids settled during this visit, newest last, for the page's "delivered" marks. */
  readonly delivered: string[] = [];

  constructor(
    private readonly store: KeyValueStore,
    private readonly link: Pick<RelayLink, "send" | "current">,
    private readonly changed: () => void = () => {},
  ) {
    // After a reload nothing is "sent": nothing settled it, so all of it goes again.
    this.list = readJson<Curve[]>(store, KEY, []).map((curve) => ({ curve, state: "queued" }));
  }

  get items(): readonly OutboxItem[] {
    return this.list;
  }

  /** Queue a curve, or a newer revision of one queued, and try to send it. */
  async put(curve: Curve): Promise<void> {
    const at = this.list.findIndex((i) => i.curve.id === curve.id);
    if (at >= 0) {
      if (revOf(this.list[at]!.curve) >= revOf(curve)) return;
      this.list.splice(at, 1);
    }
    this.list.push({ curve, state: "queued" });
    if (this.list.length > OUTBOX_CAP) this.list.splice(0, this.list.length - OUTBOX_CAP);
    this.save();
    await this.flush();
  }

  /** The person gave up on a curve, or deleted its photo. */
  forget(id: string): void {
    const before = this.list.length;
    this.list = this.list.filter((i) => i.curve.id !== id);
    if (this.list.length !== before) this.save();
  }

  /** The desktop holds revision `rev` of curve `id`: an older or equal one here is done. */
  settle(id: string, rev: number): void {
    const item = this.list.find((i) => i.curve.id === id);
    if (!item || revOf(item.curve) > rev) return;
    this.delivered.push(id);
    this.list = this.list.filter((i) => i !== item);
    this.save();
  }

  /** Wire these to the link's events. */
  onStatus(status: LinkStatus, previous: LinkStatus): void {
    if (status.state !== "open") {
      // Whatever was in flight on the old connection may not have arrived.
      for (const i of this.list) i.state = "queued";
      this.changed();
    } else if (status.peers > previous.peers || previous.state !== "open") {
      void this.flush(true);
    }
  }

  onMessage(message: Message): void {
    if (message.type === "hello" && message.role === "desktop") void this.flush(true);
    else if (message.type === "state") for (const c of message.curves) this.settle(c.id, c.rev);
    else if (message.type === "ack") {
      // A desktop from before `state`: it stored the curve (and, keeping one copy
      // an id, may not take a later revision). Settled as it is.
      for (const id of message.ids) {
        const item = this.list.find((i) => i.curve.id === id);
        if (item) this.settle(id, revOf(item.curve));
      }
    }
  }

  /** Send what is queued; with `again`, also what was sent and not yet settled. */
  private async flush(again = false): Promise<void> {
    const status = this.link.current;
    if (status.state !== "open" || status.peers < 1) return;
    for (const item of [...this.list]) {
      if (item.state === "sent" && !again) continue;
      if (await this.link.send({ type: "curve", curve: item.curve })) item.state = "sent";
    }
    this.changed();
  }

  private save(): void {
    this.store.set(KEY, JSON.stringify(this.list.map((i) => i.curve)));
    this.changed();
  }
}
