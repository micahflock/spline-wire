// The phone's side of delivery: every curve sent to the desktop stays here
// until the desktop acknowledges it.
//
// Rules, which make delivery at-least-once and the desktop's inbox makes it
// exactly-once:
//   - send(curve) stores first, then tries the link;
//   - everything unacknowledged is sent again whenever a desktop might have
//     appeared: the link opened, the room's peer count rose, a hello arrived;
//   - an ack removes its curves. Nothing else does, except the cap below and
//     the person (forget).
//
// The page shows `items` so the person knows what has and has not arrived
// ("queued: the computer's page is not open").

import type { RelayLink, LinkStatus } from "./link";
import type { Curve, Message } from "./messages";
import { readJson, type KeyValueStore } from "./store";

export interface OutboxItem {
  curve: Curve;
  /** queued: not yet handed to an open link with a peer in the room. sent: handed over, no ack yet. */
  state: "queued" | "sent";
}

/** Unacknowledged curves kept. Beyond it the oldest are dropped: a phone that
 * never meets its desktop must not fill its storage. */
export const OUTBOX_CAP = 50;
const KEY = "outbox";

export class Outbox {
  private list: OutboxItem[];
  /** Ids delivered during this visit, newest last, for the page's "delivered" marks. */
  readonly delivered: string[] = [];

  constructor(
    private readonly store: KeyValueStore,
    private readonly link: Pick<RelayLink, "send" | "current">,
    private readonly changed: () => void = () => {},
  ) {
    // After a reload nothing is "sent": no ack came, so all of it goes again.
    this.list = readJson<Curve[]>(store, KEY, []).map((curve) => ({ curve, state: "queued" }));
  }

  get items(): readonly OutboxItem[] {
    return this.list;
  }

  /** Queue a curve and try to send it. */
  async send(curve: Curve): Promise<void> {
    if (this.list.some((i) => i.curve.id === curve.id)) return;
    this.list.push({ curve, state: "queued" });
    if (this.list.length > OUTBOX_CAP) this.list.splice(0, this.list.length - OUTBOX_CAP);
    this.save();
    await this.flush();
  }

  /** The person gave up on a curve. */
  forget(id: string): void {
    this.list = this.list.filter((i) => i.curve.id !== id);
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
    if (message.type !== "ack") return;
    const acked = new Set(message.ids);
    const before = this.list.length;
    for (const i of this.list) if (acked.has(i.curve.id)) this.delivered.push(i.curve.id);
    this.list = this.list.filter((i) => !acked.has(i.curve.id));
    if (this.list.length !== before) this.save();
  }

  /** Send what is queued; with `again`, also what was sent and not yet acknowledged. */
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
