// The desktop's side of delivery: store each curve once, acknowledge it every
// time it arrives (the phone repeats a curve until it hears the ack, and an
// ack can be lost too).

import type { RelayLink } from "./link";
import type { Curve, Message } from "./messages";
import { readJson, type KeyValueStore } from "./store";

/** Curves the desktop keeps. Older ones go, but their ids are remembered
 * (SEEN_CAP) so a late repeat is not shown as new. */
export const INBOX_CAP = 100;
export const SEEN_CAP = 1000;
const KEY = "inbox";
const SEEN_KEY = "inbox-seen";

export class Inbox {
  private list: Curve[];
  private seen: string[];

  constructor(
    private readonly store: KeyValueStore,
    private readonly link: Pick<RelayLink, "send">,
    /** Called with a curve the first time it arrives. */
    private readonly arrived: (curve: Curve) => void = () => {},
  ) {
    this.list = readJson<Curve[]>(store, KEY, []);
    this.seen = readJson<string[]>(store, SEEN_KEY, []);
  }

  /** Received curves, newest first. */
  get curves(): readonly Curve[] {
    return this.list;
  }

  /** Wire this to the link's message event. */
  onMessage(message: Message): void {
    if (message.type !== "curve") return;
    const curve = message.curve;
    if (!this.seen.includes(curve.id)) {
      this.seen.push(curve.id);
      if (this.seen.length > SEEN_CAP) this.seen.splice(0, this.seen.length - SEEN_CAP);
      this.list.unshift(curve);
      if (this.list.length > INBOX_CAP) this.list.length = INBOX_CAP;
      // Stored before the ack goes out: an acknowledged curve is never lost.
      this.store.set(KEY, JSON.stringify(this.list));
      this.store.set(SEEN_KEY, JSON.stringify(this.seen));
      this.arrived(curve);
    }
    void this.link.send({ type: "ack", ids: [curve.id] });
  }

  /** The person deleted a curve from the list. It stays "seen". */
  remove(id: string): void {
    this.list = this.list.filter((c) => c.id !== id);
    this.store.set(KEY, JSON.stringify(this.list));
  }
}
