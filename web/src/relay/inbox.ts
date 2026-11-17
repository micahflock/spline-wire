// The desktop's side of delivery: store each curve once, acknowledge it every
// time it arrives (the phone repeats a curve until it hears the ack, and an
// ack can be lost too). Curves measured on the desktop itself, from a photo
// dropped on the page (D9), join the same list, marked as made there.

import type { RelayLink } from "./link";
import type { Curve, Message } from "./messages";
import { readJson, type KeyValueStore } from "./store";

/** Curves the desktop keeps. Older ones go, but their ids are remembered
 * (SEEN_CAP) so a late repeat is not shown as new. */
export const INBOX_CAP = 100;
export const SEEN_CAP = 1000;
const KEY = "inbox";
const SEEN_KEY = "inbox-seen";
/** Ids of the curves in the list that were measured on this computer. */
const HERE_KEY = "inbox-here";

export class Inbox {
  private list: Curve[];
  private seen: string[];
  private here: string[];

  constructor(
    private readonly store: KeyValueStore,
    private readonly link: Pick<RelayLink, "send">,
    /** Called with a curve the first time it arrives. */
    private readonly arrived: (curve: Curve) => void = () => {},
  ) {
    this.list = readJson<Curve[]>(store, KEY, []);
    this.seen = readJson<string[]>(store, SEEN_KEY, []);
    this.here = readJson<string[]>(store, HERE_KEY, []);
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

  /** A curve measured on this computer. Stored like one that arrived, but
   * nothing is acknowledged: no phone is waiting for it. */
  addLocal(curve: Curve): void {
    if (this.seen.includes(curve.id)) return;
    this.seen.push(curve.id);
    if (this.seen.length > SEEN_CAP) this.seen.splice(0, this.seen.length - SEEN_CAP);
    this.list.unshift(curve);
    if (this.list.length > INBOX_CAP) this.list.length = INBOX_CAP;
    this.here.unshift(curve.id);
    this.store.set(KEY, JSON.stringify(this.list));
    this.store.set(SEEN_KEY, JSON.stringify(this.seen));
    this.saveHere();
  }

  /** Whether the curve was measured on this computer rather than sent by a phone. */
  isLocal(id: string): boolean {
    return this.here.includes(id);
  }

  /** The person deleted a curve from the list. It stays "seen". */
  remove(id: string): void {
    this.list = this.list.filter((c) => c.id !== id);
    this.store.set(KEY, JSON.stringify(this.list));
    if (this.here.includes(id)) this.saveHere();
  }

  /** Only ids still in the list are kept, so the cap and deletes tidy this too. */
  private saveHere(): void {
    const ids = new Set(this.list.map((c) => c.id));
    this.here = this.here.filter((id) => ids.has(id));
    this.store.set(HERE_KEY, JSON.stringify(this.here));
  }
}
