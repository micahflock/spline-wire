// The desktop's side of delivery: store the highest revision of each curve,
// and say what it holds every time a curve arrives (the phone repeats a curve
// until it hears that, and a reply can be lost too). Stars, names and
// deletions are kept in step with the phone (sync.ts). Curves measured on the desktop itself,
// from a photo dropped on the page (D9), join the same list, marked as made
// there; no phone knows them, so their stars, names and deletions stay here.

import type { RelayLink } from "./link";
import { STATE_LIMITS, type Curve, type CurveState, type Message } from "./messages";
import { readJson, type KeyValueStore } from "./store";
import { cleanName, DeletedIds, laterName, laterStar, NO_NAME, NO_STAR, sameName, sameStar, stamp, type Name, type Star } from "./sync";

/** Curves the desktop keeps. Older ones go, but their revisions are remembered
 * (SEEN_CAP) so a late repeat is not shown as new. */
export const INBOX_CAP = 100;
export const SEEN_CAP = 1000;
const KEY = "inbox";
/** id → the highest revision received. */
const REVS_KEY = "inbox-revs";
/** Before revisions: the ids received, each revision 1. */
const SEEN_KEY = "inbox-seen";
/** Ids of the curves in the list that were measured on this computer. */
const HERE_KEY = "inbox-here";
const STARS_KEY = "inbox-stars";
const NAMES_KEY = "inbox-names";

export interface InboxEvents {
  /** A curve arrived: new, or a newer revision of one in the list (`updated`). */
  arrived(curve: Curve, updated: boolean): void;
  /** The phone deleted these. */
  removed(curves: Curve[]): void;
  /** The phone starred, unstarred or renamed these. */
  edited(ids: string[]): void;
}

export class Inbox {
  private list: Curve[];
  private revs: Record<string, number>;
  private here: string[];
  private stars: Record<string, Star>;
  private names: Record<string, Name>;
  private readonly deleted: DeletedIds;
  private readonly events: InboxEvents;

  constructor(
    private readonly store: KeyValueStore,
    private readonly link: Pick<RelayLink, "send">,
    events: Partial<InboxEvents> = {},
  ) {
    this.events = { arrived: () => {}, removed: () => {}, edited: () => {}, ...events };
    this.list = readJson<Curve[]>(store, KEY, []);
    this.revs = readJson<Record<string, number> | null>(store, REVS_KEY, null)
      ?? Object.fromEntries(readJson<string[]>(store, SEEN_KEY, []).map((id) => [id, 1]));
    this.here = readJson<string[]>(store, HERE_KEY, []);
    this.stars = readJson<Record<string, Star>>(store, STARS_KEY, {});
    this.names = readJson<Record<string, Name>>(store, NAMES_KEY, {});
    this.deleted = new DeletedIds(store, "inbox-deleted");
  }

  /** Received curves, newest first. */
  get curves(): readonly Curve[] {
    return this.list;
  }

  /** Wire this to the link's message event. */
  onMessage(message: Message): void {
    if (message.type === "curve") this.receive(message.curve);
    else if (message.type === "state") this.merge(message.curves, message.deleted);
  }

  /** What this desktop holds of the phones' curves: sent whenever a phone may have appeared. */
  fullState(): Message {
    const curves = this.list.filter((c) => !this.isLocal(c.id)).slice(0, STATE_LIMITS.curves).map((c) => this.stateOf(c.id));
    return { type: "state", full: true, curves, deleted: this.deleted.recent() };
  }

  /** A curve measured on this computer. Stored like one that arrived, but
   * nothing is said: no phone is waiting for it. */
  addLocal(curve: Curve): void {
    if (this.revs[curve.id]) return;
    this.revs[curve.id] = curve.rev ?? 1;
    this.list.unshift(curve);
    this.cap();
    this.here.unshift(curve.id);
    this.saveList();
    this.saveHere();
  }

  /** Whether the curve was measured on this computer rather than sent by a phone. */
  isLocal(id: string): boolean {
    return this.here.includes(id);
  }

  isStarred(id: string): boolean {
    return (this.stars[id] ?? NO_STAR).on;
  }

  /** The person starred or unstarred a curve here. */
  setStar(id: string, on: boolean, now = Date.now()): void {
    if (!this.list.some((c) => c.id === id)) return;
    // Later than anything seen for it, even if this computer's clock is behind the phone's.
    this.stars[id] = { on, at: stamp(this.stars[id] ?? NO_STAR, now) };
    this.saveStars();
    this.tell(id);
  }

  /** The person's name for a curve ("" when none): what the page shows in place of the photo's. */
  nameOf(id: string): string {
    return (this.names[id] ?? NO_NAME).text;
  }

  /** The person renamed a curve here ("" gives back the photo's name). */
  setName(id: string, text: string, now = Date.now()): void {
    if (!this.list.some((c) => c.id === id)) return;
    const name = cleanName(text);
    if (name === this.nameOf(id)) return;
    this.names[id] = { text: name, at: stamp(this.names[id] ?? NO_NAME, now) };
    this.saveStars();
    this.tell(id);
  }

  /** A change to one curve, to the phone at once (the next meeting repeats it if it is lost). */
  private tell(id: string): void {
    if (!this.isLocal(id)) void this.link.send({ type: "state", full: false, curves: [this.stateOf(id)], deleted: [] });
  }

  /** The person deleted a curve here: the phone deletes it too. */
  remove(id: string): void {
    const local = this.isLocal(id);
    this.drop([id]);
    if (!local) {
      this.deleted.add([id]);
      void this.link.send({ type: "state", full: false, curves: [], deleted: [id] });
    }
  }

  private receive(curve: Curve): void {
    const rev = curve.rev ?? 1;
    if (this.deleted.has(curve.id)) {
      // Deleted here while the phone was away: it deletes its copy on hearing this.
      void this.link.send({ type: "state", full: false, curves: [], deleted: [curve.id] });
      void this.link.send({ type: "ack", ids: [curve.id] });
      return;
    }
    if (rev > (this.revs[curve.id] ?? 0)) {
      this.revs[curve.id] = rev;
      const at = this.list.findIndex((c) => c.id === curve.id);
      if (at >= 0) this.list[at] = curve;
      else {
        this.list.unshift(curve);
        this.cap();
      }
      // Stored before the reply goes out: a curve the phone settles is never lost.
      this.saveList();
      try {
        this.events.arrived(curve, at >= 0);
      } finally {
        // Stored is stored: the phone hears so even if the page fails to show it.
        this.reply(curve.id);
      }
      return;
    }
    this.reply(curve.id);
  }

  private reply(id: string): void {
    void this.link.send({ type: "ack", ids: [id] });
    void this.link.send({ type: "state", full: false, curves: [this.stateOf(id)], deleted: [] });
  }

  private merge(curves: readonly CurveState[], deleted: readonly string[]): void {
    const gone = this.list.filter((c) => deleted.includes(c.id) && !this.isLocal(c.id));
    this.deleted.add(deleted);
    if (gone.length) {
      this.drop(gone.map((c) => c.id));
      this.events.removed(gone);
    }
    const edited: string[] = [];
    for (const s of curves) {
      if (!this.list.some((c) => c.id === s.id) || this.isLocal(s.id)) continue;
      const mine = this.stars[s.id] ?? NO_STAR;
      const won = laterStar(mine, { on: s.star, at: s.star_at });
      if (!sameStar(won, mine)) this.stars[s.id] = won;
      const myName = this.names[s.id] ?? NO_NAME;
      const named = laterName(myName, { text: cleanName(s.name ?? ""), at: s.name_at ?? 0 });
      if (!sameName(named, myName)) this.names[s.id] = named;
      if (!sameStar(won, mine) || !sameName(named, myName)) edited.push(s.id);
    }
    if (edited.length) {
      this.saveStars();
      this.events.edited(edited);
    }
  }

  private stateOf(id: string): CurveState {
    const star = this.stars[id] ?? NO_STAR, name = this.names[id] ?? NO_NAME;
    return { id, rev: this.revs[id] ?? 1, star: star.on, star_at: star.at, name: name.text, name_at: name.at };
  }

  private drop(ids: readonly string[]): void {
    this.list = this.list.filter((c) => !ids.includes(c.id));
    for (const id of ids) {
      delete this.stars[id];
      delete this.names[id];
    }
    this.saveList();
    this.saveStars();
    this.saveHere();
  }

  private cap(): void {
    if (this.list.length > INBOX_CAP) this.list.length = INBOX_CAP;
    const ids = Object.keys(this.revs);
    if (ids.length > SEEN_CAP) {
      // The oldest ids go first (ids are time-ordered); those in the list stay.
      const keep = new Set(this.list.map((c) => c.id));
      for (const id of ids.sort().slice(0, ids.length - SEEN_CAP)) if (!keep.has(id)) delete this.revs[id];
    }
  }

  private saveList(): void {
    this.store.set(KEY, JSON.stringify(this.list));
    this.store.set(REVS_KEY, JSON.stringify(this.revs));
  }

  /** The stars and the names, of curves still in the list. */
  private saveStars(): void {
    const ids = new Set(this.list.map((c) => c.id));
    for (const id of Object.keys(this.stars)) if (!ids.has(id)) delete this.stars[id];
    for (const id of Object.keys(this.names)) if (!ids.has(id)) delete this.names[id];
    this.store.set(STARS_KEY, JSON.stringify(this.stars));
    this.store.set(NAMES_KEY, JSON.stringify(this.names));
  }

  /** Only ids still in the list are kept, so the cap and deletes tidy this too. */
  private saveHere(): void {
    const ids = new Set(this.list.map((c) => c.id));
    this.here = this.here.filter((id) => ids.has(id));
    this.store.set(HERE_KEY, JSON.stringify(this.here));
  }
}
