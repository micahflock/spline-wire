// A phone and a desktop kept in step through a relay (sync.ts): revisions of
// a photo's curve, stars and deletions, with the pages open together or apart.
// The phone here is a model wired as phone/main.ts wires the page (send.ts and
// the outbox); the desktop is the real inbox, wired as desktop/main.ts wires it.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRelay } from "../../relay/memory-relay";
import type { MeasurementDocument, Point } from "../core/types";
import type { PhotoRecord } from "../phone/photos";
import { applyState, nextRevision, phoneState, setStar, starOf, syncOf } from "../phone/send";
import { deriveKeys, newSecret, type Keys } from "./crypto";
import { Inbox } from "./inbox";
import { RelayLink, type LinkStatus } from "./link";
import { type Curve, type Message } from "./messages";
import { Outbox } from "./outbox";
import { MemoryStore } from "./store";
import { DeletedIds, laterStar } from "./sync";

let relay: MemoryRelay;
let keys: Keys;

function doc(points: Point[], side: "inside" | "outside"): MeasurementDocument {
  return {
    schema: "spline-wire/measurement@1", name: "IMG_0001.jpg", ok: true, error: null, view: null,
    info: { format: "JPEG", width: 1, height: 1, make: null, model: null, focal_mm: null, focal_35mm: null },
    curve: { points, pins: points, spline: points },
    result: { object_side: side } as MeasurementDocument["result"],
  };
}

function curveOf(r: PhotoRecord, id: string, rev: number): Curve {
  return {
    v: 1, id, rev, measured_at: "2026-11-19T08:00:00Z", photo: r.name,
    chain: { name: "Snap-fit chain", pitch_mm: 13, half_width_mm: 4, n_pins: 13 },
    points_mm: r.doc!.curve!.points, pins_mm: r.doc!.curve!.pins, object_side: r.doc!.result!.object_side,
    warnings: [], notes: [],
    diagnostics: { focal_source: "exif", focal_35mm: 26, tilt_deg: 3, missing_pins: 0, link_residual_rms_mm: 0.02, link_residual_max_mm: 0.04 },
  };
}

/** The phone page: its photos, outbox and deleted ids, kept in `store` and `photos` between visits. */
function phonePage(store = new MemoryStore(), photos = new Map<string, PhotoRecord>()) {
  let last: LinkStatus = { state: "off", peers: 0 };
  const deleted = new DeletedIds(store, "deleted");
  const removedHere: string[] = [];
  const full = () => void link.send(phoneState(photos.values(), deleted.recent(), keys.room, true));
  const link: RelayLink = new RelayLink({
    relayUrl: "ws://relay.test", keys, role: "phone", socket: relay.socket,
    events: {
      status(s) {
        outbox.onStatus(s, last);
        if (s.state === "open" && (last.state !== "open" || s.peers > last.peers)) full();
        last = s;
      },
      message(m: Message) {
        outbox.onMessage(m);
        if (m.type === "hello" && m.role === "desktop") full();
        if (m.type !== "state") return;
        for (const r of applyState(photos.values(), m, keys.room).deleted) {
          deleted.add([r.sync!.id]);
          outbox.forget(r.sync!.id);
          photos.delete(r.id);
          removedHere.push(r.id);
        }
      },
    },
  });
  const outbox = new Outbox(store, link);
  let n = 0;
  return {
    link, outbox, photos, store, removedHere,
    /** A photo taken while paired, measured. */
    async take(points: Point[], side: "inside" | "outside" = "outside"): Promise<PhotoRecord> {
      const r = { id: `p${++n}`, name: `IMG_000${n}.jpg`, autoSend: true, doc: doc(points, side) } as unknown as PhotoRecord;
      photos.set(r.id, r);
      await this.measured(r);
      return r;
    },
    /** The photo measured again (an edit, the side): its curve goes if it changed. */
    async measured(r: PhotoRecord, d?: MeasurementDocument): Promise<void> {
      if (d) r.doc = d;
      const next = nextRevision(r, keys.room);
      if (!next) return;
      r.sync = next;
      await outbox.put(curveOf(r, next.id, next.rev));
    },
    star(r: PhotoRecord, on: boolean): void {
      setStar(r, on);
      void link.send(phoneState(photos.values(), [], keys.room, false, [r.sync!.id]));
    },
    remove(r: PhotoRecord): void {
      deleted.add([r.sync!.id]);
      outbox.forget(r.sync!.id);
      photos.delete(r.id);
      void link.send(phoneState(photos.values(), [r.sync!.id], keys.room, false, []));
    },
  };
}

function desktopPage(store = new MemoryStore()) {
  const events: string[] = [];
  let inbox: Inbox;
  const link: RelayLink = new RelayLink({
    relayUrl: "ws://relay.test", keys, role: "desktop", socket: relay.socket,
    events: {
      status() {},
      message(m: Message) {
        inbox.onMessage(m);
        if (m.type === "state" && m.full) void link.send(inbox.fullState());
      },
    },
  });
  inbox = new Inbox(store, link, {
    arrived: (c, updated) => events.push(`${updated ? "updated" : "new"} ${c.id} rev ${c.rev} ${c.object_side}`),
    removed: (cs) => events.push(...cs.map((c) => `removed ${c.id}`)),
    starred: (ids) => events.push(...ids.map((id) => `starred ${id}`)),
  });
  return { link, inbox, events, store };
}

const realSetTimeout = globalThis.setTimeout;
async function advance(ms: number): Promise<void> {
  const step = Math.max(50, ms / 40);
  for (let t = 0; t < ms; t += step) {
    await vi.advanceTimersByTimeAsync(step);
    await new Promise((r) => realSetTimeout(r, 3));
  }
}
const settle = () => advance(400);

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  relay = new MemoryRelay();
  relay.now = () => Date.now();
  keys = await deriveKeys(newSecret());
});

const line: Point[] = [[0, 0], [13, 1], [26, 3]];

describe("a photo's curve", () => {
  it("goes by itself once measured, and a change replaces it on the desktop", async () => {
    const phone = phonePage(), desktop = desktopPage();
    phone.link.start();
    desktop.link.start();
    await settle();
    const r = await phone.take(line, "outside");
    await settle();
    const id = r.sync!.id;
    expect(desktop.inbox.curves.map((c) => [c.id, c.rev, c.object_side])).toEqual([[id, 1, "outside"]]);
    await phone.measured(r, doc(line, "inside"));               // the person chose Inside
    await settle();
    expect(desktop.inbox.curves.map((c) => [c.id, c.rev, c.object_side])).toEqual([[id, 2, "inside"]]);
    expect(desktop.events).toEqual([`new ${id} rev 1 outside`, `updated ${id} rev 2 inside`]);
    expect(phone.outbox.items).toEqual([]);
    expect(r.sync!.remoteRev).toBe(2);                           // the desktop's state says which revision it holds
    // A measurement that gives the same curve again sends nothing.
    await phone.measured(r, doc(line, "inside"));
    expect(phone.outbox.items).toEqual([]);
  });

  it("sends only the latest change made while the desktop was closed, and an old revision never wins", async () => {
    const phone = phonePage();
    phone.link.start();
    await settle();
    const r = await phone.take(line, "outside");
    await phone.measured(r, doc(line, "inside"));
    await phone.measured(r, doc([[0, 0], [13, 2]], "inside"));
    expect(phone.outbox.items.map((i) => i.curve.rev)).toEqual([3]);
    const desktop = desktopPage();
    desktop.link.start();
    await settle();
    expect(desktop.inbox.curves.map((c) => c.rev)).toEqual([3]);
    expect(desktop.events).toEqual([`new ${r.sync!.id} rev 3 inside`]);
    // A stale copy of revision 1 (another tab, a slow network) changes nothing.
    desktop.inbox.onMessage({ type: "curve", curve: curveOf({ ...r, doc: doc(line, "outside") }, r.sync!.id, 1) });
    expect(desktop.inbox.curves.map((c) => [c.rev, c.object_side])).toEqual([[3, "inside"]]);
  });

  it("a photo from before the pairing is not sent until the person sends it", () => {
    const r = { id: "p1", name: "a.jpg", doc: doc(line, "inside") } as unknown as PhotoRecord;
    expect(nextRevision(r, "room")).toBeNull();
    r.autoSend = true;
    expect(nextRevision(r, "room", () => "c-000000001")).toMatchObject({ id: "c-000000001", rev: 1, remoteRev: 0 });
  });

  it("counts a curve sent before revisions as revision 1 of the photo's curve", () => {
    const d = doc(line, "inside");
    const r = { id: "p1", name: "a.jpg", doc: d, sent: [{ id: "c-old000001", key: "", at: "", delivered: true }] } as unknown as PhotoRecord;
    expect(syncOf(r, "room")).toMatchObject({ id: "c-old000001", rev: 1, remoteRev: 1 });
    expect(nextRevision(r, "room")).toMatchObject({ id: "c-old000001", rev: 2 });
    expect(syncOf(r, "another room")).toBeNull();                // another pairing starts afresh
  });
});

describe("stars", () => {
  it("go both ways while both pages are open", async () => {
    const phone = phonePage(), desktop = desktopPage();
    phone.link.start();
    desktop.link.start();
    await settle();
    const r = await phone.take(line);
    await settle();
    phone.star(r, true);
    await settle();
    expect(desktop.inbox.isStarred(r.sync!.id)).toBe(true);
    desktop.inbox.setStar(r.sync!.id, false);
    await settle();
    expect(starOf(r).on).toBe(false);
  });

  it("set apart, meet: the later setting wins on both", async () => {
    const pStore = new MemoryStore(), photos = new Map<string, PhotoRecord>();
    let phone = phonePage(pStore, photos);
    const desktop = desktopPage();
    phone.link.start();
    desktop.link.start();
    await settle();
    const a = await phone.take(line), b = await phone.take(line);
    await settle();
    phone.link.stop();                                           // the phone's page closed
    await advance(1000);
    desktop.inbox.setStar(a.sync!.id, true);                     // starred on the desktop meanwhile
    await advance(1000);
    setStar(b, true);                                            // and b starred on the phone, offline
    phone = phonePage(pStore, photos);
    phone.link.start();
    await settle();
    expect(starOf(a).on).toBe(true);
    expect(desktop.inbox.isStarred(b.sync!.id)).toBe(true);
    // Unstarred on both while apart: the later one wins on both.
    phone.link.stop();
    await advance(1000);
    desktop.inbox.setStar(a.sync!.id, false);
    await advance(1000);
    setStar(a, true);
    phone = phonePage(pStore, photos);
    phone.link.start();
    await settle();
    expect(starOf(a).on).toBe(true);
    expect(desktop.inbox.isStarred(a.sync!.id)).toBe(true);
  });

  it("merge the same in any order", () => {
    const x = { on: true, at: 5 }, y = { on: false, at: 9 }, tie = { on: false, at: 5 };
    expect(laterStar(x, y)).toBe(y);
    expect(laterStar(y, x)).toBe(y);
    expect(laterStar(x, tie)).toBe(x);
    expect(laterStar(tie, x)).toBe(x);
  });
});

describe("deleting", () => {
  it("on the desktop deletes the photo on the phone, now or when it next opens", async () => {
    const pStore = new MemoryStore(), photos = new Map<string, PhotoRecord>();
    let phone = phonePage(pStore, photos);
    const desktop = desktopPage();
    phone.link.start();
    desktop.link.start();
    await settle();
    const a = await phone.take(line), b = await phone.take(line);
    await settle();
    desktop.inbox.remove(a.sync!.id);
    await settle();
    expect(phone.removedHere).toEqual([a.id]);
    phone.link.stop();
    desktop.inbox.remove(b.sync!.id);                            // the phone's page is closed
    await settle();
    phone = phonePage(pStore, photos);
    phone.link.start();
    await settle();
    expect(phone.removedHere).toEqual([b.id]);
    expect([...photos.keys()]).toEqual([]);
  });

  it("on the phone deletes the curve on the desktop, now or when it next opens", async () => {
    const dStore = new MemoryStore();
    const phone = phonePage();
    let desktop = desktopPage(dStore);
    phone.link.start();
    desktop.link.start();
    await settle();
    const a = await phone.take(line), b = await phone.take(line);
    await settle();
    phone.remove(a);
    await settle();
    expect(desktop.inbox.curves.map((c) => c.id)).toEqual([b.sync!.id]);
    expect(desktop.events).toContain(`removed ${a.sync!.id}`);
    desktop.link.stop();
    phone.remove(b);                                             // the desktop's page is closed
    await settle();
    desktop = desktopPage(dStore);
    desktop.link.start();
    await settle();
    expect(desktop.inbox.curves).toEqual([]);
    expect(desktop.events).toEqual([`removed ${b.sync!.id}`]);
  });

  it("wins over a change made on the other side before it heard", async () => {
    const pStore = new MemoryStore(), photos = new Map<string, PhotoRecord>();
    let phone = phonePage(pStore, photos);
    const desktop = desktopPage();
    phone.link.start();
    desktop.link.start();
    await settle();
    const r = await phone.take(line, "outside");
    await settle();
    phone.link.stop();
    desktop.inbox.remove(r.sync!.id);
    await phone.measured(r, doc(line, "inside"));                // changed on the phone, not yet heard of the delete
    phone = phonePage(pStore, photos);
    phone.link.start();
    await settle();
    expect(desktop.inbox.curves).toEqual([]);                    // not taken back in
    expect(phone.removedHere).toEqual([r.id]);
    expect(phone.outbox.items).toEqual([]);
  });

  it("of a curve measured on the desktop says nothing to the phone", async () => {
    const desktop = desktopPage();
    const sent: Message[] = [];
    const box = new Inbox(new MemoryStore(), { send: async (m) => (sent.push(m), true) });
    const here = curveOf({ name: "here.jpg", doc: doc(line, "inside") } as PhotoRecord, "c-here00001", 1);
    box.addLocal(here);
    box.setStar(here.id, true);
    box.remove(here.id);
    expect(sent).toEqual([]);
    expect(desktop.inbox.fullState()).toEqual({ type: "state", full: true, curves: [], deleted: [] });
  });
});
