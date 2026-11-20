// Phone to desktop through a relay, end to end: the delivery rules of
// outbox.ts and inbox.ts against the dull cases of web-build.md step 3.
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRelay } from "../../relay/memory-relay";
import { deriveKeys, newSecret, type Keys, type Role } from "./crypto";
import { Inbox } from "./inbox";
import { RelayLink, type LinkStatus } from "./link";
import { newCurveId, parseCurve, pointsTsv, type Curve, type Message } from "./messages";
import { Outbox } from "./outbox";
import { desktopSecret, pairingUrl, phoneSecret } from "./pairing";
import { MemoryStore } from "./store";

function curve(id = newCurveId()): Curve {
  return {
    v: 1, id, rev: 1, measured_at: "2026-11-18T08:00:00Z", photo: "IMG_0001.jpg",
    chain: { name: "O-ring chain", pitch_mm: 10, half_width_mm: 4, n_pins: 13 },
    points_mm: [[0, 0], [9.5, 1.25], [19, 3]], pins_mm: [[0, 4], [10, 5], [20, 7]],
    object_side: "inside", warnings: [], notes: ["ignored 2 look-alike detection(s) not on the chain"],
    diagnostics: { focal_source: "exif", focal_35mm: 26, tilt_deg: 12.5, missing_pins: 0,
      link_residual_rms_mm: 0.03, link_residual_max_mm: 0.06 },
  };
}

let relay: MemoryRelay;
let keys: Keys;

/** A page: a link with an outbox (phone) or an inbox (desktop) wired as the pages wire them. */
function page(role: Role, store = new MemoryStore()) {
  const arrived: Curve[] = [];
  let last: LinkStatus = { state: "off", peers: 0 };
  let outbox: Outbox | undefined;
  let inbox: Inbox | undefined;
  const link: RelayLink = new RelayLink({
    relayUrl: "ws://relay.test", keys, role, socket: relay.socket,
    events: {
      status(s) {
        outbox?.onStatus(s, last);
        last = s;
      },
      message(m: Message) {
        outbox?.onMessage(m);
        inbox?.onMessage(m);
      },
    },
  });
  if (role === "phone") outbox = new Outbox(store, link);
  else inbox = new Inbox(store, link, { arrived: (c) => arrived.push(c) });
  return { link, outbox: outbox!, inbox: inbox!, arrived, store };
}

const realSetTimeout = globalThis.setTimeout;

// Timers are faked, but WebCrypto finishes on real time: give both a few turns.
async function advance(ms: number): Promise<void> {
  const step = Math.max(50, ms / 40);
  for (let t = 0; t < ms; t += step) {
    await vi.advanceTimersByTimeAsync(step);
    await new Promise((r) => realSetTimeout(r, 3));
  }
}
const settle = () => advance(250);

beforeEach(async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  relay = new MemoryRelay();
  relay.now = () => Date.now();
  keys = await deriveKeys(newSecret());
});

describe("delivery", () => {
  it("delivers a curve, once, and the relay cannot read it", async () => {
    const phone = page("phone"), desktop = page("desktop");
    phone.link.start();
    desktop.link.start();
    await settle();
    expect(phone.link.current).toEqual({ state: "open", peers: 1 });
    const c = curve();
    await phone.outbox.put(c);
    await settle();
    expect(desktop.arrived).toEqual([c]);
    expect(phone.outbox.items).toEqual([]);
    expect(phone.outbox.delivered).toEqual([c.id]);
    expect(relay.forwarded.join("")).not.toContain("IMG_0001");
    expect(relay.forwarded.join("")).not.toContain(c.id);
  });

  it("queues while the desktop page is closed and delivers when it opens", async () => {
    const phone = page("phone");
    phone.link.start();
    await settle();
    const a = curve(), b = curve();
    await phone.outbox.put(a);
    await phone.outbox.put(b);
    expect(phone.outbox.items.map((i) => i.state)).toEqual(["queued", "queued"]);
    const desktop = page("desktop");
    desktop.link.start();
    await settle();
    expect(desktop.arrived.map((c) => c.id)).toEqual([a.id, b.id]);
    expect(phone.outbox.items).toEqual([]);
  });

  it("survives a reload of the phone page with curves still queued", async () => {
    const store = new MemoryStore();
    const before = page("phone", store);
    const c = curve();
    await before.outbox.put(c);                  // never connected
    const phone = page("phone", store), desktop = page("desktop");
    expect(phone.outbox.items.map((i) => i.curve.id)).toEqual([c.id]);
    phone.link.start();
    desktop.link.start();
    await settle();
    expect(desktop.arrived.map((x) => x.id)).toEqual([c.id]);
  });

  it("sends again after a lost reply, and the desktop shows the curve once", async () => {
    const phone = page("phone"), desktop = page("desktop");
    phone.link.start();
    desktop.link.start();
    await settle();
    let dropped = 0;
    // The desktop's replies (an ack and a state) to the curve, the first frame forwarded.
    relay.lose = () => relay.forwarded.length >= 1 && dropped++ < 2;
    const c = curve();
    await phone.outbox.put(c);
    await settle();
    expect(desktop.arrived.length).toBe(1);
    expect(phone.outbox.items.map((i) => i.state)).toEqual(["sent"]);       // no reply heard
    relay.cutAll();                                                          // both reconnect
    await advance(3000);
    expect(phone.outbox.items).toEqual([]);
    expect(desktop.arrived.length).toBe(1);
    expect(desktop.inbox.curves.length).toBe(1);
  });

  it("reconnects with backoff while the relay is down, and delivers after", async () => {
    const phone = page("phone"), desktop = page("desktop");
    relay.up = false;
    phone.link.start();
    desktop.link.start();
    await settle();
    expect(phone.link.current.state).toBe("waiting");
    await phone.outbox.put(curve());
    await advance(10_000);
    expect(phone.link.current.state).toBe("waiting");
    expect(phone.link.current.retryInMs).toBeGreaterThan(1000);              // backed off
    relay.up = true;
    await advance(40_000);
    expect(desktop.arrived.length).toBe(1);
    expect(phone.outbox.items).toEqual([]);
  });

  it("notices a dead connection by its missing pongs", async () => {
    const phone = page("phone");
    phone.link.start();
    await settle();
    const real = relay.fromPage.bind(relay);
    relay.fromPage = () => {};                                               // the network went silent
    relay.up = false;
    await advance(51_000);
    expect(phone.link.current.state).not.toBe("open");
    relay.fromPage = real;
  });

  it("keeps a second phone and a second pairing apart", async () => {
    const phone = page("phone"), phone2 = page("phone"), desktop = page("desktop");
    keys = await deriveKeys(newSecret());
    const stranger = page("desktop");
    for (const p of [phone, phone2, desktop, stranger]) p.link.start();
    await settle();
    const a = curve(), b = curve();
    await phone.outbox.put(a);
    await phone2.outbox.put(b);
    await settle();
    expect(desktop.arrived.map((c) => c.id).sort()).toEqual([a.id, b.id].sort());
    expect(stranger.arrived).toEqual([]);
    expect(phone.outbox.items).toEqual([]);
    expect(phone2.outbox.items).toEqual([]);
  });

  it("refuses a fifth tab in a room and says why", async () => {
    const pages = [1, 2, 3, 4, 5].map(() => page("desktop"));
    for (const p of pages) p.link.start();
    await settle();
    expect(pages.filter((p) => p.link.current.state === "open").length).toBe(4);
    expect(pages[4]!.link.current.problem).toMatch(/too many tabs/);
  });
});

describe("pairing", () => {
  it("goes from the desktop to the phone through the QR link's fragment", () => {
    const desk = new MemoryStore(), phone = new MemoryStore();
    const secret = desktopSecret(desk);
    expect(desktopSecret(desk)).toBe(secret);                                // made once
    const url = pairingUrl("https://example.org/spline-wire/phone/", secret);
    expect(new URL(url).search).toBe("");                                    // never sent to a server
    let cleared = false;
    expect(phoneSecret(phone, new URL(url).hash, () => (cleared = true))).toBe(secret);
    expect(cleared).toBe(true);
    expect(phoneSecret(phone, "", () => {})).toBe(secret);                   // remembered
    expect(phoneSecret(new MemoryStore(), "#pair=nonsense", () => {})).toBeNull();
  });
});

describe("messages", () => {
  it("checks a curve off the network and drops what it does not know", () => {
    const c = curve();
    expect(parseCurve({ ...c, extra: "from a newer build" })).toEqual(c);
    expect(parseCurve({ ...c, v: 2 })).toBeNull();
    expect(parseCurve({ ...c, points_mm: [[0, "1"]] })).toBeNull();
    expect(parseCurve({ ...c, points_mm: [[0, Infinity], [1, 1]] })).toBeNull();
    expect(parseCurve({ ...c, warnings: ["x".repeat(501)] })).toBeNull();
    expect(parseCurve({ ...c, id: "a b" })).toBeNull();
  });

  it("writes the clipboard table exactly as output.points_tsv does", () => {
    // From Python: points_tsv(np.array([[0, 0], [9.5, 1.25], [-0.00004, 123.45675], [1e-9, -2.5]]))
    expect(pointsTsv([[0, 0], [9.5, 1.25], [-0.00004, 123.45675], [1e-9, -2.5]])).toBe(
      "x_mm\ty_mm\n0.0000\t0.0000\n9.5000\t1.2500\n-0.0000\t123.4567\n0.0000\t-2.5000\n",
    );
  });

  it("adds a corner column to the clipboard table when the curve has sharp corners", () => {
    expect(pointsTsv([[0, 0], [10, 0], [10, 10]], [1])).toBe("x_mm\ty_mm\tcorner\n0.0000\t0.0000\t0\n10.0000\t0.0000\t1\n10.0000\t10.0000\t0\n");
    expect(pointsTsv([[0, 0], [10, 0]], [])).toBe(pointsTsv([[0, 0], [10, 0]]));
  });
});
