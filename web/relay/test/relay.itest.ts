// The client backbone (src/relay/) through a real relay over real sockets:
// the important cases of src/relay/delivery.test.ts, plus the relay's limits
// as a page sees them. Runs against the dev relay, or with RELAY_TARGET=worker
// against worker.ts under `wrangler dev`.
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import WebSocketNode from "ws";
import { deriveKeys, newSecret, type Keys, type Role } from "../../src/relay/crypto.ts";
import { Inbox } from "../../src/relay/inbox.ts";
import { RelayLink, type LinkStatus } from "../../src/relay/link.ts";
import { newCurveId, type Curve, type Message } from "../../src/relay/messages.ts";
import { Outbox } from "../../src/relay/outbox.ts";
import { MemoryStore } from "../../src/relay/store.ts";
import { CLOSE, MAX_FRAME_CHARS, RATE_LIMIT, roomPath } from "../rules.ts";
import { makeTarget, targetName, type Target } from "./targets.ts";

const slow = targetName === "worker" ? 4 : 1;

function curve(id = newCurveId()): Curve {
  return {
    v: 1, id, rev: 1, measured_at: "2026-11-18T08:00:00Z", photo: "IMG_0001.jpg",
    chain: { name: "O-ring chain", pitch_mm: 10, half_width_mm: 4, n_pins: 13 },
    points_mm: [[0, 0], [9.5, 1.25], [19, 3]], pins_mm: [[0, 4], [10, 5], [20, 7]],
    object_side: "inside", warnings: [], notes: [],
    diagnostics: { focal_source: "exif", focal_35mm: 26, tilt_deg: 12.5, missing_pins: 0,
      link_residual_rms_mm: 0.03, link_residual_max_mm: 0.06 },
  };
}

let target: Target;
let keys: Keys;
const running: RelayLink[] = [];

/** A page as the pages wire it: a link with an outbox (phone) or an inbox (desktop). */
function page(role: Role) {
  const arrived: Curve[] = [];
  let last: LinkStatus = { state: "off", peers: 0 };
  let outbox: Outbox | undefined;
  let inbox: Inbox | undefined;
  const link: RelayLink = new RelayLink({
    relayUrl: target.url, keys, role,
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
  running.push(link);
  if (role === "phone") outbox = new Outbox(new MemoryStore(), link);
  else inbox = new Inbox(new MemoryStore(), link, { arrived: (c) => arrived.push(c) });
  return { link, outbox: outbox!, inbox: inbox!, arrived };
}

async function until(what: string, cond: () => boolean, ms = 10_000 * slow): Promise<void> {
  const end = Date.now() + ms;
  while (!cond()) {
    if (Date.now() > end) throw new Error(`timed out waiting for: ${what}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}
const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A bare socket, for the limits. */
function raw(room = keys.room) {
  const ws = new WebSocket(target.url + roomPath(room));
  const got: string[] = [];
  let closed: { code: number; reason: string } | null = null;
  let opened = false;
  ws.onopen = () => (opened = true);
  ws.onmessage = (ev) => got.push(String(ev.data));
  ws.onclose = (ev) => (closed = { code: ev.code, reason: ev.reason });
  return { ws, got, closed: () => closed, opened: () => opened };
}

beforeAll(async () => {
  target = await makeTarget();
  await target.start();
}, 90_000);

afterAll(async () => {
  await target.stop();
}, 30_000);

afterEach(async () => {
  for (const l of running.splice(0)) l.stop();
});

describe(`delivery through the ${targetName} relay`, () => {
  beforeAll(async () => {
    keys = await deriveKeys(newSecret());
  });

  it("delivers a curve once and the reply clears the phone's outbox", async () => {
    keys = await deriveKeys(newSecret());
    const phone = page("phone"), desktop = page("desktop");
    phone.link.start();
    desktop.link.start();
    await until("both open with a peer", () => phone.link.current.peers === 1 && desktop.link.current.peers === 1);
    const c = curve();
    await phone.outbox.put(c);
    await until("delivered", () => phone.outbox.items.length === 0);
    expect(desktop.arrived).toEqual([c]);
    expect(phone.outbox.delivered).toEqual([c.id]);
  });

  it("queues while the desktop page is closed and delivers when it opens", async () => {
    keys = await deriveKeys(newSecret());
    const phone = page("phone");
    phone.link.start();
    await until("phone open", () => phone.link.current.state === "open");
    const a = curve(), b = curve();
    await phone.outbox.put(a);
    await phone.outbox.put(b);
    await pause(200);
    expect(phone.outbox.items.map((i) => i.state)).toEqual(["queued", "queued"]);
    const desktop = page("desktop");
    desktop.link.start();
    await until("both delivered", () => phone.outbox.items.length === 0);
    expect(desktop.arrived.map((c) => c.id)).toEqual([a.id, b.id]);
  });

  it("survives the relay restarting mid-session: both reconnect, the queued curve arrives once", async () => {
    keys = await deriveKeys(newSecret());
    const phone = page("phone"), desktop = page("desktop");
    phone.link.start();
    desktop.link.start();
    await until("paired", () => phone.link.current.peers === 1 && desktop.link.current.peers === 1);
    const first = curve();
    await phone.outbox.put(first);
    await until("first delivered", () => phone.outbox.items.length === 0);

    await target.stop();
    await until("both lost the relay", () =>
      phone.link.current.state !== "open" && desktop.link.current.state !== "open");
    const c = curve();
    await phone.outbox.put(c);
    expect(phone.outbox.items.map((i) => i.state)).toEqual(["queued"]);

    await target.start();
    await until("delivered after the restart", () => phone.outbox.items.length === 0, 30_000 * slow);
    await pause(500);
    expect(desktop.arrived.map((x) => x.id)).toEqual([first.id, c.id]);
    expect(desktop.inbox.curves.length).toBe(2);
    expect(phone.link.current).toMatchObject({ state: "open", peers: 1 });
    expect(desktop.link.current).toMatchObject({ state: "open", peers: 1 });
  }, 90_000);

  // A Durable Object hibernates after 10 s without events; pings are answered
  // without waking it. After that its memory is gone and only the sockets and
  // their attachments remain. Whether workerd hibernated is not observable
  // from outside; the idle gap is long enough that it may.
  it.runIf(targetName === "worker")("keeps a quiet room working after an idle spell", async () => {
    keys = await deriveKeys(newSecret());
    const phone = page("phone"), desktop = page("desktop");
    phone.link.start();
    desktop.link.start();
    await until("paired", () => phone.link.current.peers === 1 && desktop.link.current.peers === 1);
    const a = raw();
    await until("joined", () => phone.link.current.peers === 2);
    a.ws.send("?");
    await pause(15_000);
    a.ws.close();
    await until("the room shrank", () => phone.link.current.peers === 1 && desktop.link.current.peers === 1);
    await phone.outbox.put(curve());
    await until("delivered", () => phone.outbox.items.length === 0);
    expect(desktop.arrived.length).toBe(1);
  }, 60_000);

  it("keeps two pairings apart", async () => {
    keys = await deriveKeys(newSecret());
    const phone = page("phone"), desktop = page("desktop");
    keys = await deriveKeys(newSecret());
    const stranger = page("desktop");
    for (const p of [phone, desktop, stranger]) p.link.start();
    await until("paired", () => phone.link.current.peers === 1 && stranger.link.current.state === "open");
    await phone.outbox.put(curve());
    await until("delivered", () => phone.outbox.items.length === 0);
    await pause(200);
    expect(desktop.arrived.length).toBe(1);
    expect(stranger.arrived).toEqual([]);
    expect(stranger.link.current.peers).toBe(0);
  });
});

describe(`the ${targetName} relay's limits`, () => {
  it("refuses a fifth socket in a room with 4001, and a link says why", async () => {
    keys = await deriveKeys(newSecret());
    const four = [1, 2, 3, 4].map(() => raw());
    await until("four open", () => four.every((s) => s.opened()));
    await until("the last hears four peers", () => four[3]!.got.includes('!{"peers":4}'));
    const fifth = raw();
    await until("fifth closed", () => fifth.closed() !== null);
    expect(fifth.closed()!.code).toBe(CLOSE.roomFull);
    const tab = page("desktop");
    tab.link.start();
    await until("link refused", () => tab.link.current.problem !== undefined);
    expect(tab.link.current.problem).toMatch(/too many tabs/);
    expect(four.every((s) => s.closed() === null)).toBe(true);
    for (const s of four) s.ws.close();
  });

  it("closes a socket that sends an oversize frame with 4003, and tells the room", async () => {
    keys = await deriveKeys(newSecret());
    const a = raw(), b = raw();
    await until("both open", () => a.opened() && b.opened() && b.got.includes('!{"peers":2}'));
    a.ws.send("{" + "x".repeat(MAX_FRAME_CHARS));
    await until("closed", () => a.closed() !== null);
    expect(a.closed()!.code).toBe(CLOSE.tooBig);
    await until("the other hears it", () => b.got.at(-1) === '!{"peers":1}');
    expect(b.got.some((f) => f.startsWith("{"))).toBe(false);          // not forwarded
    b.ws.close();
  });

  it("answers pings without counting them, and closes a socket over the rate with 4002", async () => {
    keys = await deriveKeys(newSecret());
    const a = raw(), b = raw();
    await until("both open", () => a.opened() && b.opened());
    for (let i = 0; i < RATE_LIMIT.frames + 20; i++) a.ws.send("?");
    await until("pongs", () => a.got.filter((f) => f === "!").length === RATE_LIMIT.frames + 20);
    expect(a.closed()).toBeNull();
    for (let i = 0; i < RATE_LIMIT.frames + 1; i++) a.ws.send(`{"n":${i}}`);
    await until("closed", () => a.closed() !== null);
    expect(a.closed()!.code).toBe(CLOSE.tooFast);
    await until("forwarded up to the limit", () => b.got.filter((f) => f.startsWith("{")).length === RATE_LIMIT.frames);
    b.ws.close();
  });

  it("refuses a bad room path with 404 and answers plain HTTP with text", async () => {
    const status = await new Promise<number>((resolve, reject) => {
      const ws = new WebSocketNode(target.url + "/room/too-short");
      ws.on("unexpected-response", (_req, res) => resolve(res.statusCode ?? 0));
      ws.on("open", () => reject(new Error("opened")));
      ws.on("error", () => {});
    });
    expect(status).toBe(404);
    const home = await fetch(target.http + "/");
    expect(home.status).toBe(200);
    expect(home.headers.get("content-type")).toMatch(/^text\/plain/);
    expect(await home.text()).toMatch(/relay/);
  });
});

describe(`the ${targetName} relay's origin check`, () => {
  let guarded: Target;
  beforeAll(async () => {
    guarded = await makeTarget(["https://pages.example"]);
    await guarded.start();
  }, 90_000);
  afterAll(async () => {
    await guarded.stop();
  }, 30_000);

  function tryOrigin(origin: string): Promise<number> {
    return new Promise((resolve) => {
      const ws = new WebSocketNode(guarded.url + roomPath("AAAAAAAAAAAAAAAAAAAAAA"), { origin });
      ws.on("unexpected-response", (_req, res) => resolve(res.statusCode ?? 0));
      ws.on("open", () => {
        ws.close();
        resolve(101);
      });
      ws.on("error", () => {});
    });
  }

  it("lets the listed origin in and refuses others with 403", async () => {
    expect(await tryOrigin("https://pages.example")).toBe(101);
    expect(await tryOrigin("https://other.example")).toBe(403);
  });
});
