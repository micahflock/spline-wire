// The Fusion palette spike (README.md here; docs/fusion-palette.md, "The spike").
// The watchdog in index.html keeps the facts and the summary; this module adds
// the checks that need the site's own code: the pairing crypto, the live relay,
// localStorage across restarts, and the round trips with the spike add-in
// (fusion/SplineWirePaletteSpike/), whose actions are:
//
//   page -> adsk.fusionSendData(action, json) -> add-in's returnData (json text)
//     "spike-echo"   {v:1, nonce}         -> {ok, nonce, fusion, os, started, sketch, palettes_made, switches, palette_lost}
//     "spike-insert" {v:1, points_mm}     -> {ok}; then the add-in pushes "spike-inserted" {ok, sketch, points | error}
//     "spike-report" {v:1, text}          -> {ok, path}
//   a refusal is {ok: false, error: <a message for a person>}.
//
// It is an es2022 module with a top-level await, as the site's pages are, so its
// running at all says the desktop page's build would run here.

import { deriveKeys, newSecret, open, seal, type Keys } from "../../src/relay/crypto";
import { pageRelaySetting, relayAddress } from "../../src/relay/config";
import { PING, PONG, roomPath } from "../../relay/rules";

interface Push {
  action: string;
  data: string;
  at: number;
}

interface Spike {
  set(section: "browser" | "web" | "fusion", key: string, text: string): void;
  module: boolean;
  sendAt: number | null;
  onPush: ((p: Push) => void) | null;
  pushes: Push[];
}

declare global {
  interface Window {
    __spike: Spike;
    adsk?: { fusionSendData?: (action: string, data: string) => unknown };
  }
}

const S = window.__spike;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ms = (t: number) => `${Math.round(t)} ms`;
const why = (e: unknown) => (e instanceof Error ? e.message : String(e));
const STORE = "splinewire-fusion-spike";

await Promise.resolve();                         // top-level await: Chromium 89 or later
S.module = true;
S.set("browser", "page script", "ran (an es2022 module with top-level await, as the site's pages are)");
const version = `Spline Wire ${import.meta.env.APP_VERSION} (build ${import.meta.env.BUILD_ID})`;
$("version").textContent = version;
S.set("browser", "site", version);

// --- localStorage: does it outlive the add-in and Fusion? ---------------------

interface Stored {
  first: string;
  last: string;
  loads: number;
  /** When the add-in started, one per run of it: more than one means storage outlived a restart. */
  starts: string[];
}

function readStore(): Stored | null {
  const parsed = JSON.parse(localStorage.getItem(STORE) ?? "null") as Stored | null;
  return parsed && typeof parsed.loads === "number" && Array.isArray(parsed.starts) ? parsed : null;
}

function storage(): void {
  try {
    const prev = readStore();
    const now = new Date().toISOString();
    const next: Stored = { first: prev?.first ?? now, last: now, loads: (prev?.loads ?? 0) + 1, starts: prev?.starts ?? [] };
    localStorage.setItem(STORE, JSON.stringify(next));
    S.set("web", "localStorage", prev
      ? `kept from ${prev.first} (load ${next.loads})`
      : "empty: first load here. Restart Fusion, open the palette again and look here");
  } catch (e) {
    S.set("web", "localStorage", `not usable: ${why(e)}`);
  }
}

function noteStart(started: string): void {
  try {
    const s = readStore();
    if (!s) return;
    if (!s.starts.includes(started)) s.starts = [...s.starts, started].slice(-10);
    localStorage.setItem(STORE, JSON.stringify(s));
    S.set("web", "localStorage across restarts", s.starts.length > 1
      ? `yes: kept through ${s.starts.length - 1} restart(s) of the add-in`
      : "not tested yet: restart Fusion (or stop and run the add-in) and open the palette again");
  } catch (e) {
    S.set("web", "localStorage across restarts", `not usable: ${why(e)}`);
  }
}

// --- the pairing's crypto and the relay ---------------------------------------

async function cryptoCheck(): Promise<Keys> {
  const t = performance.now();
  const keys = await deriveKeys(newSecret());
  S.set("web", "HKDF", `ok (room id and key in ${ms(performance.now() - t)})`);
  const frame = await seal(keys, "desktop", { spike: "aes" });
  const back = await open(keys, frame);
  const tampered = await open(keys, frame.replace(/"c":"(.)/, (_m, c: string) => `"c":"${c === "A" ? "B" : "A"}`));
  if ((back?.body as { spike?: string } | undefined)?.spike !== "aes" || tampered !== null) {
    throw new Error("a sealed frame did not open, or a tampered one did");
  }
  S.set("web", "AES-GCM", "ok (sealed, opened, a tampered frame refused)");
  return keys;
}

class Socket {
  readonly ws: WebSocket;
  private readonly frames: string[] = [];
  private wake: (() => void) | null = null;
  closed: CloseEvent | null = null;

  constructor(url: string) {
    this.ws = new WebSocket(url);
    this.ws.onmessage = (e) => { this.frames.push(String(e.data)); this.wake?.(); };
    this.ws.onclose = (e) => { this.closed = e; this.wake?.(); };
  }

  opened(timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`no connection in ${timeoutMs / 1000} s`)), timeoutMs);
      this.ws.addEventListener("open", () => { clearTimeout(timer); resolve(); });
      this.ws.addEventListener("close", (e) => {
        clearTimeout(timer);
        reject(new Error(`could not connect (close code ${e.code}): the network, or the relay refusing this page's origin`));
      });
    });
  }

  /** The first frame (from now on or already received) that `want` accepts. */
  async next(want: (f: string) => boolean, what: string, timeoutMs = 10_000): Promise<string> {
    const end = performance.now() + timeoutMs;
    for (;;) {
      const i = this.frames.findIndex(want);
      if (i >= 0) return this.frames.splice(i, 1)[0]!;
      if (this.closed) throw new Error(`the relay closed the socket (${this.closed.code}) before ${what}`);
      const left = end - performance.now();
      if (left <= 0) throw new Error(`no ${what} in ${timeoutMs / 1000} s`);
      await new Promise<void>((r) => { this.wake = r; setTimeout(r, left); });
      this.wake = null;
    }
  }
}

const peers = (n: number) => (f: string) => f.startsWith("!{") && (JSON.parse(f.slice(1)) as { peers?: number }).peers === n;

async function relayCheck(keys: Keys): Promise<void> {
  const base = relayAddress(pageRelaySetting());
  if (!base) {
    S.set("web", "relay", "this build has no relay address");
    return;
  }
  const url = base + roomPath(keys.room);
  const sockets: Socket[] = [];
  try {
    let t = performance.now();
    const a = new Socket(url);
    sockets.push(a);
    await a.opened(10_000);
    const connect = performance.now() - t;
    t = performance.now();
    a.ws.send(PING);
    await a.next((f) => f === PONG, "pong");
    const pong = performance.now() - t;
    const b = new Socket(url);
    sockets.push(b);
    await b.opened(10_000);
    await a.next(peers(2), "peer count 2");
    const nonce = newSecret();
    a.ws.send(await seal(keys, "desktop", { spike: nonce }));
    const got = await open(keys, await b.next((f) => f.startsWith("{"), "the sealed frame"));
    if ((got?.body as { spike?: string } | undefined)?.spike !== nonce || got?.sender !== "desktop") {
      throw new Error("the frame that arrived did not open to what was sent");
    }
    S.set("web", "relay", `ok: ${base}, connected in ${ms(connect)}, ping/pong ${ms(pong)}, peers 2, `
      + "a sealed frame crossed and opened");
  } catch (e) {
    S.set("web", "relay", `FAILED: ${why(e)} (${base})`);
  } finally {
    for (const s of sockets) s.ws.close();
  }
}

// --- the add-in ---------------------------------------------------------------

type Reply = Record<string, unknown> & { ok?: boolean; error?: string };

/** One call through adsk.fusionSendData: the add-in's parsed reply, how long it
 * took, and what fusionSendData returned (a Promise in Fusion's Qt browser). */
async function send(action: string, body: object): Promise<{ reply: Reply; took: number; returned: string }> {
  const fn = window.adsk?.fusionSendData;
  if (typeof fn !== "function") throw new Error("adsk.fusionSendData is missing");
  const t = performance.now();
  const r = fn.call(window.adsk, action, JSON.stringify({ v: 1, ...body }));
  const returned = r instanceof Promise || typeof (r as { then?: unknown } | null)?.then === "function" ? "a Promise" : typeof r;
  const text = await r;
  if (typeof text !== "string") throw new Error(`the reply was ${typeof text}, not text (fusionSendData returned ${returned})`);
  const reply = JSON.parse(text) as Reply;
  if (reply.ok !== true) throw new Error(`the add-in refused: ${reply.error ?? text}`);
  return { reply, took: performance.now() - t, returned };
}

async function whenAdsk(timeoutMs: number): Promise<boolean> {
  const end = performance.now() + timeoutMs;
  while (S.sendAt === null && performance.now() < end) await new Promise((r) => setTimeout(r, 20));
  return S.sendAt !== null;
}

async function echo(): Promise<void> {
  try {
    const nonce = newSecret();
    const { reply, took, returned } = await send("spike-echo", { nonce });
    if (reply.nonce !== nonce) throw new Error("the reply did not carry the nonce sent");
    S.set("fusion", "fusionSendData", `ok: round trip ${ms(took)}, it returns ${returned}`);
    S.set("fusion", "Fusion", `${reply.fusion} on ${reply.os}; sketch being edited: ${reply.sketch ?? "none"}`);
    S.set("fusion", "palette", `made ${reply.palettes_made} time(s); workspace switches ${reply.switches}, `
      + `after which the palette was gone ${reply.palette_lost} time(s) and made again`);
    if (typeof reply.started === "string") noteStart(reply.started);
  } catch (e) {
    S.set("fusion", "fusionSendData", `FAILED: ${why(e)}`);
  }
}

let insertedAt = 0;

async function insert(): Promise<void> {
  S.set("fusion", "insert", "asked…");
  insertedAt = performance.now();
  try {
    await send("spike-insert", { points_mm: [[0, 0], [20, 10], [40, 0]] });
    S.set("fusion", "insert", "accepted; waiting for the add-in to say it is done…");
  } catch (e) {
    S.set("fusion", "insert", `FAILED: ${why(e)}`);
  }
}

S.onPush = (p) => {
  if (p.action !== "spike-inserted") {
    S.set("fusion", "pushed", `unexpected "${p.action}"`);
    return;
  }
  const r = JSON.parse(p.data) as Reply & { sketch?: string; points?: number };
  S.set("fusion", "insert", r.ok
    ? `${r.points} points and a spline into ${r.sketch}, as a command; pushed back after ${ms(p.at - insertedAt)}`
    : `FAILED in Fusion: ${r.error}`);
};

async function save(): Promise<void> {
  try {
    const { reply } = await send("spike-report", { text: ($("summary") as HTMLTextAreaElement).value });
    $("copied").textContent = `saved to ${reply.path}`;
  } catch (e) {
    $("copied").textContent = `could not save: ${why(e)}`;
  }
}

$("insert").addEventListener("click", () => void insert());
$("again").addEventListener("click", () => void echo());
$("save").addEventListener("click", () => void save());
$<HTMLSelectElement>("undo").addEventListener("change", (e) => {
  const v = (e.target as HTMLSelectElement).value;
  S.set("fusion", "one Ctrl+Z", v || "not tried");
});

storage();
S.set("fusion", "one Ctrl+Z", "not tried");
const relay = cryptoCheck().then(relayCheck, (e) => S.set("web", "HKDF / AES-GCM", `FAILED: ${why(e)}`));
if (await whenAdsk(10_000)) {
  for (const id of ["insert", "again", "save"]) $<HTMLButtonElement>(id).disabled = false;
  await echo();
} else {
  S.set("fusion", "fusionSendData", "no adsk.fusionSendData: not in a Fusion palette");
}
await relay;
