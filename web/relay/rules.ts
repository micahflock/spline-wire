// The relay's whole behaviour, as pure functions. The Cloudflare Worker
// (relay/worker.ts) and the local relay for development and tests
// (relay/dev-relay.ts) are thin shells around this file, so they cannot
// disagree about the protocol.
//
// The relay is a dumb pipe between the pages of one pairing ("room"):
//   - it forwards text frames between the sockets of a room, unread;
//   - it tells every socket how many are in the room when that changes;
//   - it answers a ping.
// It stores nothing and cannot read anything: frames are encrypted by the
// pages with a key the relay never sees (src/relay/crypto.ts), and the room id
// is a hash of that key's secret. Delivery is the pages' job (acks and the
// phone's outbox, src/relay/outbox.ts), which is why a dropped frame is safe.
//
// Wire format, all text frames:
//   page  -> relay   "?"            ping
//   relay -> page    "!"            pong
//   relay -> page    "!{json}"      control: {"peers": n} (sockets in the room, this one included)
//   page  -> relay   "{json}"       a sealed frame, forwarded as is to the other sockets
//   anything else is dropped.

/** Room ids are 16 bytes, base64url without padding. */
export const ROOM_RE = /^[A-Za-z0-9_-]{22}$/;

/** A phone and a desktop, plus a second tab of each. More is not a pairing. */
export const MAX_SOCKETS_PER_ROOM = 4;

/** A curve message is about 1-3 KB sealed. The cap leaves room and stops abuse. */
export const MAX_FRAME_CHARS = 64 * 1024;

/** Frames a socket may send per window before it is closed. */
export const RATE_LIMIT = { frames: 60, windowMs: 10_000 };

export const PING = "?";
export const PONG = "!";

/** Close codes the pages understand (4000-4999 are for applications). */
export const CLOSE = {
  roomFull: 4001,
  tooFast: 4002,
  tooBig: 4003,
} as const;

/** The path of a room's WebSocket, shared with the pages. */
export function roomPath(room: string): string {
  return `/room/${room}`;
}

/** The room a request path asks for, or null if it is not a room path. */
export function roomFromPath(path: string): string | null {
  const m = /^\/room\/([^/]+)$/.exec(path);
  return m && ROOM_RE.test(m[1]!) ? m[1]! : null;
}

export function peersFrame(count: number): string {
  return "!" + JSON.stringify({ peers: count });
}

export type Route =
  | { do: "pong" }
  | { do: "forward" }
  | { do: "drop" }
  | { do: "close"; code: number; reason: string };

/** What to do with a frame a page sent. `recent` is how many frames this
 * socket sent in the current rate window, this one included. */
export function route(frame: unknown, recent: number): Route {
  if (typeof frame !== "string") return { do: "drop" };          // binary: not ours
  if (frame === PING) return { do: "pong" };
  if (recent > RATE_LIMIT.frames) return { do: "close", code: CLOSE.tooFast, reason: "too many messages" };
  if (frame.length > MAX_FRAME_CHARS) return { do: "close", code: CLOSE.tooBig, reason: "message too large" };
  return frame.startsWith("{") ? { do: "forward" } : { do: "drop" };
}

/** A RateWindow as plain data, for relays that must keep it outside memory
 * (the Worker's Durable Object forgets its memory when it hibernates). */
export interface RateState {
  start: number;
  count: number;
}

/** A fixed-window counter for RATE_LIMIT; one per socket. */
export class RateWindow {
  private start = 0;
  private count = 0;
  constructor(state?: RateState | null) {
    if (state && Number.isFinite(state.start) && Number.isFinite(state.count)) {
      this.start = state.start;
      this.count = state.count;
    }
  }
  /** Count a frame at time `now` (ms) and return the count in its window. */
  hit(now: number): number {
    if (now - this.start >= RATE_LIMIT.windowMs) {
      this.start = now;
      this.count = 0;
    }
    return ++this.count;
  }
  get state(): RateState {
    return { start: this.start, count: this.count };
  }
}

/** The allowed page origins from a comma-separated setting. Empty allows any
 * origin, for development. */
export function parseOrigins(setting: string | undefined): string[] {
  return (setting ?? "")
    .split(",")
    .map((o) => o.trim().replace(/\/+$/, ""))
    .filter((o) => o !== "");
}

/** Whether a page at `origin` may open a socket. Only browsers send Origin,
 * so this keeps other sites' pages off the relay, not other programs: the
 * rate and size limits are what protect it. */
export function originAllowed(origin: string | null, allowed: readonly string[]): boolean {
  return allowed.length === 0 || (origin !== null && allowed.includes(origin));
}

/** The answer to anything that is not a room's WebSocket. */
export const INFO_TEXT =
  "Spline Wire relay. It passes sealed messages between a paired phone and computer and stores nothing.\n";
