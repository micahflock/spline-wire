// One page's connection to its pairing's room on the relay: connect, stay
// connected, seal what goes out, open and check what comes in.
//
// The link promises nothing about delivery. It reports what it sees (state,
// how many peers are in the room, messages that arrive); outbox.ts and
// inbox.ts build delivery on top.
//
// It has no DOM dependencies beyond WebSocket and timers, both injectable, so
// it runs under vitest against relay/rules.ts.

import { PING, PONG, roomPath } from "../../relay/rules";
import { open, seal, type Keys, type Role } from "./crypto";
import { parseMessage, type Message } from "./messages";

export type LinkState =
  /** No socket, and not trying (not started, or stopped). */
  | "off"
  | "connecting"
  | "open"
  /** Lost or refused; will retry. `LinkStatus.retryInMs` says when. */
  | "waiting";

export interface LinkStatus {
  state: LinkState;
  /** Other sockets in the room (0 when not open). Any peer may be a second tab of our own role. */
  peers: number;
  retryInMs?: number;
  /** Why the last connection ended, for the person, when it is worth saying. */
  problem?: string;
}

export interface LinkEvents {
  status(status: LinkStatus): void;
  /** A checked message from a peer of the other role. */
  message(message: Message): void;
}

/** The parts of WebSocket the link uses, so tests can supply a fake. */
export interface SocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
}

export interface LinkOptions {
  /** The relay's base address, ws:// or wss://, with no trailing slash. */
  relayUrl: string;
  keys: Keys;
  role: Role;
  events: LinkEvents;
  socket?: (url: string) => SocketLike;
  /** Timers, injectable for tests. */
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
  random?: () => number;
}

/** Reconnect waits: doubling from 1 s to 30 s, each with ±30% jitter so the
 * pages of a pairing do not retry in step. */
export const BACKOFF = { firstMs: 1000, maxMs: 30_000, jitter: 0.3 };
/** A ping every 25 s keeps idle connections alive through proxies and tells
 * a dead socket from a quiet one. No pong (or anything) in 2 pings: reconnect. */
export const PING_MS = 25_000;
/** A connection attempt that has neither opened nor failed by then is given
 * up and retried. Some never end on their own: a server that accepts the TCP
 * connection but does not answer the upgrade (a relay starting up, a captive
 * portal, a half-dead network), and a browser may wait minutes before failing it. */
export const CONNECT_TIMEOUT_MS = 10_000;

export class RelayLink {
  private socket: SocketLike | null = null;
  private status: LinkStatus = { state: "off", peers: 0 };
  private attempt = 0;
  private retryTimer: unknown = null;
  private pingTimer: unknown = null;
  private connectTimer: unknown = null;
  private heard = true;                 // anything received since the last ping
  private readonly o: Required<Omit<LinkOptions, "socket">> & { socket: (url: string) => SocketLike };

  constructor(options: LinkOptions) {
    this.o = {
      socket: (url) => new WebSocket(url) as unknown as SocketLike,
      setTimeout: (fn, ms) => setTimeout(fn, ms),
      clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
      random: Math.random,
      ...options,
    };
  }

  get current(): LinkStatus {
    return this.status;
  }

  /** Connect and stay connected until stop(). */
  start(): void {
    if (this.status.state !== "off") return;
    this.connect();
  }

  stop(): void {
    this.o.clearTimeout(this.retryTimer);
    this.o.clearTimeout(this.connectTimer);
    this.o.clearTimeout(this.pingTimer);
    const s = this.socket;
    this.socket = null;
    if (s) {
      s.onclose = s.onmessage = s.onopen = s.onerror = null;
      s.close(1000, "bye");
    }
    this.set({ state: "off", peers: 0 });
  }

  /** Try again now instead of waiting (the page became visible, the network came back). */
  retryNow(): void {
    if (this.status.state !== "waiting") return;
    this.o.clearTimeout(this.retryTimer);
    this.connect();
  }

  /** Seal and send. False if there is no open connection: the message is not
   * queued here (see outbox.ts). True means handed to the socket, not delivered. */
  async send(message: Message): Promise<boolean> {
    const frame = await seal(this.o.keys, this.o.role, message);
    const s = this.socket;
    if (!s || this.status.state !== "open") return false;
    try {
      s.send(frame);
      return true;
    } catch {
      return false;
    }
  }

  private connect(): void {
    this.set({ state: "connecting", peers: 0 });
    let s: SocketLike;
    try {
      s = this.o.socket(this.o.relayUrl + roomPath(this.o.keys.room));
    } catch (err) {
      this.retry(`could not reach the relay: ${String(err)}`);
      return;
    }
    this.socket = s;
    this.connectTimer = this.o.setTimeout(() => {
      if (this.socket !== s || this.status.state !== "connecting") return;
      this.socket = null;
      s.onclose = s.onmessage = s.onopen = s.onerror = null;
      try {
        s.close();
      } catch {
        /* not open yet in some implementations */
      }
      this.retry();
    }, CONNECT_TIMEOUT_MS);
    s.onopen = () => {
      if (this.socket !== s) return;
      this.o.clearTimeout(this.connectTimer);
      this.attempt = 0;
      this.heard = true;
      this.set({ state: "open", peers: 0 });
      this.schedulePing();
    };
    s.onmessage = (ev) => {
      if (this.socket !== s) return;
      this.heard = true;
      void this.receive(ev.data);
    };
    s.onclose = (ev) => {
      if (this.socket !== s) return;
      this.socket = null;
      this.o.clearTimeout(this.pingTimer);
      this.o.clearTimeout(this.connectTimer);
      // 4001: the room is full. Retrying at once would not help, but a stale
      // tab may close, so keep trying at the slowest pace and say why.
      if (ev.code === 4001) {
        this.attempt = 99;
        this.retry("this pairing is already open in too many tabs");
      } else {
        this.retry();
      }
    };
    s.onerror = () => {};       // a close always follows
  }

  private retry(problem?: string): void {
    const base = Math.min(BACKOFF.maxMs, BACKOFF.firstMs * 2 ** Math.min(this.attempt, 10));
    const wait = Math.round(base * (1 + BACKOFF.jitter * (2 * this.o.random() - 1)));
    this.attempt++;
    this.set({ state: "waiting", peers: 0, retryInMs: wait, problem });
    this.retryTimer = this.o.setTimeout(() => this.connect(), wait);
  }

  private schedulePing(): void {
    this.pingTimer = this.o.setTimeout(() => {
      const s = this.socket;
      if (!s) return;
      if (!this.heard) {
        s.close(4000, "no answer");        // onclose reconnects
        // Some sockets take long to report a close on a dead network: do not wait for it.
        if (this.socket === s) {
          s.onclose = s.onmessage = null;
          this.socket = null;
          this.retry();
        }
        return;
      }
      this.heard = false;
      try {
        s.send(PING);
      } catch {
        /* the close handler deals with it */
      }
      this.schedulePing();
    }, PING_MS);
  }

  private async receive(data: unknown): Promise<void> {
    if (typeof data !== "string" || data === PONG) return;
    if (data.startsWith("!")) {
      try {
        const control = JSON.parse(data.slice(1)) as { peers?: unknown };
        if (typeof control.peers === "number" && this.status.state === "open") {
          this.set({ state: "open", peers: Math.max(0, control.peers - 1) });
        }
      } catch {
        /* not a control frame we know */
      }
      return;
    }
    const opened = await open(this.o.keys, data);
    // Frames from our own role are another tab of ours: not for us.
    if (!opened || opened.sender === this.o.role) return;
    const message = parseMessage(opened.body);
    if (message) this.o.events.message(message);
  }

  private set(status: LinkStatus): void {
    this.status = status;
    this.o.events.status(status);
  }
}
