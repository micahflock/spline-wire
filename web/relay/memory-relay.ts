// The relay, in memory, for tests: rooms of fake sockets run by relay/rules.ts.
// It behaves as the real one does at the level the pages can see, and can be
// told to misbehave (drop frames, cut connections).

import type { SocketLike } from "../src/relay/link";
import { CLOSE, MAX_SOCKETS_PER_ROOM, PONG, RateWindow, peersFrame, roomFromPath, route } from "./rules";

class Peer implements SocketLike {
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  readonly rate = new RateWindow();
  closed = false;
  constructor(readonly relay: MemoryRelay, readonly room: string) {}

  send(data: string): void {
    if (this.closed) throw new Error("socket is closed");
    queueMicrotask(() => this.relay.fromPage(this, data));
  }
  close(code = 1000, reason = ""): void {
    this.relay.drop(this, code, reason);
  }
  deliver(data: string): void {
    queueMicrotask(() => !this.closed && this.onmessage?.({ data }));
  }
}

export class MemoryRelay {
  private readonly rooms = new Map<string, Set<Peer>>();
  /** While false, connections are refused (the relay is down). */
  up = true;
  /** Return true to lose a forwarded frame. */
  lose: (frame: string) => boolean = () => false;
  now: () => number = Date.now;
  /** Every forwarded frame, as the relay saw it: for asserting it cannot read them. */
  readonly forwarded: string[] = [];

  /** Pass as LinkOptions.socket. */
  readonly socket = (url: string): SocketLike => {
    const room = roomFromPath(new URL(url).pathname);
    const peer = new Peer(this, room ?? "");
    queueMicrotask(() => {
      if (!this.up || !room) return this.end(peer, 1006, "");
      const peers = this.rooms.get(room) ?? new Set<Peer>();
      if (peers.size >= MAX_SOCKETS_PER_ROOM) return this.end(peer, CLOSE.roomFull, "room full");
      this.rooms.set(room, peers.add(peer));
      peer.onopen?.({});
      this.announce(room);
    });
    return peer;
  };

  fromPage(peer: Peer, data: string): void {
    if (peer.closed) return;
    const r = route(data, data === "?" ? 0 : peer.rate.hit(this.now()));
    if (r.do === "pong") peer.deliver(PONG);
    else if (r.do === "close") this.drop(peer, r.code, r.reason);
    else if (r.do === "forward") {
      if (this.lose(data)) return;
      this.forwarded.push(data);
      for (const other of this.rooms.get(peer.room) ?? []) if (other !== peer) other.deliver(data);
    }
  }

  /** Close a socket from either end. */
  drop(peer: Peer, code: number, reason: string): void {
    if (peer.closed) return;
    this.rooms.get(peer.room)?.delete(peer);
    this.end(peer, code, reason);
    this.announce(peer.room);
  }

  /** Cut every connection, as a relay restart or a network change does. */
  cutAll(): void {
    for (const peers of [...this.rooms.values()]) for (const p of [...peers]) this.drop(p, 1006, "");
  }

  count(room: string): number {
    return this.rooms.get(room)?.size ?? 0;
  }

  private end(peer: Peer, code: number, reason: string): void {
    peer.closed = true;
    queueMicrotask(() => peer.onclose?.({ code, reason }));
  }

  private announce(room: string): void {
    const peers = this.rooms.get(room);
    if (!peers) return;
    if (peers.size === 0) this.rooms.delete(room);
    for (const p of peers) p.deliver(peersFrame(peers.size));
  }
}
