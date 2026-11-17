// The relay on Cloudflare: a Worker in front, one Durable Object per room.
// Every decision is relay/rules.ts's; this file only connects it to the
// platform. See README.md in this folder for deploying it and for the costs.
//
// Why a Durable Object: the two pages of a pairing must reach the same
// process, and a Durable Object addressed by name (the room id) is exactly
// that. It uses the WebSocket Hibernation API, so a room whose pages are
// connected but quiet is evicted from memory and costs nothing while it waits;
// the pings that keep the sockets alive are answered by the platform
// (setWebSocketAutoResponse) without waking it.
//
// It stores nothing: no storage calls at all. The only state is per socket and
// lives in the socket's attachment, because memory is lost on hibernation.

import { DurableObject } from "cloudflare:workers";
import {
  CLOSE,
  INFO_TEXT,
  MAX_SOCKETS_PER_ROOM,
  PING,
  PONG,
  RateWindow,
  originAllowed,
  parseOrigins,
  peersFrame,
  roomFromPath,
  route,
  type RateState,
} from "./rules";

export interface Env {
  ROOMS: DurableObjectNamespace<Room>;
  /** Comma-separated page origins, e.g. "https://someone.github.io". Empty allows any. */
  ALLOWED_ORIGINS?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
      return text(url.pathname === "/" ? 200 : 404, INFO_TEXT);
    }
    // Refused here, before the Durable Object, so stray requests cost one
    // Worker request and no Durable Object request.
    if (!originAllowed(request.headers.get("Origin"), parseOrigins(env.ALLOWED_ORIGINS))) {
      return text(403, "origin not allowed\n");
    }
    const room = roomFromPath(url.pathname);
    if (!room) return text(404, "no such room\n");
    return env.ROOMS.get(env.ROOMS.idFromName(room)).fetch(request);
  },
} satisfies ExportedHandler<Env>;

/** What a socket keeps across hibernation. */
interface Attachment {
  rate: RateState;
}

export class Room extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // Answered by the platform, even while the object hibernates. Pings are
    // therefore never counted against the rate limit, as in rules.route.
    ctx.setWebSocketAutoResponse(new WebSocketRequestResponsePair(PING, PONG));
  }

  override async fetch(_request: Request): Promise<Response> {
    const [client, server] = Object.values(new WebSocketPair()) as [WebSocket, WebSocket];
    if (this.open().length >= MAX_SOCKETS_PER_ROOM) {
      // Accepted only to be closed with a code the page understands; a plain
      // HTTP refusal would reach the page as an anonymous 1006. accept(), not
      // acceptWebSocket(): this socket never belongs to the room.
      server.accept();
      server.close(CLOSE.roomFull, "room full");
      return new Response(null, { status: 101, webSocket: client });
    }
    this.ctx.acceptWebSocket(server);
    server.serializeAttachment({ rate: new RateWindow().state } satisfies Attachment);
    this.announce();
    return new Response(null, { status: 101, webSocket: client });
  }

  override async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): Promise<void> {
    let recent = 0;
    if (message !== PING) {
      // The window is read from and written back to the attachment on every
      // frame: an object may hibernate between any two frames, and a counter
      // in memory would then start again from zero. Writing it costs nothing
      // billable (the attachment is not storage).
      const rate = new RateWindow((ws.deserializeAttachment() as Attachment | null)?.rate);
      recent = rate.hit(Date.now());
      ws.serializeAttachment({ rate: rate.state } satisfies Attachment);
    }
    const r = route(message, recent);
    if (r.do === "pong") ws.send(PONG);
    else if (r.do === "close") {
      ws.close(r.code, r.reason);
      this.announce(ws);
    } else if (r.do === "forward") {
      for (const other of this.open()) if (other !== ws) send(other, message as string);
    }
  }

  // With the compatibility date in wrangler.toml (on or after 2026-04-07,
  // web_socket_auto_reply_to_close) the platform completes the close
  // handshake itself, so there is nothing to close here; only the others
  // need to hear that the room got smaller.
  override async webSocketClose(ws: WebSocket): Promise<void> {
    this.announce(ws);
  }

  override async webSocketError(ws: WebSocket): Promise<void> {
    this.announce(ws);
  }

  /** The room's sockets that can still talk. getWebSockets() may include
   * sockets that are closing, and the one whose close is being handled. */
  private open(leaving?: WebSocket): WebSocket[] {
    return this.ctx.getWebSockets().filter((s) => s !== leaving && s.readyState === WebSocket.OPEN);
  }

  private announce(leaving?: WebSocket): void {
    const peers = this.open(leaving);
    const frame = peersFrame(peers.length);
    for (const s of peers) send(s, frame);
  }
}

/** A socket may close between the check and the send; that is not an error here. */
function send(ws: WebSocket, data: string): void {
  try {
    ws.send(data);
  } catch {
    /* gone: its close handler announces it */
  }
}

function text(status: number, body: string): Response {
  return new Response(body, { status, headers: { "content-type": "text/plain; charset=utf-8" } });
}
