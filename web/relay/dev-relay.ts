// The relay as a small Node server, for development and tests: the same
// behaviour as worker.ts because both are shells around rules.ts.
//
//   npm run relay                     port 8787 (or PORT), any origin
//   ALLOWED_ORIGINS=http://localhost:5173 npm run relay
//
// Rooms live in memory and vanish with the process, as the Worker's do.

import { createServer, type IncomingMessage, type Server } from "node:http";
import type { Duplex } from "node:stream";
import { pathToFileURL } from "node:url";
import { WebSocketServer, type RawData, type WebSocket } from "ws";
import {
  CLOSE,
  INFO_TEXT,
  MAX_FRAME_CHARS,
  MAX_SOCKETS_PER_ROOM,
  PING,
  PONG,
  RateWindow,
  originAllowed,
  parseOrigins,
  peersFrame,
  roomFromPath,
  route,
} from "./rules.ts";

export interface DevRelayOptions {
  /** 0 picks a free port. */
  port?: number;
  host?: string;
  /** Allowed page origins; empty allows any. */
  origins?: string[];
}

export interface DevRelay {
  port: number;
  /** ws://host:port, what the pages take as their relay address. */
  url: string;
  /** Stop, cutting every socket without a close handshake, as a crash or a
   * network loss would. */
  close(): Promise<void>;
}

interface Member {
  rate: RateWindow;
}

export function startDevRelay(options: DevRelayOptions = {}): Promise<DevRelay> {
  const host = options.host ?? "127.0.0.1";
  const origins = options.origins ?? [];
  const rooms = new Map<string, Map<WebSocket, Member>>();
  // Big enough that route() sees an oversize frame and closes it with its own
  // code; beyond this ws closes with 1009, as Cloudflare does beyond 32 MiB.
  const wss = new WebSocketServer({ noServer: true, maxPayload: 4 * MAX_FRAME_CHARS + 1024 });

  const server: Server = createServer((_req, res) => {
    res.writeHead(_req.url === "/" ? 200 : 404, { "content-type": "text/plain; charset=utf-8" });
    res.end(INFO_TEXT);
  });

  server.on("upgrade", (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    if (!originAllowed(req.headers.origin ?? null, origins)) return refuse(socket, 403, "origin not allowed");
    const room = roomFromPath(new URL(req.url ?? "/", "http://relay").pathname);
    if (!room) return refuse(socket, 404, "no such room");
    wss.handleUpgrade(req, socket, head, (ws) => join(room, ws));
  });

  function join(room: string, ws: WebSocket): void {
    const members = rooms.get(room) ?? new Map<WebSocket, Member>();
    if (members.size >= MAX_SOCKETS_PER_ROOM) {
      ws.close(CLOSE.roomFull, "room full");
      return;
    }
    rooms.set(room, members.set(ws, { rate: new RateWindow() }));
    announce(room);
    ws.on("message", (data: RawData, isBinary: boolean) => {
      const member = members.get(ws);
      if (!member) return;
      const frame = isBinary ? data : data.toString();
      const r = route(frame, frame === PING ? 0 : member.rate.hit(Date.now()));
      if (r.do === "pong") ws.send(PONG);
      else if (r.do === "close") {
        leave(room, ws);
        ws.close(r.code, r.reason);
      } else if (r.do === "forward") {
        for (const other of members.keys()) if (other !== ws) other.send(frame as string);
      }
    });
    ws.on("close", () => leave(room, ws));
    ws.on("error", () => leave(room, ws));
  }

  function leave(room: string, ws: WebSocket): void {
    const members = rooms.get(room);
    if (!members?.delete(ws)) return;
    if (members.size === 0) rooms.delete(room);
    announce(room);
  }

  function announce(room: string): void {
    const members = rooms.get(room);
    if (!members) return;
    const frame = peersFrame(members.size);
    for (const ws of members.keys()) ws.send(frame);
  }

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port ?? 0, host, () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      resolve({
        port,
        url: `ws://${host}:${port}`,
        close: () =>
          new Promise<void>((done) => {
            for (const ws of wss.clients) ws.terminate();
            rooms.clear();
            wss.close();
            server.close(() => done());
            server.closeAllConnections();
          }),
      });
    });
  });
}

function refuse(socket: Duplex, status: number, message: string): void {
  socket.end(`HTTP/1.1 ${status} ${message}\r\nContent-Type: text/plain\r\nContent-Length: ${message.length + 1}\r\nConnection: close\r\n\r\n${message}\n`);
}

// Run directly: npm run relay.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.env.PORT ?? 8787);
  const origins = parseOrigins(process.env.ALLOWED_ORIGINS);
  const relay = await startDevRelay({ port, host: process.env.HOST ?? "127.0.0.1", origins });
  console.log(`relay on ${relay.url} (origins: ${origins.length ? origins.join(", ") : "any"})`);
}
