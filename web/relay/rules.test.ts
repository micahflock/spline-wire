// The relay's rules, alone. The shells around them (worker.ts, dev-relay.ts)
// are checked with real sockets in test/relay.itest.ts.
import { describe, expect, it } from "vitest";
import {
  CLOSE,
  MAX_FRAME_CHARS,
  RATE_LIMIT,
  RateWindow,
  originAllowed,
  parseOrigins,
  peersFrame,
  roomFromPath,
  roomPath,
  route,
} from "./rules";

const ROOM = "AAAAAAAAAAAAAAAAAAAAAA";

describe("room paths", () => {
  it("parses what roomPath makes, and nothing else", () => {
    expect(roomFromPath(roomPath(ROOM))).toBe(ROOM);
    expect(roomFromPath("/room/abc_DEF-0123456789xyz")).toBe("abc_DEF-0123456789xyz");
    for (const bad of [
      "/", "/room", "/room/", `/room/${ROOM}/`, `/room/${ROOM}x`, `/room/${ROOM.slice(1)}`,
      `/rooms/${ROOM}`, `room/${ROOM}`, `/room/${ROOM.slice(1)}=`, `/room/${ROOM.slice(1)}+`,
      `/x/room/${ROOM}`, `/room/${ROOM}/x`,
    ]) {
      expect(roomFromPath(bad), bad).toBeNull();
    }
  });
});

describe("route", () => {
  it("answers pings, forwards sealed frames and drops the rest", () => {
    expect(route("?", 0)).toEqual({ do: "pong" });
    expect(route('{"v":1}', 1)).toEqual({ do: "forward" });
    expect(route("hello", 1)).toEqual({ do: "drop" });
    expect(route("!{\"peers\":9}", 1)).toEqual({ do: "drop" });            // pages cannot fake control frames
    expect(route("", 1)).toEqual({ do: "drop" });
    expect(route(new ArrayBuffer(4), 1)).toEqual({ do: "drop" });
  });

  it("closes a socket that sends too much or too fast", () => {
    expect(route("{" + "x".repeat(MAX_FRAME_CHARS - 1), 1)).toEqual({ do: "forward" });
    expect(route("{" + "x".repeat(MAX_FRAME_CHARS), 1)).toMatchObject({ do: "close", code: CLOSE.tooBig });
    expect(route("{}", RATE_LIMIT.frames)).toEqual({ do: "forward" });
    expect(route("{}", RATE_LIMIT.frames + 1)).toMatchObject({ do: "close", code: CLOSE.tooFast });
  });

  it("answers a ping even from a socket over its rate", () => {
    expect(route("?", RATE_LIMIT.frames + 100)).toEqual({ do: "pong" });
  });
});

describe("RateWindow", () => {
  it("counts frames in fixed windows", () => {
    const w = new RateWindow();
    expect(w.hit(1_000_000)).toBe(1);
    expect(w.hit(1_000_000 + RATE_LIMIT.windowMs - 1)).toBe(2);
    expect(w.hit(1_000_000 + RATE_LIMIT.windowMs)).toBe(1);               // a new window
  });

  it("picks up where it left off from its state (the Worker's attachment)", () => {
    const w = new RateWindow();
    for (let i = 0; i < 10; i++) w.hit(5000 + i);
    const copy = new RateWindow(JSON.parse(JSON.stringify(w.state)));
    expect(copy.hit(5100)).toBe(11);
    expect(new RateWindow(null).hit(5100)).toBe(1);
    expect(new RateWindow({ start: NaN, count: 3 }).hit(5100)).toBe(1);   // garbage starts afresh
  });
});

describe("origins", () => {
  it("allows any origin when none are set", () => {
    expect(parseOrigins(undefined)).toEqual([]);
    expect(parseOrigins(" , ")).toEqual([]);
    expect(originAllowed(null, [])).toBe(true);
    expect(originAllowed("https://evil.example", [])).toBe(true);
  });

  it("allows only the listed origins otherwise", () => {
    const allowed = parseOrigins(" https://someone.github.io/ ,http://localhost:5173");
    expect(allowed).toEqual(["https://someone.github.io", "http://localhost:5173"]);
    expect(originAllowed("https://someone.github.io", allowed)).toBe(true);
    expect(originAllowed("http://localhost:5173", allowed)).toBe(true);
    expect(originAllowed("https://someone.github.io.evil.example", allowed)).toBe(false);
    expect(originAllowed("http://someone.github.io", allowed)).toBe(false);
    expect(originAllowed(null, allowed)).toBe(false);
  });
});

it("tells pages how many are in the room", () => {
  expect(peersFrame(2)).toBe('!{"peers":2}');
});
