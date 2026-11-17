// RelayLink on its own, for what the in-memory relay cannot imitate.
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { deriveKeys, newSecret } from "./crypto";
import { CONNECT_TIMEOUT_MS, RelayLink, type LinkStatus, type SocketLike } from "./link";

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
});
afterEach(() => {
  vi.useRealTimers();
});

/** A socket that neither opens nor fails until told to. */
class Silent implements SocketLike {
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  closed = false;
  send(): void {}
  close(): void {
    this.closed = true;
  }
}

// Found against `wrangler dev` restarting: its port accepted the connection
// before the relay behind it was up, and the upgrade was never answered. The
// link sat in "connecting" for good and the queued curve never went.
it("gives up a connection attempt that never answers, and tries again", async () => {
  const sockets: Silent[] = [];
  const seen: LinkStatus[] = [];
  const link = new RelayLink({
    relayUrl: "ws://relay.test",
    keys: await deriveKeys(newSecret()),
    role: "phone",
    socket: () => {
      const s = new Silent();
      sockets.push(s);
      return s;
    },
    random: () => 0.5,
    events: { status: (s) => seen.push(s), message: () => {} },
  });
  link.start();
  expect(link.current.state).toBe("connecting");
  await vi.advanceTimersByTimeAsync(CONNECT_TIMEOUT_MS + 10);
  expect(sockets[0]!.closed).toBe(true);
  expect(link.current.state).toBe("waiting");
  await vi.advanceTimersByTimeAsync(link.current.retryInMs!);
  expect(sockets.length).toBe(2);
  sockets[1]!.onopen?.({});
  expect(link.current.state).toBe("open");
  // An opened socket is not timed out by the attempt that made it.
  await vi.advanceTimersByTimeAsync(CONNECT_TIMEOUT_MS * 2);
  expect(link.current.state).toBe("open");
  expect(sockets.length).toBe(2);
  link.stop();
});
