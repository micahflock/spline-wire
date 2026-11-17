// Shared by the two demo pages: where the relay is, and how to say the link's state.
import type { LinkStatus } from "../../src/relay/link";

/** ?relay=ws://… on the page's address, else a dev relay on this host. */
export function relayUrl(): string {
  return new URLSearchParams(location.search).get("relay") ?? `ws://${location.hostname}:8787`;
}

export function describe(s: LinkStatus): string {
  if (s.state === "open") return `open, ${s.peers} peer(s)`;
  if (s.state === "waiting") return `waiting ${s.retryInMs} ms${s.problem ? ": " + s.problem : ""}`;
  return s.state;
}

export function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}
