// Where the pages find the relay and each other. Shared by the desktop and
// phone pages.
//
// The relay's address is baked into the build (VITE_RELAY_URL, set by the
// deploy workflow; relay/README.md, "Deploying"). In development it defaults
// to the dev relay on the page's own host (`cd web/relay && npm run relay`),
// which also works from a phone on the same network. `?relay=ws://…` on the
// page's address overrides both, for tests and for trying another relay; the
// desktop passes it on in the pairing link so the phone uses the same one.

export interface RelaySetting {
  /** location.search */
  search: string;
  /** location.hostname */
  hostname: string;
  /** import.meta.env.VITE_RELAY_URL */
  built?: string | undefined;
  /** import.meta.env.DEV */
  dev: boolean;
}

export const DEV_RELAY_PORT = 8787;
const PARAM = "relay";

/** The relay's base address, or null when this build has none (a production
 * build made without VITE_RELAY_URL): the page then says so. */
export function relayAddress(s: RelaySetting): string | null {
  const given = new URLSearchParams(s.search).get(PARAM);
  if (given && isRelayUrl(given)) return trim(given);
  if (s.built && isRelayUrl(s.built)) return trim(s.built);
  if (s.dev) return `ws://${s.hostname}:${DEV_RELAY_PORT}`;
  return null;
}

/** The phone page's address, for the pairing link: `phone/` under the site's
 * root, carrying a `?relay=` override along. */
export function phonePageUrl(siteRoot: string, search: string): string {
  const url = new URL("phone/", siteRoot);
  const given = new URLSearchParams(search).get(PARAM);
  if (given && isRelayUrl(given)) url.searchParams.set(PARAM, given);
  return url.href;
}

/** The setting for the page this runs in. */
export function pageRelaySetting(): RelaySetting {
  return {
    search: location.search,
    hostname: location.hostname,
    built: import.meta.env.VITE_RELAY_URL,
    dev: import.meta.env.DEV,
  };
}

function isRelayUrl(text: string): boolean {
  try {
    const u = new URL(text);
    return (u.protocol === "ws:" || u.protocol === "wss:") && !u.search && !u.hash;
  } catch {
    return false;
  }
}

function trim(url: string): string {
  return url.replace(/\/+$/, "");
}
