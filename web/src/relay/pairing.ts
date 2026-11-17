// Making and remembering a pairing (web-build.md, step 3).
//
// The desktop page makes the secret once and shows it as a QR code holding
// the phone page's address with the secret in the #fragment. The phone page
// reads it, stores it, and removes it from the address bar. Both remember it,
// so pairing is done once per phone and computer.
//
// One desktop secret pairs any number of phones. "Unpair" on the desktop makes
// a new secret, which cuts every phone off; on a phone it forgets the desktop.

import { isSecret, newSecret } from "./crypto";
import type { KeyValueStore } from "./store";

const KEY = "pairing";
const PARAM = "pair";

export function storedSecret(store: KeyValueStore): string | null {
  const s = store.get(KEY);
  return isSecret(s) ? s : null;
}

/** The desktop's secret, made on first use. */
export function desktopSecret(store: KeyValueStore): string {
  const existing = storedSecret(store);
  if (existing) return existing;
  const secret = newSecret();
  store.set(KEY, secret);
  return secret;
}

export function forgetPairing(store: KeyValueStore): void {
  store.remove(KEY);
}

/** The address the QR code holds. `phonePage` is the phone page's URL without a fragment. */
export function pairingUrl(phonePage: string, secret: string): string {
  return `${phonePage.split("#")[0]}#${PARAM}=${secret}`;
}

/** The secret in a location.hash, if there is one. */
export function secretFromHash(hash: string): string | null {
  const s = new URLSearchParams(hash.replace(/^#/, "")).get(PARAM);
  return isSecret(s) ? s : null;
}

/** On the phone at page load: take a secret from the address if there is one
 * (a newly scanned code replaces the old pairing), else the stored one.
 * `clearHash` removes it from the address bar so it is not shared or bookmarked. */
export function phoneSecret(store: KeyValueStore, hash: string, clearHash: () => void): string | null {
  const fromLink = secretFromHash(hash);
  if (fromLink) {
    store.set(KEY, fromLink);
    clearHash();
    return fromLink;
  }
  return storedSecret(store);
}
