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

/** `hash` without its `key` parameter, the others kept as written ("" if none is
 * left). The pairing secret and a chain card's link (`cal`, pipeline/chainlink.ts)
 * share the fragment, `#pair=…&cal=…`, and each reader strips only its own. */
export function hashWithout(hash: string, key: string): string {
  const kept = hash.replace(/^#/, "").split("&").filter((p) => p !== "" && p.split("=")[0] !== key);
  return kept.length ? `#${kept.join("&")}` : "";
}

/** On the phone at page load: take a secret from the address if there is one
 * (a newly scanned code replaces the old pairing), else the stored one.
 * `replaceHash` is given the fragment without the secret, so it is not shared or
 * bookmarked, while anything else there (a chain card's `cal`) stays for its own reader. */
export function phoneSecret(store: KeyValueStore, hash: string, replaceHash: (rest: string) => void): string | null {
  const fromLink = secretFromHash(hash);
  if (fromLink) {
    store.set(KEY, fromLink);
    replaceHash(hashWithout(hash, PARAM));
    return fromLink;
  }
  return storedSecret(store);
}
