// A QR code seen by the live camera (the desktop page's pairing code): what it
// means for this page. The scanning itself is qr-scan.ts, the decoding
// qr-worker.ts (jsQR).

import { secretFromHash } from "../relay/pairing";

export type CodeMeaning =
  | { kind: "pair"; secret: string }
  /** A pairing code made by another copy of the site (another host or path): not used. */
  | { kind: "elsewhere"; where: string }
  /** Not a pairing code at all. */
  | { kind: "other" };

/** `here` is this phone page's address. A pairing code is the phone page's address
 * with #pair=<secret> (relay/pairing.ts); only one for this very page is used,
 * so a code can never send the phone to another site. */
export function codeMeaning(text: string, here: string): CodeMeaning {
  let url: URL, page: URL;
  try {
    url = new URL(text);
    page = new URL(here);
  } catch {
    return { kind: "other" };
  }
  const secret = secretFromHash(url.hash);
  if (!secret) return { kind: "other" };
  if (url.origin !== page.origin || url.pathname !== page.pathname) return { kind: "elsewhere", where: url.host + url.pathname };
  return { kind: "pair", secret };
}
