// End-to-end encryption between paired pages (web-build.md, O4).
//
// A pairing is one random secret, made by the desktop page and carried to the
// phone in the QR link's #fragment (which browsers never send to a server).
// From it both pages derive, with HKDF-SHA-256:
//   - the room id, which the relay sees and routes by;
//   - an AES-GCM key, which it never sees.
// Knowing the room id gives neither the key nor the secret.
//
// A sealed frame is JSON text: {"v":1,"s":"phone"|"desktop","n":nonce,"c":ciphertext}.
// The sender's role and the room are bound in as additional data, so a frame
// cannot be replayed into another room or passed off as the other side's.
// Replays within a room are harmless by design: curves are deduplicated by id
// and acks are idempotent (inbox.ts, outbox.ts).

export type Role = "phone" | "desktop";

export const FRAME_VERSION = 1;
const SALT = "spline-wire/pairing@1";
const SECRET_BYTES = 16;

export interface Keys {
  /** Public: the relay routes by it. */
  room: string;
  key: CryptoKey;
}

export interface Opened {
  sender: Role;
  body: unknown;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

export function b64url(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function unb64url(text: string): Uint8Array<ArrayBuffer> {
  if (!/^[A-Za-z0-9_-]*$/.test(text)) throw new Error("not base64url");
  const s = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

/** A new pairing secret: 128 random bits as 22 base64url characters. */
export function newSecret(): string {
  return b64url(crypto.getRandomValues(new Uint8Array(SECRET_BYTES)));
}

export function isSecret(text: unknown): text is string {
  return typeof text === "string" && /^[A-Za-z0-9_-]{22}$/.test(text);
}

export async function deriveKeys(secret: string): Promise<Keys> {
  if (!isSecret(secret)) throw new Error("not a pairing secret");
  const base = await crypto.subtle.importKey("raw", unb64url(secret), "HKDF", false, ["deriveBits", "deriveKey"]);
  const params = (info: string) => ({ name: "HKDF", hash: "SHA-256", salt: enc.encode(SALT), info: enc.encode(info) });
  const room = new Uint8Array(await crypto.subtle.deriveBits(params("room"), base, 128));
  const key = await crypto.subtle.deriveKey(params("key"), base, { name: "AES-GCM", length: 256 }, false, [
    "encrypt",
    "decrypt",
  ]);
  return { room: b64url(room), key };
}

function aad(room: string, sender: Role): Uint8Array<ArrayBuffer> {
  return enc.encode(`spline-wire/${FRAME_VERSION}|${room}|${sender}`);
}

/** Encrypt `body` (anything JSON) as a frame for the relay. */
export async function seal(keys: Keys, sender: Role, body: unknown): Promise<string> {
  // A random 96-bit nonce per frame: safe for far more frames than a pairing will ever send.
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const cipher = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: nonce, additionalData: aad(keys.room, sender) },
    keys.key,
    enc.encode(JSON.stringify(body)),
  );
  return JSON.stringify({ v: FRAME_VERSION, s: sender, n: b64url(nonce), c: b64url(new Uint8Array(cipher)) });
}

/** Decrypt a frame. null if it is not a frame of this pairing: malformed,
 * another version, tampered with, or sealed with another key. Never throws:
 * frames come from the network. */
export async function open(keys: Keys, frame: string): Promise<Opened | null> {
  try {
    const f = JSON.parse(frame) as Record<string, unknown>;
    if (f.v !== FRAME_VERSION || (f.s !== "phone" && f.s !== "desktop")) return null;
    if (typeof f.n !== "string" || typeof f.c !== "string") return null;
    const plain = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: unb64url(f.n), additionalData: aad(keys.room, f.s) },
      keys.key,
      unb64url(f.c),
    );
    return { sender: f.s, body: JSON.parse(dec.decode(plain)) };
  } catch {
    return null;
  }
}
