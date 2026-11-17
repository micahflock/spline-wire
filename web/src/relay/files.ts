// The output files inside a curve message (decision D14): text in, gzip, base64.
//
// The phone packs PhotoSession.outputs(); the desktop unpacks them for its
// downloads. CompressionStream is in every browser the site supports and in
// Node 18+, so this runs in tests as is.

import { b64url, unb64url } from "./crypto";

/** The files the message carries. "tsv" is left out: the desktop builds it from the points (pointsTsv). */
export const FILE_KINDS = ["json", "csv", "svg", "dxf", "fusion_csv"] as const;
export type FileKind = (typeof FILE_KINDS)[number];

/** Packed files: kind → base64url of the gzipped UTF-8 text. */
export type PackedFiles = Partial<Record<FileKind, string>>;

/** Cap on all packed files together, in base64 characters. A 13-pin chain packs to about 10 K;
 * the relay's frame cap is 64 K, and the rest of the message needs room too. */
export const MAX_PACKED_CHARS = 40_000;
/** Cap on one unpacked file: a gzip bomb must not fill the desktop's memory. */
export const MAX_FILE_CHARS = 2_000_000;

export async function pack(texts: Partial<Record<FileKind | "tsv", string>>): Promise<PackedFiles> {
  const out: PackedFiles = {};
  for (const kind of FILE_KINDS) {
    const text = texts[kind];
    if (typeof text === "string") out[kind] = b64url(await gzip(new TextEncoder().encode(text)));
  }
  if (packedSize(out) > MAX_PACKED_CHARS) {
    // Too big for one message (a very long chain): send the curve without its files.
    // The desktop then offers only what it can make from the points.
    return {};
  }
  return out;
}

export async function unpack(files: PackedFiles): Promise<Partial<Record<FileKind, string>>> {
  const out: Partial<Record<FileKind, string>> = {};
  for (const kind of FILE_KINDS) {
    const packed = files[kind];
    if (typeof packed !== "string") continue;
    const text = await gunzip(unb64url(packed));
    if (text !== null) out[kind] = text;
  }
  return out;
}

/** Checks packed files off the network: known kinds, base64url strings, within the cap. */
export function parseFiles(v: unknown): PackedFiles | null {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return null;
  const out: PackedFiles = {};
  for (const [kind, value] of Object.entries(v)) {
    if (!(FILE_KINDS as readonly string[]).includes(kind)) continue;     // a newer build's kind: ignore it
    if (typeof value !== "string" || !/^[A-Za-z0-9_-]*$/.test(value)) return null;
    out[kind as FileKind] = value;
  }
  return packedSize(out) <= MAX_PACKED_CHARS ? out : null;
}

function packedSize(files: PackedFiles): number {
  return Object.values(files).reduce((n, s) => n + (s?.length ?? 0), 0);
}

async function gzip(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** null if the data is not gzip or unpacks beyond MAX_FILE_CHARS. */
async function gunzip(bytes: Uint8Array<ArrayBuffer>): Promise<string | null> {
  try {
    const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.length;
      if (total > MAX_FILE_CHARS) {
        await reader.cancel();
        return null;
      }
      chunks.push(value);
    }
    const all = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) {
      all.set(c, at);
      at += c.length;
    }
    return new TextDecoder("utf-8", { fatal: true }).decode(all);
  } catch {
    return null;
  }
}
