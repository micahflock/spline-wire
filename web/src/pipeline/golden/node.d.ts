// The Node functions the golden tests use to read golden/*.json.gz and test
// photos (the page's tsconfig has no Node types; vitest runs in Node).
declare module "node:fs" {
  export function readFileSync(path: URL): Uint8Array;
  export function existsSync(path: URL | string): boolean;
}
declare module "node:zlib" {
  export function gunzipSync(data: Uint8Array): { toString(encoding: "utf-8"): string };
  export function inflateSync(data: Uint8Array): Uint8Array;
}
declare module "node:crypto" {
  export function createHash(algorithm: "sha256"): { update(data: Uint8Array): { digest(encoding: "hex"): string } };
}
