// The two Node functions the golden tests use to read golden/*.json.gz
// (the page's tsconfig has no Node types; vitest runs in Node).
declare module "node:fs" {
  export function readFileSync(path: URL): Uint8Array;
}
declare module "node:zlib" {
  export function gunzipSync(data: Uint8Array): { toString(encoding: "utf-8"): string };
}
