// The service worker's decisions (sw.ts), kept apart so they are tested in Node.
//
// What it caches: the core's assets, which public/core-manifest.json describes:
// Pyodide under pyodide/ and the wheels under wheels/, about 40 MB. They are
// served cache-first, so a second visit reads them from the phone instead of
// downloading them again. Everything else (the pages, their scripts, the
// manifest itself) goes to the network as if there were no service worker;
// the manifest is kept as a fallback for when the network is down.
//
// Versioning: Pyodide's files do not carry its version in their paths
// (pyodide/pyodide.asm.wasm), so the cache is named after the version the
// manifest gives, and a manifest with another version starts a new cache and
// deletes the old ones. The wheels' paths do change with their content
// (wheels/<hash>/...): wheels the current manifest no longer lists are deleted
// from the cache, so a new build of the package does not grow it.

export const CACHE_PREFIX = "splinewire-core-";
/** Bump when the cache's layout changes; old caches are then deleted. */
export const CACHE_LAYOUT = "1";
/** Holds the last manifest seen, so a restarted service worker knows its cache. */
export const META_CACHE = "splinewire-meta-" + CACHE_LAYOUT;
export const MANIFEST = "core-manifest.json";

export interface CoreManifest {
  pyodide: { version: string; index: string };
  wheels: Record<string, string>;
}

export type Route = "manifest" | "core" | "other";

/** What to do with a request. `scope` is the service worker's scope (the site's root). */
export function routeOf(url: string, scope: string, method = "GET"): Route {
  if (method !== "GET" || !url.startsWith(scope)) return "other";
  const path = url.slice(scope.length).split(/[?#]/)[0]!;
  if (path === MANIFEST) return "manifest";
  if (path.startsWith("pyodide/") || path.startsWith("wheels/")) return "core";
  return "other";
}

export function cacheName(manifest: CoreManifest): string {
  return `${CACHE_PREFIX}${CACHE_LAYOUT}-pyodide-${manifest.pyodide.version}`;
}

/** Caches of ours that are not the current one: delete them. */
export function staleCaches(names: readonly string[], current: string): string[] {
  return names.filter((n) => (n.startsWith(CACHE_PREFIX) && n !== current) ||
    (n.startsWith("splinewire-meta-") && n !== META_CACHE));
}

/** Cached URLs to delete under a manifest: wheels it no longer lists. */
export function staleEntries(urls: readonly string[], scope: string, manifest: CoreManifest): string[] {
  const wanted = new Set(Object.values(manifest.wheels).map((p) => new URL(p, scope).href));
  return urls.filter((u) => u.startsWith(scope + "wheels/") && !wanted.has(u.split(/[?#]/)[0]!));
}

export function parseManifest(text: string): CoreManifest | null {
  try {
    const m = JSON.parse(text) as Partial<CoreManifest>;
    if (typeof m?.pyodide?.version !== "string" || typeof m.wheels !== "object" || !m.wheels) return null;
    return m as CoreManifest;
  } catch {
    return null;
  }
}

/** Only whole, successful answers are kept (not a 206, not an opaque redirect). */
export function cacheable(status: number, type: string): boolean {
  return status === 200 && (type === "basic" || type === "default");
}
