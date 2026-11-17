// The site's service worker: keeps the core's ~40 MB (Pyodide and the wheels)
// on the phone so the second visit does not download them again. The rules
// are in sw-rules.ts; this file only wires them to the service worker events.
//
// Built to sw.js at the site's root (vite.config.ts), so its scope covers both
// pages and the core worker they start. The pages work without it: it is
// registered only where the browser allows one, and caches nothing they need.
// It must import nothing but sw-rules.ts, which the build inlines: it is
// registered as a classic script, which every browser supports.

import {
  META_CACHE, MANIFEST, cacheName, cacheable, parseManifest, routeOf, staleCaches, staleEntries, type CoreManifest,
} from "./sw-rules";

declare const self: ServiceWorkerGlobalScope;

const scope = () => self.registration.scope;
let current: { name: string; manifest: CoreManifest } | null = null;

self.addEventListener("install", () => {
  // A new version takes over at once: it only changes how files are cached.
  void self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    await self.clients.claim();
    await known();
    await tidy();
  })());
});

self.addEventListener("fetch", (event) => {
  const route = routeOf(event.request.url, scope(), event.request.method);
  if (route === "manifest") event.respondWith(manifest(event));
  else if (route === "core") event.respondWith(cacheFirst(event));
});

/** Network first; remembers the manifest to name the cache, and serves the
 * remembered one when the network is down. */
async function manifest(event: FetchEvent): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(event.request);
  } catch (err) {
    const kept = await (await caches.open(META_CACHE)).match(scope() + MANIFEST);
    if (kept) return kept;
    throw err;
  }
  if (!res.ok) return res;
  const text = await res.clone().text();
  const m = parseManifest(text);
  if (m) event.waitUntil(learn(m, text));
  return res;
}

async function cacheFirst(event: FetchEvent): Promise<Response> {
  const c = await known();
  if (!c) return fetch(event.request);           // no manifest seen yet: nothing to name the cache by
  const cache = await caches.open(c.name);
  const hit = await cache.match(event.request);
  if (hit) return hit;
  const res = await fetch(event.request);
  if (cacheable(res.status, res.type)) event.waitUntil(cache.put(event.request, res.clone()).catch(() => {}));
  return res;
}

/** The current cache: from memory, else the remembered manifest, else the network's. */
async function known(): Promise<typeof current> {
  if (current) return current;
  const meta = await caches.open(META_CACHE);
  let res = await meta.match(scope() + MANIFEST);
  if (!res) {
    try {
      res = await fetch(scope() + MANIFEST, { cache: "no-cache" });
    } catch {
      return null;
    }
    if (!res.ok) return null;
  }
  const text = await res.text();
  const m = parseManifest(text);
  if (m) await learn(m, text);
  return current;
}

async function learn(m: CoreManifest, text: string): Promise<void> {
  const name = cacheName(m);
  const changed = !current || current.name !== name || JSON.stringify(current.manifest.wheels) !== JSON.stringify(m.wheels);
  current = { name, manifest: m };
  await (await caches.open(META_CACHE)).put(scope() + MANIFEST,
    new Response(text, { headers: { "content-type": "application/json" } }));
  if (changed) await tidy();
}

/** Delete other versions' caches, and wheels the manifest no longer lists. */
async function tidy(): Promise<void> {
  if (!current) return;
  for (const name of staleCaches(await caches.keys(), current.name)) await caches.delete(name);
  const cache = await caches.open(current.name);
  const urls = (await cache.keys()).map((r) => r.url);
  for (const url of staleEntries(urls, scope(), current.manifest)) await cache.delete(url);
}
