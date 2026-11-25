// The site's service worker, retired (D29). It kept the Python engine's
// ~40 MB (Pyodide and the wheels) on the phone; the site no longer has them.
// A browser that installed it fetches this script again when it checks for an
// update on a visit, and this version deletes the caches it made and
// unregisters itself. No page registers it any more (the phone page also
// removes it, phone/main.ts). Delete this file, and its entry in
// vite.config.ts, once old installs have had a few months to update.

declare const self: ServiceWorkerGlobalScope;
export {};                                       // a module, so `self` above is this file's

self.addEventListener("install", () => {
  void self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) if (name.startsWith("splinewire-")) await caches.delete(name);
    await self.registration.unregister();
  })());
});
