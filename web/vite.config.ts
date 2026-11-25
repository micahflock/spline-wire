import { defineConfig } from "vitest/config";
import pkg from "./package.json";

// The version and CI run the pages show (src/env.d.ts); CI sets BUILD_ID. Read through
// globalThis because this file is checked without Node's types.
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {};

// Two pages, no framework: the desktop page at / and the phone page at /phone/.
// base "./" keeps every URL relative, so the site works under any path
// (GitHub Pages serves a project site from /<repo>/).
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        desktop: "index.html", phone: "phone/index.html", sw: "src/phone/sw.ts",
        // The S2/S3 test page, for the owner to open on a phone at spikes/phone-capabilities/.
        capabilities: "spikes/phone-capabilities/index.html",
        // The decoding check (spikes/decode-check/README.md): the TypeScript engine on a phone.
        decode: "spikes/decode-check/index.html",
        // The accuracy test page (spikes/accuracy/README.md, docs/real-world-tests.md).
        accuracy: "spikes/accuracy/index.html",
        // The Fusion palette spike (spikes/fusion-palette/README.md), opened by fusion/SplineWirePaletteSpike/.
        fusion: "spikes/fusion-palette/index.html",
      },
      // The service worker keeps a fixed name at the site's root: its scope is
      // its own folder, and the pages register it by name.
      output: { entryFileNames: (chunk) => (chunk.name === "sw" ? "sw.js" : "assets/[name]-[hash].js") },
    },
  },
  define: {
    "import.meta.env.APP_VERSION": JSON.stringify((pkg as { version?: string }).version ?? "dev"),
    "import.meta.env.BUILD_ID": JSON.stringify(env.BUILD_ID || "dev"),
  },
  worker: { format: "es" },      // module workers: the engine loads libheif with import()
  test: { include: ["src/**/*.test.ts", "relay/**/*.test.ts"], environment: "node" },
});
