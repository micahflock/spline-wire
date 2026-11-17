import { defineConfig } from "vitest/config";

// Two pages, no framework: the desktop page at / and the phone page at /phone/.
// base "./" keeps every URL relative, so the site works under any path
// (GitHub Pages serves a project site from /<repo>/).
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    rollupOptions: { input: { desktop: "index.html", phone: "phone/index.html" } },
  },
  worker: { format: "es" },      // Pyodide refuses classic workers (S1)
  test: { include: ["src/**/*.test.ts", "relay/**/*.test.ts"], environment: "node" },
});
