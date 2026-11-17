import { defineConfig } from "@playwright/test";

// The end-to-end test of the core worker (e2e/core.spec.ts): builds the site,
// serves it with `vite preview` and drives the throwaway page in Chromium.
// `npm run assets` must have filled public/ first.
const PORT = 4173;

export default defineConfig({
  testDir: "e2e",
  timeout: 10 * 60_000,           // Pyodide loads twice and measures 12-megapixel photos
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${PORT}/`, browserName: "chromium", trace: "off" },
  webServer: {
    command: `npx vite build --logLevel warn && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
