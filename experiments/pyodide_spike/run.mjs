// Playwright driver for the Pyodide spike (S1).
//
//   node experiments/pyodide_spike/run.mjs PHOTO[:TRUTH] [PHOTO[:TRUTH] ...] [--out FILE] [--no-deps-check]
//
// Starts server.py, then in desktop Chromium (fresh profile):
//   1. cold load (empty HTTP cache), measure each photo (2 runs each)
//   2. warm load in a new tab of the same profile (HTTP cache warm), measure again
//   3. a separate tab checks a plain `micropip.install` of the splinewire wheel
//      (with its dependencies) and `micropip.install("ezdxf")` from PyPI
// and records timings, bytes served, wasm heap, measureUserAgentSpecificMemory
// and the peak RSS of Chromium's processes. Paths are relative to the repo root.
import { spawn, execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const PORT = 8770;
const ORIGIN = `http://127.0.0.1:${PORT}`;
const PAGE = `${ORIGIN}/experiments/pyodide_spike/index.html`;

const args = process.argv.slice(2);
let outFile = path.join(ROOT, "out/pyodide_spike/results-browser.json");
const photos = [];
let depsCheck = true;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--out") outFile = path.resolve(args[++i]);
  else if (args[i] === "--no-deps-check") depsCheck = false;
  else {
    const [photo, truth] = args[i].split(":");
    photos.push({ photo, truth });
  }
}

// Peak resident memory of Chromium's processes, sampled every 100 ms.
function chromeRss() {
  const out = {};
  for (const pid of fs.readdirSync("/proc").filter((d) => /^\d+$/.test(d))) {
    try {
      const cmd = fs.readFileSync(`/proc/${pid}/cmdline`, "utf8");
      if (!/chrom/i.test(cmd)) continue;
      const rss = +fs.readFileSync(`/proc/${pid}/status`, "utf8").match(/VmRSS:\s+(\d+)/)[1] * 1024;
      const type = (cmd.match(/--type=([\w-]+)/) || [, "browser"])[1];
      out[pid] = { type, rss };
    } catch {}
  }
  return out;
}
let peak = { renderer: 0, total: 0 };
function sample() {
  const ps = Object.values(chromeRss());
  const total = ps.reduce((s, p) => s + p.rss, 0);
  const renderer = Math.max(0, ...ps.filter((p) => p.type === "renderer").map((p) => p.rss));
  peak.renderer = Math.max(peak.renderer, renderer);
  peak.total = Math.max(peak.total, total);
}
function resetPeak() { const p = peak; peak = { renderer: 0, total: 0 }; return p; }

const stats = async () => (await fetch(`${ORIGIN}/__stats`)).json();
const resetStats = () => fetch(`${ORIGIN}/__reset`);

async function waitForServer() {
  for (let i = 0; i < 100; i++) {
    try { await fetch(`${ORIGIN}/__stats`); return; } catch { await new Promise((r) => setTimeout(r, 100)); }
  }
  throw new Error("server did not start");
}

async function withTimeout(p, ms) {
  return Promise.race([p, new Promise((r) => setTimeout(() => r("timeout"), ms))]);
}

async function session(context, label, opts) {
  await resetStats();
  resetPeak();
  const page = await context.newPage();
  page.on("console", (m) => process.stderr.write(`[${label}] ${m.text()}\n`));
  page.on("pageerror", (e) => process.stderr.write(`[${label}] pageerror ${e}\n`));
  const t0 = Date.now();
  await page.goto(PAGE);
  const loaded = await page.evaluate((o) => spike.load(o), opts);
  const loadWall = Date.now() - t0;
  const served = await stats();
  const loadPeak = resetPeak();
  const out = { label, load_wall_ms: loadWall, load: loaded, served_bytes: served,
                served_total: Object.values(served).reduce((a, b) => a + b, 0),
                rss_peak_load: loadPeak, measurements: [] };
  process.stderr.write(`[${label}] loaded in ${loadWall} ms, served ${out.served_total} bytes\n`);
  for (const { photo, truth } of photos) {
    resetPeak();
    const t1 = Date.now();
    const r = await page.evaluate(([u, n, t]) => spike.measure(u, n, t, 2),
      [`${ORIGIN}/${photo}`, path.basename(photo), truth ? `${ORIGIN}/${truth}` : null]);
    r.page_wall_ms = Date.now() - t1;
    r.rss_peak = resetPeak();
    out.measurements.push(r);
    process.stderr.write(`[${label}] ${photo}: ${r.pins_px.length} pins, runs ${r.runs.map((x) => x.total_s.toFixed(2)).join(", ")} s, heap ${(r.heap_bytes / 2 ** 20).toFixed(0)} MiB\n`);
  }
  out.ua_memory_bytes = await withTimeout(page.evaluate(() => spike.memory()), 60000);
  await page.close();
  return out;
}

const server = spawn("python3", [path.join(ROOT, "experiments/pyodide_spike/server.py"), "--port", String(PORT)],
                     { stdio: ["ignore", "inherit", "inherit"] });
const timer = setInterval(sample, 100);
const results = { page: PAGE, chromium: null, sessions: [] };
try {
  await waitForServer();
  const profile = path.join(ROOT, "out/pyodide_spike/profile");
  fs.rmSync(profile, { recursive: true, force: true });
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "127.0.0.1,localhost" } : undefined;
  const context = await chromium.launchPersistentContext(profile, { proxy });
  results.chromium = context.browser()?.version() ?? null;
  results.sessions.push(await session(context, "cold", {}));
  results.sessions.push(await session(context, "warm", {}));
  if (depsCheck) {
    try {
      results.sessions.push(await session(context, "deps-check", { tryDeps: true, ezdxfFromPyPI: true }));
    } catch (e) {
      results.sessions.push({ label: "deps-check", error: String(e) });
    }
  }
  await context.close();
} finally {
  clearInterval(timer);
  server.kill();
}
fs.writeFileSync(outFile, JSON.stringify(results, null, 1));
process.stderr.write(`wrote ${outFile}\n`);
