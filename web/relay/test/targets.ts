// The relays the integration tests run against: the Node dev relay, or the
// Worker under `wrangler dev` (workerd, Cloudflare's runtime, run locally).
// Chosen with RELAY_TARGET=dev|worker; dev is the default.

import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";
import { startDevRelay, type DevRelay } from "../dev-relay.ts";

export interface Target {
  /** ws://host:port */
  url: string;
  /** http://host:port */
  http: string;
  start(): Promise<void>;
  /** Stop the relay process; every socket drops. */
  stop(): Promise<void>;
}

export const targetName = process.env.RELAY_TARGET === "worker" ? "worker" : "dev";

export async function makeTarget(origins: string[] = []): Promise<Target> {
  const port = await freePort();
  return targetName === "worker" ? workerTarget(port, origins) : devTarget(port, origins);
}

function devTarget(port: number, origins: string[]): Target {
  let relay: DevRelay | null = null;
  return {
    url: `ws://127.0.0.1:${port}`,
    http: `http://127.0.0.1:${port}`,
    async start() {
      relay = await startDevRelay({ port, origins });
    },
    async stop() {
      await relay?.close();
      relay = null;
    },
  };
}

const relayDir = fileURLToPath(new URL("..", import.meta.url));

function workerTarget(port: number, origins: string[]): Target {
  let child: ChildProcess | null = null;
  const http = `http://127.0.0.1:${port}`;
  return {
    url: `ws://127.0.0.1:${port}`,
    http,
    async start() {
      const args = ["wrangler", "dev", "--ip", "127.0.0.1", "--port", String(port), "--inspector-port", "0"];
      args.push("--var", `ALLOWED_ORIGINS:${origins.join(",")}`);
      child = spawn("npx", args, {
        cwd: relayDir,
        env: { ...process.env, WRANGLER_SEND_METRICS: "false", CI: "1", NO_COLOR: "1" },
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,               // its own process group, so stop() takes workerd with it
      });
      let log = "";
      child.stdout!.on("data", (d) => (log += d));
      child.stderr!.on("data", (d) => (log += d));
      const deadline = Date.now() + 60_000;
      for (;;) {
        try {
          if ((await fetch(http + "/")).ok) return;
        } catch {
          /* not up yet */
        }
        if (child.exitCode !== null || Date.now() > deadline) throw new Error("wrangler dev did not start:\n" + log);
        await new Promise((r) => setTimeout(r, 250));
      }
    },
    async stop() {
      const c = child;
      child = null;
      if (!c || c.exitCode !== null) return;
      const exited = new Promise((r) => c.once("exit", r));
      process.kill(-c.pid!, "SIGKILL");
      await exited;
      // workerd may hold the port a moment after its parent is gone.
      for (let i = 0; i < 40; i++) {
        try {
          await fetch(http + "/");
        } catch {
          return;
        }
        await new Promise((r) => setTimeout(r, 100));
      }
    },
  };
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.once("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const a = s.address();
      s.close(() => resolve(typeof a === "object" && a ? a.port : 0));
    });
  });
}
