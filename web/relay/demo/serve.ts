// Builds the demo pages and serves them with a dev relay beside them.
//   npm run demo            pages on http://127.0.0.1:8788/desktop.html, relay on 8787
// The pages import the client backbone (src/relay/) as TypeScript; esbuild
// bundles it, so the demo needs nothing from web/'s own build.

import { readFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "esbuild";
import { startDevRelay, type DevRelay } from "../dev-relay.ts";

const here = fileURLToPath(new URL(".", import.meta.url));
const TYPES: Record<string, string> = { ".html": "text/html", ".js": "text/javascript" };

export interface Demo {
  /** The desktop page, already pointed at the relay. */
  desktop: string;
  relay: DevRelay;
  /** Restart the relay on the same port (for the dropped-socket check). */
  restartRelay(): Promise<void>;
  stopRelay(): Promise<void>;
  close(): Promise<void>;
}

export async function startDemo(pagePort = 0, relayPort = 0): Promise<Demo> {
  await build({
    entryPoints: [join(here, "desktop.ts"), join(here, "phone.ts")],
    outdir: join(here, "dist"),
    bundle: true,
    format: "esm",
    target: "es2022",
    logLevel: "warning",
  });
  let relay = await startDevRelay({ port: relayPort });
  const port = relay.port;
  const server: Server = createServer(async (req, res) => {
    const path = normalize(new URL(req.url ?? "/", "http://x").pathname).replace(/^\/+/, "");
    if (!/^(desktop\.html|phone\.html|dist\/(desktop|phone)\.js)$/.test(path)) {
      res.writeHead(404).end("not found\n");
      return;
    }
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "text/plain", "cache-control": "no-store" });
    res.end(await readFile(join(here, path)));
  });
  await new Promise<void>((r) => server.listen(pagePort, "127.0.0.1", r));
  const address = server.address();
  const pages = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
  const demo: Demo = {
    desktop: `${pages}/desktop.html?relay=${encodeURIComponent(relay.url)}`,
    relay,
    async stopRelay() {
      await relay.close();
    },
    async restartRelay() {
      relay = await startDevRelay({ port });
      demo.relay = relay;
    },
    async close() {
      await relay.close();
      server.closeAllConnections();
      await new Promise((r) => server.close(r));
    },
  };
  return demo;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const demo = await startDemo(8788, 8787);
  console.log(`desktop page: ${demo.desktop}`);
}
