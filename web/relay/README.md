# web/relay/ — the relay between a paired phone and computer

The relay forwards sealed frames between the pages of one pairing ("room") and stores nothing
(web-build.md, O2 and O4; spike S4). Its whole behaviour is `rules.ts`; everything else here is a
thin shell around it.

| File | What it is |
|---|---|
| `rules.ts` | The protocol and every decision: room paths, routing, limits, rate window, origin check. Shared with the pages (`src/relay/link.ts`). |
| `memory-relay.ts` | The rules in memory, for the client's unit tests (`src/relay/delivery.test.ts`). |
| `worker.ts`, `wrangler.toml` | The real relay: a Cloudflare Worker with one Durable Object per room, WebSocket Hibernation API. |
| `dev-relay.ts` | The same relay as a Node server (`ws`), for development and tests. |
| `test/relay.itest.ts` | The client backbone (link, outbox, inbox) through a relay over real sockets; runs against either relay. |
| `test/browser-check.ts`, `demo/` | Two Chromium pages paired by the link, curves passed, reload, relay restart. |
| `test/desktop-check.ts` | The site's desktop page (`web/desktop/`) under the vite dev server, paired with a test phone (`desktop-check-phone.html`) that sends a Python measurement: clipboard, downloads, reload, unpair. |

This folder has its own `package.json` so its packages (ws, wrangler, Cloudflare's types, esbuild)
stay out of the site's. `web/`'s `npm test` runs `rules.test.ts` only; the rest is run here.

```bash
cd web/relay
npm install
npm run relay            # dev relay on ws://127.0.0.1:8787 (PORT, HOST, ALLOWED_ORIGINS)
npm test                 # rules + integration tests against the dev relay (~5 s)
npm run test:worker      # the same integration tests against worker.ts under `wrangler dev` (~40 s)
npm run check:browser    # two-tab check in Chromium through Playwright
npm run check:desktop    # the desktop page end to end in Chromium (SCREENSHOTS=<dir> saves them)
npm run demo             # the demo pages at http://127.0.0.1:8788/desktop.html, by hand
npm run typecheck        # worker.ts against Cloudflare's types; the rest against Node's and the DOM's
npm run dev:worker       # the Worker locally on :8787, no account needed
```

## How it behaves

- `GET /room/<id>` with a WebSocket upgrade joins the room. `<id>` must match `ROOM_RE`, else **404**.
  An `Origin` not in `ALLOWED_ORIGINS` gets **403**. Anything else gets a one-line text page.
- At most 4 sockets per room; a fifth is accepted and closed at once with **4001** (a plain HTTP
  refusal would reach the page as an anonymous 1006). Over 60 frames in 10 s: closed with **4002**.
  A frame over 64 K characters: closed with **4003**.
- `?` is answered `!` by the platform (`setWebSocketAutoResponse`), so pings neither wake the
  Durable Object nor count against the rate limit.
- When the room changes, every socket in it gets `!{"peers":n}`.

**Rate state across hibernation.** Each socket's `RateWindow` is kept in its attachment
(`serializeAttachment`), read and written back on every frame. Memory is lost when the object
hibernates, and the sockets it gets back are new objects, so a `Map` keyed by socket would be
empty after a wake. With today's numbers that happens to be harmless (the object hibernates after
10 s without events and the window is 10 s, so every window has expired by then), but the 10 s is
described by Cloudflare as "currently" and the coincidence is not worth depending on. The
attachment costs nothing billable; Durable Object storage would cost a row write per frame
(100,000 a day on the free plan) for state that only matters for 10 s.

## Cost model and limits (S4)

Checked 2026-11-17 against Cloudflare's documentation (pages marked updated 2026-10-30 to
2026-11-13). Sources:
[Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/),
[Durable Objects limits](https://developers.cloudflare.com/durable-objects/platform/limits/),
[Workers limits](https://developers.cloudflare.com/workers/platform/limits/),
[WebSockets and hibernation](https://developers.cloudflare.com/durable-objects/best-practices/websockets/),
[DurableObjectState](https://developers.cloudflare.com/durable-objects/api/state/),
[Durable Object lifecycle](https://developers.cloudflare.com/durable-objects/concepts/durable-object-lifecycle/),
[Workers WebSockets](https://developers.cloudflare.com/workers/runtime-apis/websockets/).

| | Workers Free | Workers Paid ($5/month minimum) |
|---|---|---|
| Worker requests | 100,000/day (then Error 1027) | no limit |
| Durable Object requests (HTTP, WebSocket connects, incoming WebSocket messages) | 100,000/day | 1 M/month, then $0.15/M |
| Incoming WebSocket messages | billed at 20:1 (20 messages = 1 request); outgoing messages free | same |
| Durable Object duration (128 MB per object, only while running or idle but unable to hibernate) | 13,000 GB-s/day (≈ 28 h of one awake object) | 400,000 GB-s/month, then $12.50/M GB-s |
| SQLite storage | 5 GB; the relay stores nothing | — |
| Durable Object kinds | SQLite-backed only (hence `new_sqlite_classes`) | both |
| Over a free limit | further operations of that kind fail; resets 00:00 UTC; no bill | billed |

Limits that touch the design:

- WebSocket message received: 32 MiB, above that the platform closes with 1009. Ours is 64 K
  characters (`MAX_FRAME_CHARS`); a sealed curve is 1–3 KB, about 15 KB once it carries the
  gzipped output files (D14).
- Attachment: 16,384 bytes serialized (ours: two numbers). Auto-response request and response:
  2,048 characters each (ours: one).
- Hibernation: after 10 s with no events, and only with no `setTimeout`/`setInterval` pending (the
  relay sets none). Auto-responses do not wake the object and add no duration.
- CPU: 10 ms per invocation on the free plan's Workers; 30 s per Durable Object event. The relay's
  work per frame is a string check and a few sends.
- One object: soft limit of 1,000 requests/s. Durable Object classes per account: 100 (free).

What one person's use costs: opening a page is 1 Worker request + 1 Durable Object request; a curve
and its ack are 2 incoming messages (0.1 request); a page left open sends a ping every 25 s, 1,152
in 8 h, answered without waking the object (whether auto-answered pings count as billed messages is
not stated; if they do, 58 requests). Duration is milliseconds per frame. So daily use by a
handful of people is under 1% of the free plan's request limit and its duration allowance is never
approached. A stuck page reconnecting every 30 s at worst costs 2,880 requests a day.

Not verified against a real account (none was available): everything above is the documentation's
word. The relay's behaviour was verified locally under `wrangler dev` (workerd, Cloudflare's
runtime), which does not reproduce billing, the free plan's limits, or multi-location placement.

## Deploying (owner)

1. **Account.** Sign up at <https://dash.cloudflare.com/sign-up> (free, email only, no card for the
   Workers Free plan). In the dashboard open *Workers & Pages* once and pick a `workers.dev`
   subdomain (`wrangler deploy` also asks for one if it is missing).
2. **Log in from this folder.** `cd web/relay && npm install && npx wrangler login` (opens a browser
   to authorise Wrangler). Nothing is written to the repo.
3. **Allowed origins.** In `wrangler.toml`, set `ALLOWED_ORIGINS` to the origins of the pages,
   comma-separated, scheme and host only, no path: for GitHub Pages that is
   `https://<github-user>.github.io` even when the site lives under `/spline-wire/`. Add
   `http://localhost:5173` while developing the pages against the deployed relay. Leaving it empty
   allows any origin. It is not a secret; keep it in `wrangler.toml` (a value set in the dashboard
   is replaced by the file's on the next deploy).
4. **Deploy.** `npx wrangler deploy`. It prints the address,
   `https://spline-wire-relay.<subdomain>.workers.dev`. Opening it in a browser shows the one-line
   text page. Redeploy the same way after any change; the `[[migrations]]` entry is applied once.
5. **Tell the pages.** The relay address the pages need is the same with `wss://`:
   `wss://spline-wire-relay.<subdomain>.workers.dev` (no trailing slash; `LinkOptions.relayUrl`).
   The site's pages read it at build time as the Vite variable `VITE_RELAY_URL`
   (`src/relay/config.ts`), to be set in the deploy workflow from a repository *variable* (not a
   secret: it is public in every page anyway); in development they default to `ws://<host>:8787`.
   `?relay=` on a page's address overrides it, here in the demo pages too.
6. **Later, deploying from CI** (step 4): a Cloudflare API token from the *Edit Cloudflare Workers*
   template stored as the repository secret `CLOUDFLARE_API_TOKEN`, and the account id as
   `CLOUDFLARE_ACCOUNT_ID`; the workflow runs `npx wrangler deploy` in `web/relay`. Never commit
   either.

To watch it: `npx wrangler tail` streams the Worker's logs; the dashboard shows requests and
Durable Object usage against the free limits.
