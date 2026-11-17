# web/ — the hosted build

The website that replaces `SplineWire.exe`. The plan, decisions and session log are in
[docs/web-build.md](../docs/web-build.md); this file is the code map.

```bash
npm install
npm test            # vitest: relay, overlay and core client logic, no browser needed
npm run typecheck
npm run assets      # Pyodide, its packages, the splinewire wheel and chains.json -> public/ (needs uv)
npm run dev         # the throwaway core test page at /, measuring a dropped photo
npm run build       # dist/
npm run e2e         # Playwright: builds, serves dist/, measures synthetic photos in Chromium
                    # against native Python; prints load/measure timings as [core] lines
```

`npm run assets` (`scripts/assets.py`) must run before `dev`, `build` or `e2e`, and again after a
change to `splinewire/` or `data/` (the wheel is rebuilt each time; downloads are kept while their
sha256 matches). Pyodide's core comes from the npm `pyodide` package, pinned in package-lock.json;
its packages from jsdelivr at build time only, checked against Pyodide's lock; ezdxf as uv.lock pins
it. The site serves all of it from its own origin. `public/core-manifest.json` says where each part
is, so the worker hard-codes no file names. CI: `.github/workflows/web.yml`.

## How the parts fit

```
phone page                                              desktop page
──────────                                              ────────────
photo ─▶ core worker (Pyodide) ─▶ measurement doc       curve list, drawing,
           src/core/                  │                 Copy points, downloads
                                      ▼                        ▲
camera ─▶ overlay ─▶ still      relay/outbox ══ relay ══▶ relay/inbox
          src/overlay/          (sealed frames, the relay cannot read them)
```

| Folder | What it is | State |
|---|---|---|
| `src/core/` | The page's side of the Python core. `types.ts` mirrors `splinewire/session.py`'s documents; `protocol.ts` is the worker's message contract; `client.ts` makes typed calls; `worker.ts` loads Pyodide and answers through `splinewire/bridge.py` (at most two photos open, least recently used closed first). | Written; tested with a fake worker (`client.test.ts`) and end to end in Chromium (`e2e/`). |
| `src/dev/`, `index.html` | A throwaway test page: drop a photo, see the numbers, the pins and curve, download the files. `phone/index.html` is a placeholder. | Replaced by the real pages in steps 2 and 3. |
| `e2e/`, `scripts/` | The Playwright test (`native.py` makes the photos and the native reference) and the assets script. | |
| `src/relay/` | Pairing and delivery between the phone and desktop pages: `crypto.ts` (one secret → room id + AES-GCM key), `messages.ts` (the versioned curve message, checked on arrival, and the clipboard table), `link.ts` (the reconnecting socket), `outbox.ts` / `inbox.ts` (at-least-once sending, exactly-once storing), `pairing.ts` (the QR link), `store.ts`. | Written; tested against `relay/memory-relay.ts`, and over real sockets against both relays and in two Chromium pages (`relay/README.md`). |
| `relay/` | The relay server, its own package (`relay/README.md`). `rules.ts` is its whole behaviour as pure functions; `memory-relay.ts` runs them in memory for tests; `worker.ts` is the Cloudflare Worker (a Durable Object per room, hibernating), `dev-relay.ts` the same in Node. | Written and tested locally, the Worker under `wrangler dev`. **Not deployed: needs the owner's Cloudflare account.** |
| `src/overlay/` | The live camera overlay: `blobs.ts` (fast dot detector for video frames), `reading.ts`, `coach.ts` (what to tell the person, when to take the still), `session.ts` (the loop). The chain's order and tilt come from `splinewire/live.py` through the core worker's `analyse` command. | Logic written and tested on drawn frames. Thresholds marked *provisional* in `coach.ts` are uncalibrated. No camera code, no page. |
| `spikes/phone-capabilities/` | The S2/S3 test page for a real phone. | Done; waiting to be hosted and tried. |

## Rules of the road

- **Python is the reference** (decisions D2, D3). Anything computed here that Python also computes
  (`messages.pointsTsv`, `blobs.ts` against `detect.py`) has a test pinning it to Python's output.
- **Contracts live in one file each** and say so at the top: `src/core/protocol.ts` (page ↔ worker),
  `src/core/types.ts` (↔ `splinewire/session.py`), `relay/rules.ts` (page ↔ relay),
  `src/relay/messages.ts` (phone ↔ desktop). Change a contract and its other side together.
- **Logic is kept apart from the browser.** WebSocket, Worker, storage, timers and the camera are
  passed in, so every module here has tests that run in Node. Pages are thin.
- **Errors for people** follow the repo's rule: a Python `ValueError`'s message is shown as is
  (`Failure.kind === "person"`); anything else is a bug and is shown with its type.
- **Memory decides the shape of the core** (S1: 580 MB of wasm heap for one 12-megapixel photo):
  few photos open in the worker at once, bytes transferred not copied, `CoreClient.restart()` to
  give the memory back.
