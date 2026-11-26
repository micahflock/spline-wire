# web/ — the hosted build

The website that replaced `SplineWire.exe` (retired 2026-11-18). The plan, decisions and session log are in
[docs/web-build.md](../docs/web-build.md); this file is the code map.

```bash
npm install
npm test            # vitest: relay, overlay and core client logic, no browser needed
npm run typecheck
npm run assets      # chains.json and the Fusion add-in's zip -> public/ (needs uv)
npm run dev         # the desktop page at /, the phone page at /phone/ (relay: cd relay && npm run relay)
npm run build       # dist/
npm run golden      # Python's detections and measurements on 191 scenes (the 167 benchmark renders
                    # and 24 snap-fit renders) -> src/pipeline/golden/scenes.json.gz, for the
                    # TypeScript port's tests (~1 min once cv_benchmark has rendered its scenes);
                    # scripts/golden_cv.py, golden_detect.py and golden_edits.py write the others
npm run e2e         # Playwright: builds, serves dist/, measures synthetic photos in Chromium
                    # against native Python through the core worker, the desktop page and the
                    # phone page; prints timings as [core], [desktop], [phone] and [camera] lines
```

`npm run assets` (`scripts/assets.py`) must run before `dev`, `build` or `e2e`, and again after a
change to `data/` or `fusion/SplineWire/`. It also checks that `package.json`'s version, which the
pages show, is `splinewire.__version__`: bump both together. (Until 2026-11-25 it also put Pyodide,
its packages and the splinewire wheel under `public/` for a Python engine; D29 removed it, and the
script deletes what an earlier run left.) CI: `.github/workflows/web.yml` runs every check
above and the relay's (`relay/README.md`); `deploy.yml` deploys, by hand only (see Deploying).

## How the parts fit

```
phone page                                              desktop page
──────────                                              ────────────
photo ─▶ engine worker (TS)   ─▶ measurement doc       curve list, drawing,
           src/core/                  │                 Copy points, downloads
                                      ▼                        ▲
camera ─▶ overlay ─▶ still      relay/outbox ══ relay ══▶ relay/inbox
          src/overlay/          (sealed frames, the relay cannot read them)
```

| Folder | What it is | State |
|---|---|---|
| `src/core/` | The page's side of the measuring engine. `types.ts` mirrors `splinewire/session.py`'s documents; `protocol.ts` is the worker's message contract; `client.ts` makes typed calls; `make-worker.ts` starts the worker. `engine.ts` answers with the TypeScript port in `src/pipeline/`; a HEIC photo the browser cannot decode (every browser but Safari) is decoded there with libheif in WebAssembly (`src/pipeline/heif.ts`, D28), loaded only then; `src/pipeline/photo-info.ts` is what it reports of a photo. It keeps at most two photos open, the least recently used closed first. (The Python core under Pyodide, `worker.ts` and `?engine=python`, was removed on 2026-11-25, D29.) | Tested with a fake worker (`client.test.ts`) and end to end in Chromium (`e2e/core.spec.ts`). |
| `index.html`, `src/desktop/` | The desktop page, at the site's root (step 3): the pairing QR code, the list of curves (`selection.ts` which is shown, `title.ts` the tab title), the curve drawn in mm (`plot.ts`, `spline.ts`) with sharp corners a person marks by clicking its points (`corners.ts`, D27: rebuilt from the walls either side; `wrap-sim.ts` presses the snap-fit chain onto stepped profiles for its tests), Copy points, the downloads (`names.ts`), the link status (`status.ts`). `here.ts` measures a photo dropped on the page or picked with its button (D9) with the phone page's logic and the core worker; main.ts loads it with `import()` only then, so a visit that only receives curves loads none of the engine. Its curves join the list marked "measured here" (`Inbox.addLocal`). With `?fusion` it is the page in the Fusion add-in's palette (`fusion.ts`, the bridge and its contract with `fusion/SplineWire/palette.py`): its own pairing, curves into the sketch being edited, Insert, no clipboard or downloads. | Tested in Node, end to end with a test phone (`relay/`'s `check:desktop`) and measuring a dropped photo in Chromium (`e2e/desktop.spec.ts`); the `?fusion` mode with a stubbed add-in and a phone in Node (`e2e/fusion.spec.ts`). |
| `phone/index.html`, `src/phone/` | The phone page (step 2). `photos.ts` keeps photos in IndexedDB under the core's photo ids, with each measurement's output files, so that a photo can be deleted to save space and its curve kept (`cleanupPlan`: past Settings, Photos kept, and with Delete photos, keep curves; Clear history deletes everything); `measurer.ts` is each photo's status (loading, queued, measuring, done, error) and the one-at-a-time queue; `settings.ts` the settings (those for testing only in debug mode, `?debug`); `exif.ts` and `lenses.ts` the focal length per lens (D17); `camera-view.ts` the live camera with the overlay (D16), the diagnostics behind Debug (debug mode only); `drawings.ts` the page's drawings (the chain strip under the header, the history's tiles, the working curve card's crop of the photo); `taps.ts` the expanded view's double tap (fit) in front of the shared editor's taps; `send.ts` when a photo's curve goes to the computer and how it stays in step there (D26: sent once measured, each change a new revision of the same curve, stars, names and deletions both ways); `wording.ts` what is shown (in touch or mouse words: the desktop page uses it too); `main.ts` wires them to the page, laid out as `for UI review/phone/BRIEF.md` says (header with the chain's pins, Scan and Upload, the computer pill, the working curve card and its expanded view, the history grid, Settings). `sw.ts` (built to `sw.js` at the site's root) is the retired service worker, which removed Pyodide's cache and itself (D29); `main.ts` does the same on load. | Tested in Node and in Chromium at phone size (`e2e/phone.spec.ts`, which also needs `npm install` in `relay/`; `e2e/camera.spec.ts`, with Chromium's fake camera). Not tried on a phone. |
| `src/photo/` | The photo editor both pages show: `view.ts` wires an SVG (the photo as its `<image>`, the markers of `draw.ts` over it) to `src/editor/`, for touch and mouse alike; `preview.ts` makes the downscaled copy it draws (for a HEIC the browser cannot decode, the engine makes it instead, with `downscale.ts`, and `CoreClient.onPreviews` hands it over); `photo.css` the markers' look. | Through both pages' end-to-end tests; `draw.ts` also in Node. |
| `e2e/`, `scripts/` | The Playwright tests (`core.spec.ts` the worker through the desktop page's `desktop.engine()`, `desktop.spec.ts`, `phone.spec.ts`; `native.py` makes the photos and the native reference) and the assets script. | |
| `src/relay/` | Pairing and delivery between the phone and desktop pages: `crypto.ts` (one secret → room id + AES-GCM key), `messages.ts` (the versioned curve message, checked on arrival, and the clipboard table), `link.ts` (the reconnecting socket), `outbox.ts` / `inbox.ts` (at-least-once sending of each curve's latest revision, the desktop keeping the highest), `sync.ts` (stars, names and deletions kept in step: the later star or name wins, a deletion stays), `pairing.ts` (the QR link), `store.ts`. | Written; tested against `relay/memory-relay.ts` (`delivery.test.ts`; `sync.test.ts` with a model phone, the pages open together and apart), and over real sockets against both relays and in two Chromium pages (`relay/README.md`). |
| `relay/` | The relay server, its own package (`relay/README.md`). `rules.ts` is its whole behaviour as pure functions; `memory-relay.ts` runs them in memory for tests; `worker.ts` is the Cloudflare Worker (a Durable Object per room, hibernating), `dev-relay.ts` the same in Node. | Written and tested locally, the Worker under `wrangler dev`. **Not deployed: needs the owner's Cloudflare account.** |
| `src/pipeline/` | The pipeline in TypeScript (D18), with no Python and no OpenCV: what `engine.ts` runs. `decode.ts` (the browser decodes, Pillow's gray formula); `cv.ts` (the OpenCV operations detection uses, as OpenCV 5.0 computes them); `detect.ts` (dots, the snap-fit chain's thin rims, a click snapped to one); `order.ts`, `rectify.ts` (with `lm.ts`, MINPACK's Levenberg–Marquardt as SciPy runs it), `contact.ts`, `chain.ts`, `edits.ts`, `measure.ts` (`pipeline.measure`); `session.ts` (`PhotoSession`: options, the document, the editor's view with `drawing.ts`, the truth comparison in `truth.ts`); the files in `output.ts`, `pyjson.ts` and `dxf.ts` (the same small DXF as `splinewire/dxf.py`); `live.ts` with `live-worker.ts` and `live-client.ts` (the overlay's analysis). Not ported: thick ring fiducials (no preset uses them). | Against Python, cv2 and Pillow: `cv.test.ts`, `detect.test.ts` (all 191 scenes from pixels with `SPLINEWIRE_FULL_GOLDEN=1`), `golden.test.ts`, `edits.test.ts`, `live.test.ts`, `session.test.ts` (documents and every file), `dxf.test.ts` (byte for byte), and in Chromium `e2e/decode.spec.ts`. `npm run golden` and `scripts/golden_*.py` write the files. |
| `src/overlay/`, `src/camera/` | The live camera overlay: `blobs.ts` (fast dot and rim detector for video frames), `region.ts` (which part of the frame it reads and how finely: the region round the tracked chain, its rims a set size, else the whole frame or a quarter of it), `reading.ts`, `coach.ts` (what to tell the person, when to take the still; limits calibrated on S5's renders, `minSpan` on the snap-fit chain), `session.ts` (the loop), `words.ts` (what the camera view says), `trace.ts` (the chain drawn over the video), `animate.ts` (the arrow eased and faded, the lock-on rings closing on the pins as the hold fills). `snap-frames.ts` draws the snap-fit chain into any region of a phone's frame, for the tests. `camera/qr-scan.ts` with `qr-worker.ts` (jsQR) looks for the desktop page's pairing code in the frames while no chain is tracked, and `qr.ts` says whether a code is one for this page (it pairs without leaving the camera). `camera/choose.ts` picks the camera (O10), `camera/stream.ts` the stream's size (the full sensor, or 1440×1920 once the focal length is known), `camera/focal.ts` carries focal lengths between stills and frames, `camera/grab.ts` with `grab-worker.ts` grabs frames and finds their blobs in a worker (a VideoFrame handed over, drawn into the faster of two kinds of canvas; checked against the page's own grab first, and the page grabs where that fails or with `?grab=main`). The chain's order and tilt come from `src/pipeline/live.ts` in its own small worker, so coaching starts with the first frame. | Tested in Node, and in Chromium with a fake camera (`e2e/camera.spec.ts`, `e2e/camera-snap.spec.ts` at three distances, `e2e/camera-qr.spec.ts` pairing from a code). Not tried on a phone. |
| `spikes/phone-capabilities/` | The S2/S3 test page for a real phone. | Done and tried on four phones. |
| `spikes/decode-check/` | A test page for a real phone: does the browser decode a phone's JPEG to Pillow's pixels (Safari uses its own decoder), and how fast is the TypeScript engine there. | In Chromium: Pillow's pixels and Python's pins exactly (`e2e/decode-check.spec.ts`). Waiting for a phone. |
| `spikes/fusion-palette/` | The Fusion palette spike (S6, docs/fusion-palette.md): what a Fusion palette's browser can do, opened by the throwaway add-in `fusion/SplineWirePaletteSpike/`. A classic-script watchdog (user agent, Chromium, `adsk`, errors, the summary) and a module with the site's crypto, the live relay, localStorage across restarts and the add-in's round trips. | In Chromium with a stubbed `adsk.fusionSendData` (`e2e/fusion-palette.spec.ts`); the add-in in pytest with a fake `adsk`. Waiting for the owner's runs in Fusion on Windows and macOS. |
| `spikes/accuracy/`, `src/testbench/` | The accuracy test page for the real-world test series (docs/real-world-tests.md): a photo of a fixed-chain plaque or a contact gauge measured and scored at once against its known shape (`src/pipeline/scoring.ts` and `targets.ts`, ports of `splinewire/scoring.py` and `targets.py`), kept on the phone with its photo, and a report `splinewire score --report` reads. | In Node (`scoring.test.ts` against Python's scores, `bench.test.ts`) and in Chromium (`e2e/accuracy.spec.ts`). Waiting for the owner's phones. |

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
- **Memory decides the shape of the core** (a 12-megapixel photo is 48 MB decoded, and detection takes some 235 MB more):
  few photos open in the worker at once, bytes transferred not copied, `CoreClient.restart()` to
  give the memory back.

## Deploying

`.github/workflows/deploy.yml` builds the site and deploys it to GitHub Pages, and deploys the relay
to Cloudflare Workers (`relay/README.md`, "Deploying", has the account steps). It runs only when
started by hand (Actions, Deploy, Run workflow), after the Web workflow has passed on the same commit. The site job
waits until GitHub lists the uploaded site before deploying it, and deploys once more a minute later if
GitHub Pages fails; if it still fails, start a new run (not Re-run, which leaves two uploads that
GitHub Pages refuses to choose between).
Before the first run the owner sets up, once:

1. **Pages source.** Repository Settings, Pages, Build and deployment, Source: **GitHub Actions**.
   The site is then at `https://<github-user>.github.io/spline-wire/`. The deploy job uses the
   `github-pages` environment, which may accept deployments from the default branch only: run the
   workflow from it, or allow the branch under Settings, Environments, `github-pages`.
2. **Relay origins.** In `relay/wrangler.toml`, set `ALLOWED_ORIGINS = "https://<github-user>.github.io"`
   (scheme and host only, no `/spline-wire/` path) and commit it: the relay refuses pages from other
   origins, and an empty value allows any.
3. **Secrets** (Settings, Secrets and variables, Actions, Secrets): `CLOUDFLARE_API_TOKEN`, a token
   made from Cloudflare's *Edit Cloudflare Workers* template, and `CLOUDFLARE_ACCOUNT_ID`, the
   account id shown in the Cloudflare dashboard. Without the token the relay job is skipped and only
   the site is deployed.
4. **Variable** (same page, Variables): `RELAY_URL`, the relay's address as the first deploy prints
   it, with `wss://`: `wss://spline-wire-relay.<subdomain>.workers.dev`, no trailing slash. The build
   bakes it into the pages as `VITE_RELAY_URL` (`src/relay/config.ts`). It is not a secret: every page
   carries it. Without it the pages say they have no relay. The relay must be deployed once before
   this address exists: run the workflow, set the variable, run it again.

The site needs no server configuration:

- **Paths.** `vite.config.ts` has `base: "./"`, so every address in the built pages is relative and
  the site works under the project site's path `/spline-wire/` (or any other) without a rebuild.
  The phone page is `phone/` under it. `sw.js` at the site's root is the retired service worker
  (it cached Pyodide): it removes itself and its caches from browsers that installed it, and no page
  registers it (`src/phone/sw.ts`, D29).
- **Headers.** GitHub Pages cannot send custom headers, so the site cannot be cross-origin isolated
  (COOP/COEP). It does not need to be: nothing in it uses `SharedArrayBuffer` (checked in `src/`, the
  built pages and libheif's `libheif.js`, a build without threads), and
  the end-to-end tests run against `vite preview`, which sends no such headers either
  (`e2e/desktop.spec.ts` asserts `crossOriginIsolated` is false).
- **Size.** `dist/` is about 4.4 MB, the largest files libheif's wasm (1.5 MB, fetched only for a HEIC
  photo) and the decode-check spike's test image (1.5 MB), well within the 1 GB GitHub Pages documents for a published site.
