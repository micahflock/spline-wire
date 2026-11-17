# Web build: architecture and plan

[← Back to the README](../README.md)

Spline Wire is moving from a Windows program (`SplineWire.exe`, which serves pages on the local network) to a hosted website. This doc is the plan and the running record. The design of the measurement itself is unchanged and stays in [architecture.md](architecture.md).

## Picking up where we left off

1. Read **Status** below, then the last few entries of the **Log** at the bottom.
2. Take the first step that is not done and not blocked. Steps marked **needs a phone** can only be finished by the owner on real hardware: prepare everything up to that point, say exactly what to try, and move on to the next step.
3. Before ending a session, update **Status**, move any settled question from **Open decisions** to **Decisions**, and add a dated **Log** entry: what was done, what was measured (with numbers), what is next.
4. The repo's conventions in `CLAUDE.md` still apply: a number behind every accuracy claim, an experiment before a change, a look at unmerged branches before reworking a file.

## Status

| Step | State |
|---|---|
| 0. Spikes | **S1** desktop half done (numbers in the Log); phone half needs a phone. **S2, S3**: test page written and checked in desktop Chromium (`web/spikes/phone-capabilities/`); needs hosting (O1), then a phone. **S4**: done locally (Worker under `wrangler dev` and a Node dev relay; cost model in the Log); deploying needs the owner's Cloudflare account. **S5**: in progress. |
| 1. Browser core | **Done.** `splinewire/session.py` and `bridge.py`, the Pyodide worker (`web/src/core/worker.ts`), `npm run assets`, a throwaway page that measures a dropped photo (`web/index.html`), and the Playwright test, which passes in CI (`.github/workflows/web.yml`, first run 2026-11-17). |
| 2. Phone page | Built (`web/phone/`, `web/src/phone/`) and checked end to end in Chromium at phone size with touch. The exit criterion needs a real phone: the owner's checklist is in the Log (2026-11-17, phone page). |
| 3. Pairing and relay | In progress. Done: the client side (`web/src/relay/`), the relay server (`web/relay/`), output files in the message (D14), all tested with real sockets and in a two-tab browser check. The desktop page (`web/desktop/`) is done and checked end to end against a test phone (`npm run check:desktop` in `web/relay/`, 12 steps). The phone page's half is in progress with step 2. The exit criterion needs hosting and a phone. |
| 4. Hosting, add-in install path, retire the .exe | Deploy workflow written, not run: blocked on O1 and O2 (owner). Add-in install path (O5) built, to be tried in Fusion by the owner. |
| 5. Live camera overlay | Backbone written (`web/src/overlay/`, `splinewire/live.py`), tested on drawn frames. Thresholds uncalibrated, no camera code, no page. Depends on S3 and S5. |

The code map is [web/README.md](../web/README.md). Nothing has run on a phone yet.

## Goal

- Open a website on the phone, photograph the chain, and have the curve appear on a computer that has the same site open. Copy points there, Paste points in Fusion.
- No install apart from the Fusion add-in. No shared Wi-Fi: phone and computer may be on different networks.
- Photos stay on the phone. Only the measured curve travels.
- Later: a live camera view with the detected chain drawn over it, which coaches the shot and takes it.

Not goals: accounts, cloud storage of photos or measurements, a native phone app, changes to how the measurement works.

## The design

```
Phone page (HTTPS)                         Relay                Desktop page (same site)
photo → detect → order → rectify →     ~1 KB message      curve preview, Copy points,
contact → pin editing, object side  →  keyed by a pairing →  downloads (DXF, SVG, CSV)
                                       code from a QR             → Fusion add-in: Paste points
```

### Decisions

| # | Decision | Why |
|---|---|---|
| D1 | All measuring runs in the browser on the phone. | No photo upload, no server compute, works on any network. |
| D2 | The first version runs the existing Python package in the page with Pyodide, in a web worker. | The accuracy figures and the 2-minute test suite belong to this code. A rewrite would have to earn them again. |
| D3 | Python stays the reference implementation. Any TypeScript port is checked against it with golden files (same photo in, same pins out within a stated tolerance). | Keeps `synthetic`, `scene`, the benchmark and the CLI as the test oracle. |
| D4 | The site is static files. The only server is a relay that forwards small messages between two paired pages. | Cheap, nothing to secure but the pairing. |
| D5 | The message to the desktop carries the curve and its diagnostics, never the photo. | Privacy and size. A small preview image may be added later if wanted. |
| D6 | The clipboard format is unchanged: the `x_mm\ty_mm` table from `output.points_tsv`. The Fusion add-in is not touched. | The add-in works and has been used with a real chain. |
| D7 | The live overlay coaches and triggers. The measurement comes from one still taken at that moment and run through the full pipeline. | Video frames are smaller, may be cropped by stabilisation, and blur with motion. |
| D8 | The .exe and the local server stay in the repo and keep working until step 4's exit criterion is met. | The real-photo test matrix (`next-steps.md`, item 1) is still in progress on it. |
| D9 | The desktop page can also measure a photo dropped onto it, using the same worker. | Parity with today's app, and the easiest way to test the core without a phone. |
| D10 | Front end: TypeScript and Vite, no UI framework; tests with Vitest. Logic is kept apart from the browser (sockets, workers, storage, timers and the camera are passed in), so it is tested in Node. (Was O3.) | The hard parts are protocol and state logic, which need tests more than they need components. |
| D11 | Relay messages are encrypted end to end: one random pairing secret in the QR link's `#fragment`; HKDF gives the room id (which the relay sees) and an AES-GCM key (which it does not). (Was O4.) | The relay then needs no trust and no storage. |
| D12 | The relay only forwards. Delivery is the pages' job: the phone keeps each curve until the desktop acknowledges it, and the desktop stores each curve once by its id. | A dropped frame, a closed tab or a relay restart then loses nothing, and the relay stays a few dozen lines. |
| D14 | The curve message carries the output files (JSON, CSV, SVG, DXF, Fusion CSV), gzip-compressed in the phone with `CompressionStream`, so the desktop page offers every download without loading Pyodide. | Python stays the only writer of the files (D3). For a 13-pin chain they are 24 KB, 7.5 KB gzipped (the DXF is 18 KB of boilerplate, 4.9 KB gzipped): about 10 KB sealed, well under the relay's 64 KB frame cap, and 100 kept curves fit in `localStorage`. The desktop page stays light and opens at once. |
| D13 | The overlay splits the work: TypeScript finds bright dots in every video frame (`blobs.ts`, about 4 ms); the Python `order` and `rectify`, in the worker, say which are the chain and how the camera is tilted (`live.py`, about 50 ms native). | Ordering and rectifying are the subtle parts and already exist; only the pixel work has to be fast. |

### Open decisions

| # | Question | Leaning | Who decides |
|---|---|---|---|
| O1 | Where the static site is hosted. | GitHub Pages, deployed by a workflow. Free, HTTPS, already where the code lives. | Owner |
| O2 | Where the relay runs. | A Cloudflare Worker with a Durable Object per pairing (WebSocket). Needs an account. | Owner |
| O5 | How the Fusion add-in gets installed once there is no .exe to do it. | Built as leaned (2026-11-17): the desktop page's "No Paste points button in Fusion?" offers `SplineWire-fusion-addin.zip` (made by `npm run assets` from `fusion/SplineWire`) and three steps: unzip anywhere, add the folder in Fusion's Scripts and Add-Ins dialog, Run on Startup. The menu names in the steps are unchecked against a current Fusion. | Owner: confirm, and try the steps in Fusion |
| O6 | Whether to port the pipeline to TypeScript at all. | Not yet. S1 on desktop: it works, but memory (580 MB of wasm heap for a 12 MP photo) and a 10 s load are close to what a phone allows. Try the cheap levers first (see O8), then measure on a phone. | After S1 on a phone |
| O8 | How to cut the core's memory and load time. | In order: detect on a downscaled image and refine at full size in tiles (detection's temporaries are 293 MB of the 580); load SciPy and ezdxf lazily (3.5 s of the 10 s); pin the native OpenCV to Pyodide's 4.11 so golden files match exactly. Each needs the benchmark before and after. | Agent |
| O7 | The Fusion add-in receiving points from the relay directly, in an HTML palette, with no desktop tab and no clipboard. | Worth a spike after step 4. Not checked. | Later |

## What carries over, what changes, what goes

| Today | Web build |
|---|---|
| `detect`, `order`, `rectify`, `contact`, `edits`, `pipeline.measure` | Unchanged, run under Pyodide. `measure` already takes an image array, not a path. |
| `process.process_photo` (paths in, files out) | A new entry point beside it that takes photo bytes and returns JSON. See step 1. |
| `output` writers (JSON, CSV, SVG, DXF, Fusion CSV) | Same functions, writing to memory. The page turns them into downloads. |
| `webapp/server.py` (945 lines: photo store, routes, settings, re-measure after edits) | Its job moves into the page. Photos and results live in the browser (IndexedDB), settings in `localStorage`. |
| `webapp/static/desktop.html`, `phone.html` | Starting point for the new pages. They carry the Test bench redesign (merged 2026-11-17); the phone page already has a fix view and a tap for the object side. |
| Pin editor (mouse: wheel, drag, click) | Needs a touch version: tap to add or remove, pinch to zoom, drag to pan. |
| Focal length: override, EXIF, saved setting, estimated from the chain | Same order. The saved setting becomes per-phone browser storage. |
| `data/chain*.yaml` presets | Bundled into the site at build time. |
| `fusion/SplineWire/` | Unchanged. Its installer (`splinewire/fusion_addin.py`) needs a replacement, see O5. |
| `packaging/`, `windows-exe.yml`, `webapp/firewall.py`, `selftest.py`, the QR code for the LAN address | Retired in step 4, not before (D8). |
| CLI, `synthetic`, `scene`, `experiments/`, chain CAD generators, pytest suite | Stay as developer tools. |

## Steps

Each step has an exit criterion. Keep steps small enough to finish and verify in a session.

### 0. Spikes

Answer these before building on them. Each ends with numbers in the Log.

- **S1. Does the pipeline run acceptably under Pyodide?** Load `splinewire` and its dependencies in a page and measure a 12-megapixel synthetic photo (`uv run splinewire synth --env shadow`) and one real chain photo. Record cold load time, warm load time, download size, time to measure, peak memory, and whether the pins match the CLI's to 0.01 px. Do it in desktop Chromium first (Playwright is installed in cloud sessions), then on a phone. Things to check rather than assume: that NumPy, SciPy, OpenCV, Pillow and PyYAML come from the Pyodide distribution at versions the code accepts; that `ezdxf` installs with `micropip`; that nothing imports `pillow-heif` at module load (it has no browser build, and phones hand the browser a JPEG anyway). iOS Safari's per-tab memory limit is the main worry. **The phone half needs a phone.**
- **S2. Does the focal length survive?** On the hosted test page, read EXIF from the file the browser hands over, for each way of getting a photo: camera through `<input type="file" capture>`, pick from the library, and on Android `ImageCapture.takePhoto()`. The LAN upload page lost all EXIF on an iPhone on 2026-10-01 (`next-steps.md`); which button was used was not recorded. **Needs a phone, and a hosted page (do a bare deploy for O1 first).**
- **S3. What can a live camera stream give?** On iOS Safari and Android Chrome: the largest video resolution `getUserMedia` offers on the rear camera, whether a full-resolution still can be taken without leaving the page, and whether the stream's field of view matches the still camera's. **Needs a phone.**
- **S4. Relay.** Smallest working relay for O2, two browser tabs paired by a code, a message passed, reconnect after a dropped socket. Record the cost model and limits of the service chosen.
- **S5. Is a video frame accurate enough to measure from?** No hardware needed: render the benchmark scenes at 1920×1080 and at 3840×2160 with the chain filling the frame as a person would hold it, add motion blur, and compare worst-pin error with the 12-megapixel figures, with the true focal length and with it estimated from the chain. This says whether D7 could be relaxed and what the overlay's own detector must achieve.

Exit: S1 and S5 measured and logged; S2 and S3 have a page the owner can open on a phone, and their results are logged when the owner has tried it.

### 1. Browser core

- Add an entry point in `splinewire` that works on bytes and returns JSON-ready data, with no file system: photo bytes, chain spec, focal override, saved focal, side, edits in; measurement, warnings, notes, focal source, photo info and the output files' contents out. `process_photo` should become a thin wrapper over it so the two cannot drift.
- Keep detections between calls so an edit re-measures without detecting again, as `process_photo`'s `loaded` and `detected` arguments do now.
- Build the package as a wheel the page loads from its own origin. Dependencies the browser does not need (`pillow-heif`, `segno`) must not block the install.
- A worker script that loads Pyodide and the wheel, and exposes `measure`, `remeasure` and `outputs` over `postMessage`. Errors follow the repo's rule: a `ValueError`'s message is shown to the person as is, anything else is a bug and is shown with its type.
- Tests: pytest for the new entry point; a Playwright test that loads the worker, feeds it a synthetic photo and compares the pins with the truth file.

Exit: the Playwright test passes in CI, and a throwaway page measures a dropped photo.

### 2. Phone page

- Take or pick a photo, show progress while Pyodide loads (cache it with a service worker so the second visit is quick), show the measured curve and the photo with detections.
- Touch pin editor and object side, per photo, as the desktop editor does now.
- Chain preset, pin count and saved focal length in settings.
- The phone page stands on its own: Copy points, share and downloads work there with no desktop paired. That is also the fallback when the relay is down.
- Version and build shown on the page, as `packaging/stamp_build.py` does for the .exe.

Exit: on a real phone, a photo of the chain gives the same pins as the CLI on the same file, and a wrong pin can be fixed by touch. **Needs a phone.**

### 3. Pairing and relay

- Desktop page shows a QR code. The phone opens it and the two are paired. Both remember the pairing, so it is done once per phone and computer, not once per measurement.
- Message, versioned from the start: chain name, pitch, pins and contact points in mm, side, warnings, notes, focal source, time, a client-made id. The desktop keeps a list of received curves.
- Desktop: curve drawing in mm, Copy points (and Ctrl+C), downloads, warnings in orange as now.
- Handle the dull cases: desktop tab closed when the phone sends (queue on the phone and say so), relay unreachable, pairing expired.

Exit: phone on mobile data, computer on another network, curve arrives and pastes into Fusion.

### 4. Hosting and retiring the .exe

- Deploy workflow for the site and the relay. A staging address for testing branches.
- Add-in install path (O5).
- Re-run the real-photo test matrix through the web build and compare with the .exe's results on the same files.
- Then remove `packaging/`, the Windows workflow, `webapp/server.py`, `firewall.py` and `selftest.py`, and rewrite the README, `CLAUDE.md` and `architecture.md` sections that describe them.

Exit: the same photos give the same numbers in the web build as in the .exe, and the owner agrees to remove it.

### 5. Live camera overlay

Depends on S3 and S5.

- A light detector that runs on video frames at interactive speed. Whether that is a TypeScript port of part of `detect` and `order` or the Python under Pyodide at a reduced size is decided by measuring both.
- Overlay: pins found out of pins expected, the chain traced over the video, tilt, warnings for glare and blur. Steer the person toward a straight-on view: in simulation a 2% focal error costs up to 0.05 mm at 25° tilt and 0.11 mm at 40° (`open-questions.md`), so straight-on shots forgive a poor focal length.
- Take the still when the overlay has been good for a moment, and measure it with the full pipeline (D7).
- Video frames carry no EXIF. Remember the focal length from any still of this phone that had it; otherwise the saved setting, otherwise the estimate from the chain.

Exit: on a real phone, the overlay tracks the chain and the automatic shot measures as well as a hand-taken one. **Needs a phone.**

## Work queue

The next tasks, in order, each small enough for one agent in one session. Each names the backbone it must build on; the backbone's contracts (web/README.md, "Rules of the road") are changed only on both sides at once, with tests.

1. ~~Finish step 1~~ (done).
2. ~~Finish the relay~~ (done).
3. ~~Files in the curve message (D14)~~ (done: `src/relay/files.ts`).
4. ~~Curve from a measurement~~ (done: `src/phone/curve.ts`).
5. **Phone page (step 2)** (built; left: the real-phone checklist). `phone/index.html` + `src/phone/`: take or pick a photo (keep it in IndexedDB, keyed by the id `CoreClient` uses), progress while the core loads (start `preload()` at page open), the photo with pins drawn from `view` using `src/editor/` (gestures, `editsAfterTap`, `EditHistory`, viewport), object side by a tap, warnings in orange, Copy points (`pointsTsv`), share and downloads. Settings: chain preset, pin count, saved focal length (localStorage). Pairing status from `RelayLink` and the outbox list ("queued: the computer's page is not open"). A service worker that caches the Pyodide assets so the second visit is quick. Version and build on the page.
6. ~~Desktop page (step 3)~~ (done, at the site root, with D9). `index.html` + `src/desktop/`: the pairing QR code (`desktopSecret`, `pairingUrl`; a small QR encoder from npm, pinned), the inbox list newest first, the curve drawn in mm, Copy points and Ctrl+C, downloads from `files`, warnings in orange, unpair. D9 (measure a dropped photo) reuses the phone page's photo view.
7. **Hosting (step 4), once the owner has answered O1 and O2.** `.github/workflows/deploy.yml` is written (GitHub Pages and the relay, run by hand only); `web/README.md`, "Deploying", lists what the owner sets up. Left: run it, a staging path for branches, then S2 and S3 and the phone checklist on the owner's phones.
8. **Core memory and load (O8).** Each lever measured with `experiments/pyodide_spike/` and the benchmark before and after.
9. **Overlay page (step 5), after S3 and S5.** Camera, `requestVideoFrameCallback`, work frames through `blobs.workSize` and `toGray`, `OverlaySession`, the still taken with `ImageCapture` where it exists and the file input otherwise. Calibrate the provisional limits in `src/overlay/coach.ts` with S5's numbers and recorded phone video.

## Risks

| Risk | What would show it | Fallback |
|---|---|---|
| Pyodide is too heavy for phones (download, load time, memory with a 12-megapixel image). | S1 | Downscale before detecting and refine at full size in tiles; or port to TypeScript (O6); or measure on the desktop page with the photo sent peer to peer. |
| iOS strips EXIF on every path. | S2 | Already handled: saved per-phone focal length, or the estimate from the chain. One calibration shot of the test plaque could set the saved value. |
| iOS gives no full-resolution still inside the page. | S3 | Measure from the best video frame if S5 allows, or hand off to the system camera for the final shot. |
| Browser builds of OpenCV or SciPy differ numerically from the desktop ones. | S1's pin comparison | Golden-file tolerance, and investigate any difference above 0.01 px. |

## Log

Newest last. One entry per session: date, what was done, numbers, what is next.

- **2026-11-17.** Plan written from a design discussion. Checked for this doc: the pipeline's OpenCV use is thresholds, connected components, contours, ellipse fits, `remap` and `findHomography`, plus SciPy's `least_squares` and `CubicSpline`; `measure` takes an array. `claude/test-bench-ui` (the redesign of both pages) was merged to `main` together with this doc. Nothing run in a browser yet. Next: S1 and S5, and ask the owner about O1 and O2.
- **2026-11-17/18, overnight.** Step 0 and the backbone of steps 1, 3 and 5.
  - **S1, desktop half** (`experiments/pyodide_spike/`, reproduce with its README). Pyodide 314.0.8 runs the package unchanged in a module worker. Pins on two 12 MP synthetic photos are identical (0.0 px) to native Python on the same OpenCV (4.11); against the locked native OpenCV 5.0 the worst pin differs by 0.043 px (0.003 mm), so the difference is the OpenCV version, not the browser. Download 37.9 MB compressed (SciPy 13.9, OpenCV 10.7). Load about 10 s even with a warm cache (imports 4.3 s, of which `scipy.interpolate` 2.1 and `ezdxf` 1.4). A 12 MP photo measures in 1.4–2.7 s, about 2–2.5× native. Wasm heap 162 MB after load, 555–584 MB after one photo, and it never shrinks; the renderer process peaked at 1.04 GB. The machine was heavily loaded, so absolute times are pessimistic. A plain `micropip.install` of the wheel fails on the `opencv-python-headless` name: install with `deps=False`. Classic workers are refused: use a module worker. No real chain photo is in the repo, so that comparison was not done.
  - **S2/S3**: `web/spikes/phone-capabilities/` is a single static page that records, for each way of getting a photo, what EXIF survives (parsed in the page; its focal length in pixels agrees with `camera.focal_px_from_exif` to 0.001 px on seven test files) and what the camera stream offers. Its Playwright check passes 96 of 96 in desktop Chromium. Not tried on a phone: it needs HTTPS hosting (O1). The owner's steps are in its README.
  - **Step 1**: `splinewire/session.py` is the file-free entry point (`PhotoSession`: bytes in, one JSON document out, detections kept between calls). `process_photo` is now a wrapper over it and the output writers build text first, so the CLI, the local app and the browser share one implementation; the existing process, CLI and app tests pass unchanged.
  - **Step 3**: the pairing and delivery logic (D11, D12), tested end to end against an in-memory relay: desktop closed while the phone sends, a lost acknowledgement, the relay down, a dead connection, a second phone, a second pairing. The clipboard table written in TypeScript is byte for byte `output.points_tsv`'s.
  - **Step 5**: the overlay's detector, coach and loop (D13). `blobs.ts` finds every dot of a drawn chain on dark, gray and white tables to within 0.25 px at 640×360 in about 4 ms. `live.analyse` recovers a 25° tilt to within 1° from projected pins among clutter, in 51 ms with the focal length and 258 ms estimating it (native, loaded machine). The coach's limits for span, motion and sharpness are provisional.
- **2026-11-17, afternoon.** The overnight subagents were cut off by a usage limit and resumed after it reset.
  - **S4, relay** (`web/relay/`, its README has the deploy steps). A Cloudflare Worker with one hibernating Durable Object per room, and the same relay on Node for development; both thin shells over `rules.ts`. Real-socket tests: dev relay 18 passed (1 Worker-only test skipped), the Worker under `wrangler dev` 10 of 10, including a relay restart mid-session. The two-tab browser check passes 6 of 6: pairing link, curve delivered and acknowledged, desktop reload keeps the pairing and list, a curve queued while the relay was down arrives once. Real sockets showed one client bug: an attempt that never answers left the link connecting for good; it now times out after 10 s. Cost (Cloudflare docs, checked 2026-11-17): free plan 100,000 Durable Object requests and 13,000 GB-s a day; incoming WebSocket messages count 20 to 1; a hibernating object costs no duration. Opening a page costs 2 requests, a curve and its ack about 0.1: a handful of users is under 1% of the free plan. Not deployed: needs the owner's account (O2).
  - **D14**: the curve message carries the output files. A 13-pin curve's 24.7 K of text packs to 9.8 K; unpacked on the desktop side they equal Python's byte for byte.
  - **Touch editor logic** (`web/src/editor/`): viewport, gestures (a second finger never ends in a tap), and the desktop editor's tap rules with 22 px finger targets. 12 tests.
  - **Step 1 done.** `splinewire/bridge.py` is the worker's one entry point (`call(cmd, args_json, data)`; a ValueError is a message for the person, `NotOpen` means "open the photo again", anything else is a bug; at most 2 photos open, the least recently used closed first). `npm run assets` puts Pyodide 314.0.8, the sha256-checked packages, ezdxf and the package's wheel under `web/public/` (43.2 MiB). The Playwright test passes 10 of 10 here and in CI. Desktop Chromium, headless: load 7.7–9.1 s cold and 8.2–11.4 s warm (unpacking and imports dominate, not the download); measure 2.1–2.4 s (4000×3000 pipe) and 1.2–1.6 s (12 MP shadow scene); worst pin against native 0.010 and 0.041 px (OpenCV 4.11 against 5.0), truth error the CLI's to 0.0001 mm; measuring again after an edit 82–166 ms against 1.2–1.3 s to detect again; the first `outputs` 1.4 s (importing ezdxf, which is now kept out of the first measurement), then 11 ms. Wasm heap 129 MiB after load, 555 MiB with one photo and still 555 MiB with two open and after a third: closing photos does reuse the memory. Correction: `live.analyse` is 51–105 ms only when the focal length is known; estimating it costs 280 ms native and 400–640 ms in the browser, nearly all in `least_squares`. The overlay must therefore pass a focal length (the phone's saved or EXIF value, scaled to the video frame) or analyse less often.
  - **Known gaps in step 1.** The tsv made in the browser differs from native Python's by up to 0.0002 mm (the OpenCV versions), so the e2e test compares values, not bytes. `scipy.interpolate` still costs about 2 s of the load (O8). CI has run once.
  - **Desktop page** (`web/desktop/`, `web/src/desktop/`). Pairing QR code (`qrcode-generator` 2.0.4, decoded back exactly by OpenCV's QR reader), the curve list, the curve drawn 1:1 in mm with a TypeScript port of `spline_samples` (within 1e-6 mm of Python), diagnostics, warnings in orange, Copy points and Ctrl+C (Python's `points_tsv` byte for byte), all six downloads with Python's names and bytes, delete, unpair. The relay address: `?relay=`, then the build's `VITE_RELAY_URL`, then the dev relay. Its storage has its own prefix, so a phone page open in the same browser keeps its own pairing. The end-to-end check passes 12 of 12 against the dev relay and a test phone that sends a curve made from Python's measurement.
  - **Phone page** (`web/phone/`, `web/src/phone/`). Photos are kept in IndexedDB (the original, a 2560 px preview, a 128 px thumbnail; the 30 newest) and never leave the phone. The engine starts loading when the page opens, and a photo taken meanwhile is measured when it is ready. Touch editing uses `src/editor`; taps are ignored during the 90–270 ms re-measure, since a tap's meaning depends on the last pins. Sending is a button, not automatic: the first measurement is the one most likely to need a pin fixed. A curve changed after sending goes as a new id, and undoing back to a sent curve offers nothing new. A service worker serves the 15 core files on the second visit (0 bytes from the server); on localhost that saves no time (7.3–8.3 s either way, unpacking dominates), only the download: about 30 s at 10 Mbit/s. Measure 1.7–2.1 s for a 4000×3000 photo, pins within 0.010 px of native, wasm heap 464–556 MiB. The e2e test passes 11 of 11 (touch tap, pinch and drag, side, copy, reload, an error photo, pairing and delivery with the computer's page closed and reopened).
  - **Owner's checklist for the phone page**, on iPhone Safari and Android Chrome once the site is hosted: (1) time from opening to ready, first visit; (2) the same on a second visit, and in airplane mode (it should still load); (3) a chain photo through Take photo, its JSON's pins against `splinewire measure` on the same file (step 2's exit criterion); (4) whether "26 mm in the file" or "no focal length in the file" shows, for Take photo and for Choose from library; (5) on iPhone, whether a library photo arrives as JPEG or HEIC (the core reads HEIC since 2026-11-17, loading the decoder on first need; note the time); (6) tap to remove and restore a pin, pinch and drag add nothing, the page still scrolls outside the photo; (7) five 12 MP photos in a row without the tab being reloaded; (8) Copy points pasted into Notes starts with `x_mm<TAB>y_mm`; (9) which apps Share offers, and whether it refuses `.dxf`; (10) where a DXF download lands; (11) scan the computer's code, send a curve, close the computer's tab, send again (queued), reopen (delivered); (12) dark mode, a notched phone, 360 px wide.
  - **HEIC.** Pyodide ships pillow-heif. The worker loads it (0.56 MB with cffi) only when a photo's first bytes are a HEIF brand, and registers Pillow's opener. A HEIC of the pipe photo measures in the browser with its EXIF focal length and pins within 0.05 px of native Python, in 5.4 s including the decoder's first load. Outside Safari the browser itself cannot decode HEIC, so the phone page shows no preview for one; the measurement still works.
  - **Desktop page at the root, D9, CI.** The desktop page is the site's index; the throwaway test page is gone. A photo dropped or picked on it is measured with the same worker, loaded only then (the page opens with 57.6 KiB of JS, 23 KB gzipped; dropping a photo adds 33.9 KiB and then Pyodide), edited by mouse with the photo editor now shared with the phone page (`src/photo/`), and added to the list as "measured here". Pick to pins on screen 9.8–10.9 s with the engine's load, 0.8–1.0 s once it is loaded. CI runs every check: typechecks, unit tests (142), the relay against the dev relay and the Worker, the e2e specs (core 10, desktop 7, phone 11), the two-tab and desktop checks, and the phone-capabilities check. The site needs no cross-origin isolation (no `SharedArrayBuffer` in the site or Pyodide), so GitHub Pages, which cannot send those headers, is enough. `web/package.json` carries splinewire's version (0.7.0) and the assets script refuses a mismatch.
  - **CI.** With every check in it, the Web workflow passes on GitHub (runs 9 and 10, about 5 minutes each). Run 8 had failed on a timing flake in the phone test: ezdxf writes the creation and update time into each DXF, and the test compared two files written moments apart while masking only their GUIDs. It now masks the dates too.
