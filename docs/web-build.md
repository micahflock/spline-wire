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
| 0. Spikes (S1–S5) | not started |
| 1. Browser core: `splinewire` running in the page | not started |
| 2. Phone page: photo in, curve out, pin editing | not started |
| 3. Pairing and relay: points reach the desktop page | not started |
| 4. Hosting, add-in install path, retire the .exe | not started |
| 5. Live camera overlay | not started |

Nothing of the web build exists in the repo yet. This doc was written on 2026-11-17 from a design discussion; no part of it has been tested.

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

### Open decisions

| # | Question | Leaning | Who decides |
|---|---|---|---|
| O1 | Where the static site is hosted. | GitHub Pages, deployed by a workflow. Free, HTTPS, already where the code lives. | Owner |
| O2 | Where the relay runs. | A Cloudflare Worker with a Durable Object per pairing (WebSocket). Needs an account. | Owner |
| O3 | Front-end tooling. | TypeScript and Vite, no UI framework. Today's pages are single HTML files with no build. | Agent, in step 2 |
| O4 | Whether the relay messages are encrypted end to end. | Yes: a secret in the QR link's `#fragment`, AES-GCM through WebCrypto, the room id a hash of the secret. Cheap to do from the start. | Agent, in step 3 |
| O5 | How the Fusion add-in gets installed once there is no .exe to do it. | A zip download and three lines of instructions on the site. | Owner, in step 4 |
| O6 | Whether to port the pipeline to TypeScript at all. | Only if S1 shows Pyodide is too slow or too heavy on phones, or step 5 needs it. | After S1 |
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
