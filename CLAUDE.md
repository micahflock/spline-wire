# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Curve Capture (spline-wire)

## Overview

Capture arbitrary real-world curves — pipe profiles, molding, doorknob silhouettes, anything awkward for a ruler or calipers — and drop them into CAD as reference geometry. The user wraps a stiff, self-holding planar chain around the target curve, lifts it off (the chain keeps its shape), lays it flat and photographs it with a phone. Software turns the photo into points on the target curve for a CAD sketch.

Primary CAD target: **Autodesk Fusion**. Minimize user interaction between "measure with chain" and "points in CAD."

## How it works

- **Hardware:** a chain of rigid links of fixed pitch (pin-to-pin distance, ~10 mm), with enough joint friction to hold a pose. One unlabeled fiducial, a light dot (a ring is also supported), sits on every pin, on one face only. A chain printed the other way round (dark fiducials on light links) is a setting, `fiducial_shade: dark` (Colours in the apps, `--dark-fiducials` on the CLI): detection reads its photo inverted.
- **Detection:** classical CV (OpenCV) finds dot centers to ~0.1–0.2 px. No LLM, no labels.
- **Ordering:** pins are put in chain order geometrically (walk ~one pitch at a time, turning as little as possible).
- **Deskew:** all pins lie on one plane and consecutive pins are exactly one pitch apart. With the camera focal length from EXIF, that fixes the plane's tilt and distance, so perspective is removed without any reference object in the photo.
- **Contact offset:** the target touches the chain's edge, not its pin line. Offset by the link half-width at link midpoints (convex bends) or at pins (concave bends).
- **Output:** curve points (JSON/CSV), a mm DXF and a 1:1 SVG, plus Copy points (desktop page) → Fusion add-in Paste points for a two-click path into a sketch.
- **App:** a website (`web/`). The phone page photographs and measures in the browser; only the curve goes, through a relay, to the paired desktop page.

## Repo layout

- `splinewire/` — the pipeline: `detect` → `order` → `rectify` → `contact`, glued by `pipeline`; `edits` holds a person's corrections to the detections (pins removed or added by clicking the photo, snapped to the dot under the click, then fitted to the pitch); `process` measures one photo and writes its files for `cli`; `session` does the same from photo bytes to one JSON document (each photo with its own pin count and object side), and `bridge` is the Pyodide worker's entry point to it. `synthetic` renders ideal test photos and `scene` realistic ones (FDM print defects for a 0.4 mm nozzle, tables, clutter, glare, shadows, defocus, noise, JPEG; named `PRESETS`); `fiducials` defines the dot, the ring and the alternative fiducial designs compared in `docs/cv-robustness.md`. `testpart` renders a paper test chain and `plaque` a 3D-printable one (black/white, one filament swap). The print-in-place and O-ring chain generators were retired on 2026-11-18 (`docs/printed-chain.md` keeps their design).
- `fusion/SplineWire/` — Fusion add-in: a Paste points button that reads the table the desktop page's Copy points puts on the clipboard (mm) and adds sketch points plus a fitted spline (API units are cm). The desktop page offers it as a zip (`npm run assets`). Feasibility notes: `docs/fusion-import.md`.
- `web/` — the website: phone and desktop pages, the TypeScript port of the pipeline, the relay (`web/relay/`). Code map: `web/README.md`. `.github/workflows/web.yml` tests it (`python.yml` runs pytest); `deploy.yml` (run by hand) puts it on GitHub Pages and the relay on Cloudflare. The Windows app (`SplineWire.exe`, `packaging/`, `splinewire/webapp/`) was retired on 2026-11-18 (D24 in `docs/web-build.md`): don't build, test or document it.
- **The snap-fit chain is the default** (the owner's choice, 2026-11-18): `data/chain-snap.yaml` (a hand-designed one-part link, 13 mm pitch, currently 3DP_Link_Ver_MF: `contact: round`, since its round sleeves and heads at the pins touch the target rather than a link edge, and a thin white rim round each head's top edge that `detect_rims` finds by its hole; `synthetic.render_snap_photo` draws it with its relief; `experiments/snap_link_geometry.py` reads the numbers from the link's STEP file, kept in `experiments/snap_link/`). `chain.DEFAULT_PRESET` is `snap`: the CLI's `measure` and `synth`, the web pages, the decode-check page and the end-to-end tests start with it. Use it for anything tested by hand or on a phone; don't default to the O-ring chain. `data/chain.yaml` is the O-ring dot chain (preset `oring`, `chain.oring_chain_path()`): pitch, half-width, fiducial type and size, pin count, the sharpest bend the ordering accepts (125° for the O-ring chain, 80° for the print-in-place one). The printed test plaque, the realistic scenes (`synth --env`), the benchmark and the regression tests are built around it, so it stays as it is.
- `tests/` — pytest suite; end-to-end tests run on synthetic photos with known geometry.
- `experiments/` — standalone studies: `cv_benchmark` (detection across simulated environments, current code vs a git revision), `fiducial_study` (fiducial designs), `lens_distortion`, `self_rectification`, `relief_bias`.
- `docs/architecture.md` — design and rationale.
- `docs/cv-robustness.md` — simulated environments, benchmark results, fiducial options for a 0.4 mm nozzle.
- `docs/printed-chain.md` — design record of the retired print-in-place and O-ring chains.
- `docs/snap-chain.md` — the snap-fit chain: the link, what it changes in contact, detection and ordering, open items.
- `docs/web-build.md` — the move from the .exe to a hosted website (measuring in the browser on the phone, curve relayed to the desktop page): decisions, steps, status and a session log. Read it first when asked to continue the web build, and update its Status and Log before finishing.
- `docs/open-items.md` — what should be done, in risk order (the one list of open items).
- `docs/future-work.md` — what might be done later.
- `docs/real-world-tests.md` — the planned real-world test series (open item 1): what to build, the phases, pass criteria.

## Conventions

- **Errors for people are `ValueError`s.** The pipeline raises `ValueError` with a message a person can act on ("found 4 pins in a row…"); the CLI prints it and exits 1, the pages show it on the photo. Anything else is a bug and is shown with its type.
- **Every accuracy claim has a number behind it.** Check a suspected problem with an experiment before changing the code, and a change to `detect`/`order`/`pipeline` with `experiments/cv_benchmark.py --baseline <rev>` (per-photo records with `--json`). The benchmark's figures in the docs were measured at `max_bend_deg: 80`.
- **The browser measures with a TypeScript port** of the pipeline (`web/src/pipeline/`, decision D18 in `docs/web-build.md`); Python stays the reference. A change to `detect`, `order`, `rectify`, `contact`, `edits`, `pipeline`, `session` or `output` must be made in the port too: regenerate the golden files (`npm run golden` and `web/scripts/golden_*.py`, from `web/`) and `npm test` shows where the two differ.
- **Look for unmerged work first.** Finished branches are sometimes left on the remote without a pull request (`git ls-remote --heads origin`). Before reworking a file, check whether one of them changes it, and keep them mergeable (`git merge-tree --write-tree HEAD <branch>`).

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests (~2 min)
uv run splinewire synth --shape pipe      # synthetic photo of the default (snap-fit) chain + truth -> out/synth/
uv run splinewire synth --env shadow      # realistic printed plaque (the O-ring dot chain) in a simulated environment
uv run splinewire synth --chain data/chain.yaml --shape pipe        # the O-ring dot chain (also on measure)
uv run python experiments/cv_benchmark.py --baseline HEAD   # detection benchmark, ~10 min first run (renders cached)
uv run splinewire measure out/synth/pipe.jpg --truth out/synth/pipe-truth.json
uv run splinewire test-part               # printable paper SVG + truth -> out/test-part/
uv run splinewire test-plaque             # 3D-printable plaque STLs + truth -> out/test-plaque/
cd web && npm install && npm test        # the website's unit tests (more in web/README.md: assets, dev, e2e, golden)
```

The pages show the version and CI build in their footers. `npm run assets` bundles every `data/chain*.yaml` into `chains.json`, so a new `chain-<name>.yaml` preset reaches the pages without listing it.

## Priority

The riskiest open item is real-world accuracy with the snap-fit chain. The owner's tests on the live site show good real-photo accuracy, but no figures are recorded yet and no real measurement has been checked against an object of known size; the old test matrix (Windows app, O-ring chain) is dropped and a revised test series is planned in `docs/real-world-tests.md` (open item 1).
