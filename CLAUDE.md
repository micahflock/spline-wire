# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Curve Capture (spline-wire)

## Overview

Capture arbitrary real-world curves — pipe profiles, molding, doorknob silhouettes, anything awkward for a ruler or calipers — and drop them into CAD as reference geometry. The user wraps a stiff, self-holding planar chain around the target curve, lifts it off (the chain keeps its shape), lays it flat and photographs it with a phone. Software turns the photo into points on the target curve for a CAD sketch.

Primary CAD target: **Autodesk Fusion**. Minimize user interaction between "measure with chain" and "points in CAD."

## How it works

- **Hardware:** a chain of rigid links of fixed pitch (pin-to-pin distance, ~10 mm), with enough joint friction to hold a pose. One unlabeled fiducial, a light dot (a ring is also supported), sits on every pin, on one face only.
- **Detection:** classical CV (OpenCV) finds dot centers to ~0.1–0.2 px. No LLM, no labels.
- **Ordering:** pins are put in chain order geometrically (walk ~one pitch at a time, turning as little as possible).
- **Deskew:** all pins lie on one plane and consecutive pins are exactly one pitch apart. With the camera focal length from EXIF, that fixes the plane's tilt and distance, so perspective is removed without any reference object in the photo.
- **Contact offset:** the target touches the chain's edge, not its pin line. Offset by the link half-width at link midpoints (convex bends) or at pins (concave bends).
- **Output:** curve points (JSON/CSV), a mm DXF and a 1:1 SVG, plus Copy points → Fusion add-in Paste points for a two-click path into a sketch.

## Repo layout

- `splinewire/` — the pipeline: `detect` → `order` → `rectify` → `contact`, glued by `pipeline`; `edits` holds a person's corrections to the detections (pins removed or added by clicking the photo, snapped to the dot under the click, then fitted to the pitch); `process` measures one photo and writes its files, shared by `cli` and the app in `webapp/` (a local web server with browser UI and phone upload over Wi-Fi; `settings` holds its per-user settings). `synthetic` renders ideal test photos and `scene` realistic ones (FDM print defects for a 0.4 mm nozzle, tables, clutter, glare, shadows, defocus, noise, JPEG; named `PRESETS`); `fiducials` defines the dot, the ring and the alternative fiducial designs compared in `docs/cv-robustness.md`. `testpart` renders a paper test chain and `plaque` a 3D-printable one (black/white, one filament swap); `selftest` checks a packaged build.
- `fusion/SplineWire/` — Fusion add-in: a Paste points button that reads the table SplineWire.exe's Copy points puts on the clipboard (mm) and adds sketch points plus a fitted spline (API units are cm). `splinewire/fusion_addin.py` installs it. Feasibility notes: `docs/fusion-import.md`.
- `packaging/` — PyInstaller build of the app (`SplineWire.exe`, a console program that serves the web app); `.github/workflows/windows-exe.yml` builds, tests and uploads it on Windows.
- `data/chain.yaml` — the chain's physical parameters (pitch, half-width, fiducial type and size, pin count).
- `tests/` — pytest suite; end-to-end tests run on synthetic photos with known geometry.
- `experiments/` — standalone studies: `cv_benchmark` (detection across simulated environments, current code vs a git revision), `fiducial_study` (fiducial designs), `lens_distortion`, `self_rectification`, `relief_bias`.
- `docs/architecture.md` — design and rationale.
- `docs/cv-robustness.md` — simulated environments, benchmark results, fiducial options for a 0.4 mm nozzle.
- `docs/next-steps.md` — status and near-term tasks, in risk order.
- `docs/open-questions.md` — unresolved design questions.

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests (~1 min)
uv run splinewire synth --shape pipe      # synthetic photo + truth -> out/synth/
uv run splinewire synth --env shadow      # realistic printed plaque in a simulated environment
uv run python experiments/cv_benchmark.py --baseline HEAD   # detection benchmark, ~10 min first run (renders cached)
uv run splinewire measure out/synth/pipe.jpg --truth out/synth/pipe-truth.json
uv run splinewire test-part               # printable paper SVG + truth -> out/test-part/
uv run splinewire test-plaque             # 3D-printable plaque STLs + truth -> out/test-plaque/
uv run splinewire-app                     # the app: opens http://localhost:8765/ (phone page via its QR code)
uv run splinewire-app --selftest log.txt  # headless check over HTTP, also run on the packaged .exe
uv sync --group build && uv run python packaging/build_exe.py   # dist/SplineWire(.exe)
```

PyInstaller can't cross-compile: the Windows .exe comes from the `Windows app` GitHub Actions workflow (artifact `SplineWire-windows`). The web pages in `splinewire/webapp/static` are bundled as data files; the selftest fetches them from the packaged build. The title and pages show the version and CI build (`packaging/stamp_build.py`).

## Priority

Almost everything is validated only on synthetic photos. The riskiest open item is real-world accuracy: real phone photos (lighting, glare, lens distortion, EXIF focal accuracy) of the printed test plaque. The first real photo (s-curve plaque, ~25° tilt) measured 0.098 mm worst pin; the rest of the test matrix is still to do. See `docs/next-steps.md`, item 1.
