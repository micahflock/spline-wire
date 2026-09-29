# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Curve Capture (spline-wire)

## Overview

Capture arbitrary real-world curves — pipe profiles, molding, doorknob silhouettes, anything awkward for a ruler or calipers — and drop them into CAD as reference geometry. The user wraps a stiff, self-holding planar chain around the target curve, lifts it off (the chain keeps its shape), lays it flat and photographs it with a phone. Software turns the photo into points on the target curve for a CAD sketch.

Primary CAD target: **Autodesk Fusion**. Minimize user interaction between "measure with chain" and "points in CAD."

## How it works

- **Hardware:** a chain of rigid links of fixed pitch (pin-to-pin distance, ~10 mm), with enough joint friction to hold a pose. One unlabeled ring fiducial sits on every pin, on one face only.
- **Detection:** classical CV (OpenCV) finds ring centers to ~0.1–0.2 px. No LLM, no labels.
- **Ordering:** pins are put in chain order geometrically (walk ~one pitch at a time, turning as little as possible).
- **Deskew:** all pins lie on one plane and consecutive pins are exactly one pitch apart. With the camera focal length from EXIF, that fixes the plane's tilt and distance, so perspective is removed without any reference object in the photo.
- **Contact offset:** the target touches the chain's edge, not its pin line. Offset by the link half-width at link midpoints (convex bends) or at pins (concave bends).
- **Output:** curve points (JSON/CSV) and a 1:1 SVG for CAD.

## Repo layout

- `splinewire/` — the pipeline: `detect` → `order` → `rectify` → `contact`, glued by `pipeline`, driven by `cli`. `synthetic` renders test photos; `testpart` renders a printable test chain.
- `data/chain.yaml` — the chain's physical parameters (pitch, half-width, ring size, pin count).
- `tests/` — pytest suite; end-to-end tests run on synthetic photos with known geometry.
- `experiments/` — standalone studies (e.g. how accurate pitch-only deskewing is).
- `docs/architecture.md` — design and rationale.
- `docs/next-steps.md` — status and near-term tasks, in risk order.
- `docs/open-questions.md` — unresolved design questions.

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests (~5 s)
uv run splinewire synth --shape pipe      # synthetic photo + truth -> out/synth/
uv run splinewire measure out/synth/pipe.jpg --truth out/synth/pipe-truth.json
uv run splinewire test-part               # printable SVG + truth -> out/test-part/
```

## Priority

Everything so far is validated only on synthetic photos. The riskiest open item is real-world accuracy: real phone photos (lighting, glare, lens distortion, EXIF focal accuracy) of a printed test part. See `docs/next-steps.md`, item 1.
