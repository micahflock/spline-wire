# Next steps

Status doc: items are in risk order and get checked off as they are validated.

## Done (synthetic only)

- [x] Label-free ring fiducial on every pin; classical detection at 0.05–0.2 px on synthetic photos.
- [x] Geometric chain ordering with gap and stray handling.
- [x] Deskew from the pin pitch and EXIF focal length: ≤0.02 mm on clean synthetic photos, ~0.3 mm worst-case with 1 px center noise (`experiments/self_rectification.py`).
- [x] Contact offset from the pin line to the target curve (convex and concave), matching circular targets to 0.05 mm.
- [x] CLI: `measure`, `synth`, `test-part`; JSON/CSV/SVG output and a preview image.
- [x] Windows desktop app (`SplineWire.exe`, built by GitHub Actions) for processing photos with a GUI.

## 1. Real-photo accuracy with a printed test part  ← next

**Question:** do real phone photos keep pin-center error near 1 px or below?

- `uv run splinewire test-part` for each shape (`--shape s-curve|pipe|cove`). Print at 100% on paper and check the 50 mm scale bar with calipers. Glue it to something flat.
- Photograph each shape: 3 tilts (straight-on, ~20°, ~40°) × 2 lighting setups (daylight, indoor bulb) × 2 phones if available.
- Process them in `SplineWire.exe` with the matching truth file set under Options (any row over 1 mm is flagged), or with `uv run splinewire measure photo.jpg --truth out/test-part/<shape>-truth.json --out results/`.
- **Exit criterion:** worst pin error under **1 mm** on every photo, and under 0.5 mm on most.
- Things to watch for: EXIF focal accuracy (compare `focal_px` to an estimated-f run), lens distortion on wide lenses, glare on glossy prints, printer scale error (the scale bar check).

## 2. Chain hardware

**Only after item 1 passes.** Pitch accuracy is now the critical dimension.

- Build or adapt a chain matching `data/chain.yaml` (or update the YAML to match what's built), with rings on the pins. Options: a printed ring on each pin boss, or a contrasting hollow rivet as the pin, which is a ring for free.
- Tune joint friction.
- Measure the actual pitch with calipers across many links and put the mean in the YAML.
- **Exit criterion:** wrap a pipe or gauge of known radius, photograph, and recover the radius within 0.5 mm.

## 3. Points → Fusion

- Write a Fusion script/add-in that reads `spline-wire/points@1` JSON and adds a fitted spline through `curve_points` in the active sketch.
- Meanwhile, check whether Fusion's Insert SVG keeps the `-curve.svg` at 1:1 scale.
- **Exit criterion:** the curve appears in the active sketch, correctly scaled in mm, within 5 s.

## 4. Phone → computer

- Least-custom path first: a local web page the phone uploads to, or a watched cloud folder. Uploads must keep EXIF (some share paths strip it).
- **Exit criterion:** under 15 s from photo to points, no custom phone app.

## 5. End-to-end

- Real chain, real object (a pipe or doorknob profile), photo to curve in Fusion.
- **Exit criterion:** curve tracks the object within item 2's tolerance in under 30 s.

## Deferred

- Custom mobile app with live capture and preview.
- CAD packages beyond Fusion (the SVG output is a partial universal fallback).
- 3D / non-planar curves.
- Using the ring ellipse shapes as extra tilt information (not needed with EXIF so far).
