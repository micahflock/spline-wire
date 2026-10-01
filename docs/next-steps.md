# Next steps

Status doc: items are in risk order and get checked off as they are validated.

## Done (synthetic only)

- [x] Label-free fiducial on every pin (a dot; a ring also supported); classical detection at 0.05–0.2 px on synthetic photos.
- [x] Geometric chain ordering with gap and stray handling.
- [x] Deskew from the pin pitch and EXIF focal length: ≤0.02 mm on clean synthetic photos, ~0.3 mm worst-case with 1 px center noise (`experiments/self_rectification.py`).
- [x] Contact offset from the pin line to the target curve (convex and concave), matching circular targets to 0.05 mm.
- [x] CLI: `measure`, `synth`, `test-part`; JSON/CSV/SVG output and a preview image.
- [x] Windows app (`SplineWire.exe`, built by GitHub Actions): browser UI on the computer plus a phone upload page.
- [x] Pin editing in the app: remove a detection, add a missed pin or bring back a left-out one by clicking the photo; clicks snap to the dot under them (98% of simulated clicks, 0.1 px median error) and otherwise are fitted to the pitch (`edits.py`). Verified on simulated photos only; on real ones, see the open question about hand-placed pins.
- [x] Realistic photo simulator (`scene.py`: FDM print defects, tables, clutter, glare, shadows, defocus, shake, distortion, noise, JPEG) and a detection benchmark over 29 environments plus random ones. The detector was reworked against it: the whole chain is recovered in 155/167 simulated photos, up from 77/167, and the failures left are a lamp reflected in shiny filament. Fiducial designs compared for a 0.4 mm nozzle: a plain dot or ring; the chain uses the 5 mm dot (easiest to make, 86/92 vs the ring's 87/92 in the study). See `docs/cv-robustness.md`.

## 1. Real-photo accuracy with a printed test part  ← next

**Question:** do real phone photos keep pin-center error near 1 px or below?

- Print the test plaque: `uv run splinewire test-plaque --shape s-curve` (also `pipe`, `cove`), white then black PLA with one filament swap; see the generated `*-PRINTING.txt`. Check the 50 mm bar with calipers. (Paper alternative: `splinewire test-part`, printed at 100% and glued flat.)
- Photograph each shape: 3 tilts (straight-on, ~20°, ~40°) × 2 lighting setups (daylight, indoor bulb) × 2 phones if available.
- Send them from the phone page to `SplineWire.exe` with the matching truth file set in Settings (any photo over 1 mm is flagged), or with `uv run splinewire measure photo.jpg --truth out/test-plaque/<shape>-truth.json --out results/`. Look at both the plain and the scale-fit error: a gap between them is print scale, not measurement.
- **Exit criterion:** worst pin error under **1 mm** on every photo, and under 0.5 mm on most.
- **Results so far** (SplineWire 0.5.1, iPhone, uploaded from the phone page):

  | Date | Shape | Tilt | Lighting, background | Focal | Pins | Worst pin | RMS | After scale fit | Size vs design |
  |---|---|---|---|---|---|---|---|---|---|
  | 2026-10-01 | s-curve | 24.9° | indoor, dotted mousepad with keyboard and clutter (329 look-alikes ignored) | 24 mm set in Settings (none in file) | 13/13 | 0.098 mm | 0.063 mm | 0.099 mm | −0.12 % |

  The first photo passes by about 10×. At ~22 px/mm that rms is 1–1.5 px, close to the plaque's own print tolerance, so this plaque can't show much better than this. Estimating the focal length from the chain instead gave 26.2 mm and a 26.6° tilt; that run has not been scored against the truth file yet. Still to do: straight-on and ~40°, a second lighting setup, the `pipe` and `cove` plaques, and a second phone.
- Things to watch for: EXIF focal accuracy (compare `focal_px` to an estimated-f run), lens distortion on wide lenses, glare on glossy prints, printer scale error (the scale bar check).
- Print the plaque in matte black if possible, and include some deliberately hard photos (a lamp reflected in the part, a shadow across it, a busy table): the simulator predicts matte survives the reflection and standard PLA may not. Where a real photo fails or measures worse than simulated, reproduce the condition in `scene.py` and add it to the benchmark.

## 2. Chain hardware

**Only after item 1 passes.** Pitch accuracy is now the critical dimension.

- Build or adapt a chain matching `data/chain.yaml` (or update the YAML to match what's built), with a light dot on each pin (keep ≥1.5 mm of dark link around it). Options: a printed dot on each pin boss, or a light pin head; a pin head's centre is only as good as its fit in the hole. A contrasting hollow rivet is a ring for free (`fiducial: ring`).
- Tune joint friction.
- Measure the actual pitch with calipers across many links and put the mean in the YAML.
- **Exit criterion:** wrap a pipe or gauge of known radius, photograph, and recover the radius within 0.5 mm.

## 3. Points → Fusion

**Question:** can measured points get into a Fusion sketch with minimal friction?

- Feasibility study and test protocol: `docs/fusion-import.md`. Fusion can't paste coordinates natively, so the low-friction route is a small add-in.
- [x] Built: **Copy points** / **Install Fusion add-in…** in SplineWire.exe, the `fusion/SplineWire` add-in (**Paste points** into the sketch being edited, with a spline through the points), plus DXF (mm) and ImportSplineCSV (cm) fallbacks.
- [ ] Run the test protocol in Fusion. Nothing has been run inside Fusion yet.
- **Exit criterion:** points appear in the active sketch, correctly scaled in millimeters, within **5 seconds** of clicking Copy points.

## 4. Phone → computer

**Decision (2026-09):** direct upload from the phone's browser to the app over the local network (same Wi-Fi, or the computer on the phone's hotspot). No phone app, no cloud, nothing watching the photo library.

- [x] Built: phone page (Take photo / Choose from library) reached through a QR code on the desktop page; uploads saved byte-for-byte; results appear on the computer within seconds.
- [ ] Test with the iPhone:
  - Does the focal length survive the upload? Try both buttons. The phone page and the desktop details show "Focal length in file". If it's missing, set the iPhone focal length in Settings (24 mm for iPhone 14 Pro and iPhone 15 or later, 26 mm for earlier models).
    - First try (2026-10-01): no. The photo arrived as a 4032 × 3024 JPEG with all EXIF gone ("Camera: not recorded", "Focal length in file: missing"). Which button was used is not recorded; try the other one.
  - Time from shutter to the photo appearing on the computer.
  - Home Wi-Fi and the phone's hotspot. First try (0.5.0) timed out: Windows Firewall. 0.5.1 detects it and adds **Allow phone connections**; confirm it fixes the timeout.
- **Exit criterion:** under 15 s from photo to points, no custom phone app.

## 5. End-to-end

- Real chain, real object (a pipe or doorknob profile), photo to curve in Fusion.
- **Exit criterion:** curve tracks the object within item 2's tolerance in under 30 s.

## Deferred

- Custom mobile app with live capture and preview.
- CAD packages beyond Fusion (the SVG output is a partial universal fallback).
- 3D / non-planar curves.
- Using the fiducials' ellipse shapes as extra tilt information (not needed with EXIF so far).
