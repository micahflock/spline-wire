# Architecture

The system is a pipeline: physical curve → posed chain → phone photo → pin positions → curve points → CAD. Stages hand off plain data, so each can be developed and tested alone.

## 1. Chain (hardware)

- Rigid links of fixed **pitch** `p` (pin-to-pin distance; default 10 mm), about 12 links (~120 mm).
- Links are bars of **half-width** `w` (pin line to contact edge; default 4 mm) with semicircular ends centered on the pins. This shape is what the contact model in stage 5 assumes.
- Joint friction holds the pose under gravity and handling but allows posing by hand.
- Planar: no out-of-plane twist.
- **One ring fiducial centered on every pin**, on one face only. There are no labels or IDs, and no separate skew/end marker.

The pitch is the only dimension the software relies on to deskew the photo, so it must be accurate and consistent. Ring size only needs to be roughly right: it helps reject non-ring shapes.

Fiducials go on the pins because pin-to-pin distance is fixed. The distance between link *centers* shrinks as a joint bends (`p·cos(θ/2)`), which would break the deskew constraint.

Fiducials go on one face only because a chain photographed from the other side is a mirror image, and nothing in an unlabeled ring pattern can reveal that.

## 2. Photo

- Lay the posed chain flat, fiducial side up, on a plain contrasting surface.
- Take one photo with the phone's normal camera app, with the whole chain in view. A moderate tilt is fine: it is corrected.
- The photo's EXIF 35 mm-equivalent focal length is used (see stage 4). Screenshots, crops or messaging apps that strip EXIF lose it; `--focal-35mm` overrides.

## 3. Detection and ordering — `detect.py`, `order.py`

- **Rings:** local adaptive threshold, then contours with exactly one hole (specks of noise smaller than 2% of the ring are ignored) whose outer and inner edges both fit concentric ellipses. The inner/outer diameter ratio must be roughly the spec's; the accepted range is lopsided upward because thresholding, defocus, undersized printed windows and recess walls all thin the ring band. Both polarities are tried. The center is the mean of the two ellipse centers. On synthetic photos the error is 0.05–0.2 px.
- **Order:** the rings carry no IDs, so order comes from geometry. Walk from ring to ring stepping about one pitch, preferring the smallest turn, with joints limited to 80°. A ~2-pitch step counts as one missing ring (a *gap*). Every ring is tried as the start and the longest walk wins. Rings off the walk are rejected as strays.
- Direction along the chain is arbitrary and doesn't matter for the curve.

Known limits: joints bending more than 80°, or two parts of the chain lying within about half a pitch of each other, can confuse ordering. The preview image shows the chosen order for checking.

## 4. Deskew — `rectify.py`

All pins lie on one plane. With a pinhole camera of known focal length `f`, each detected pin defines a ray, and a plane `{X : m·X = 1}` places pin `i` at `X_i = ray_i / (m·ray_i)`. The three unknowns in `m` (plane tilt and distance) are solved by least squares so that every consecutive pair of pins is exactly `p` apart. Multiple starting tilts avoid local minima.

- With 12 links there are 12 equations for 3 unknowns, so the problem is well over-determined.
- Without `f` it becomes a 4th unknown. That works for strongly curved chains but is weak for nearly straight ones, and the CLI warns.
- A straight chain's tilt is unobservable, but that doesn't affect the result: a straight chain reconstructs straight either way.
- Output axes follow the photo (+x right, +y up), with the origin at the first pin.

Accuracy from `experiments/self_rectification.py` (worst pin error, mm, 12-link chain, 25° tilt, ~8 px/mm):

| pin-center noise | assume top-down | deskew, known f | deskew, estimated f |
|---|---|---|---|
| 0.5 px | 2.5–6.1 | 0.14–0.20 | 0.17–0.55 |
| 1 px | 2.3–5.7 | 0.28–0.38 | 0.32–0.90 |
| 3 px | 2.3–5.9 | 0.84–1.03 | 1.06–1.35 |

Error grows roughly linearly with pin-center noise. Classical ring detection (~0.2 px) is far inside the 1 mm target; LLM-style estimates (several px) are not. That is why an LLM does not do localization.

## 5. Contact offset and curve — `contact.py`

The target touches the chain's edge, not its pin line:

- **Chain bent around the object** (convex target): each link's straight edge is tangent to the target near its middle, and the pins stand off by `w` plus the chord sag `p²/8R`. Contact point: link midpoint offset by `w`.
- **Chain bent away from the object** (concave target): straight edges bridge across and the rounded link ends touch. Contact point: pin offset by `w` along the joint bisector.

Both are exact for circular arcs and a close approximation for slowly varying curvature. Which side the object was on is `--side inside` (default: the object is on the side the chain curls toward) or `--side outside`.

The curve is a cubic spline through the contact points (chord-length parameter, not-a-knot ends).

## 6. Output and CAD — `output.py`

`splinewire measure PHOTO` writes:

- `PHOTO.json` — schema `spline-wire/points@1`:
  ```json
  {
    "schema": "spline-wire/points@1",
    "units": "mm",
    "curve_points": [[x, y], ...],
    "pin_points": [[x, y], ...],
    "object_side": "inside",
    "diagnostics": {"pins_found": 13, "missing_pins": 0, "rejected_detections": 0,
                    "focal_px": 3005.0, "focal_estimated": false, "tilt_deg": 25.0,
                    "link_residual_rms_mm": 0.002, "link_residual_max_mm": 0.004},
    "warnings": []
  }
  ```
- `PHOTO-curve.csv` — the curve points.
- `PHOTO-curve.svg` — 1:1 drawing in mm (fitted curve, pins, curve points) for SVG import.
- `PHOTO-preview.jpg` — the photo with detections and chain order drawn on, for checking.

- `PHOTO-curve.dxf` — the curve as a spline plus its points, in mm (`$INSUNITS`), for Insert > Insert DXF.
- `PHOTO-fusion-cm.csv` — headerless `x,y,z` in cm for Fusion's built-in ImportSplineCSV script.

Into Fusion, the main path is the `fusion/SplineWire` add-in: Copy points in the app, then Paste points in a sketch. See `docs/fusion-import.md` for the options compared and the test protocol.

## Desktop app — `gui.py`

A Tkinter window for Windows (packaged as a single `SplineWire.exe`) that wraps the same `process_photo` as the CLI:

- Add photos (JPEG, PNG, HEIC, …) or drop them onto the .exe. Chain settings, object side, focal-length override, an optional truth file and the output location are set in the left panel and remembered between runs (`%APPDATA%\SplineWire\settings.json`).
- "Process all" runs in a background thread. Each photo's row shows pins found and tilt, or max error against a truth file, with warnings flagged.
- Tabs: **Detections** (the preview cropped to the chain), **Curve** (pins, curve points and fitted spline in mm), **Details** (diagnostics, warnings, saved files).
- Output files are the same as the CLI's, by default in a `splinewire-out` folder next to each photo.
- Unexpected errors are shown and appended to `%APPDATA%\SplineWire\errors.log`.

## Testing without hardware

- `splinewire synth` renders a photo of a posed chain through a simulated tilted phone camera (with EXIF), plus a truth file.
- `splinewire test-part` writes a paper-printable SVG of a chain with exactly known pins, plus a truth file.
- `splinewire test-plaque` writes a 3D-printable version (STLs plus printing notes and truth): a white 2.4 mm plate, one manual filament swap, then a 0.4 mm black chain layer with a ring-shaped window over each pin and a 50 mm caliper bar. White goes underneath because white PLA is translucent; black is opaque in two layers, which keeps the window walls shallow.
- Truth comparisons report the error after a rigid fit and after a scale fit. The scale fit removes the test part's own print or paper scale error, and the fitted scale shows how far off-size the part (or the pitch setting) is.
- `experiments/relief_bias.py` ray-casts tilted photos of the plaque with its real window walls. At 0.4 mm depth all pins are found up to 40° tilt and the error matches a flat print (≈0.01–0.02 mm), because the walls shift every ring about the same way. At 0.8 mm, rings start being lost at 40°.
