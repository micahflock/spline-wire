# Architecture

The system is a pipeline: physical curve → posed chain → phone photo → pin positions → curve points → CAD. Stages hand off plain data, so each can be developed and tested alone.

## 1. Chain (hardware)

- Rigid links of fixed **pitch** `p` (pin-to-pin distance; default 10 mm), about 12 links (~120 mm).
- Links are bars of **half-width** `w` (pin line to contact edge; default 4 mm) with semicircular ends centered on the pins. This shape is what the contact model in stage 5 assumes.
- Joint friction holds the pose under gravity and handling but allows posing by hand.
- Planar: no out-of-plane twist.
- **One fiducial centered on every pin**, on one face only: a light 5 mm dot on the dark link (`fiducial: dot` in `data/chain.yaml`), chosen because it is the simplest to make; a ring (`fiducial: ring`) is also supported and slightly more robust (`docs/cv-robustness.md` §5). There are no labels or IDs, and no separate skew/end marker.

The pitch is the only dimension the software relies on to deskew the photo, so it must be accurate and consistent. The fiducial size only needs to be roughly right: it predicts the pitch in the image and helps reject look-alikes.

Fiducials go on the pins because pin-to-pin distance is fixed. The distance between link *centers* shrinks as a joint bends (`p·cos(θ/2)`), which would break the deskew constraint.

Fiducials go on one face only because a chain photographed from the other side is a mirror image, and nothing in an unlabeled dot or ring pattern can reveal that.

## 2. Photo

- Lay the posed chain flat, fiducial side up, on a plain contrasting surface.
- Take one photo with the phone's normal camera app, with the whole chain in view. A moderate tilt is fine: it is corrected.
- The photo's EXIF 35 mm-equivalent focal length is used (see stage 4). Screenshots, crops or messaging apps that strip EXIF lose it; `--focal-35mm` overrides.

## 3. Detection and ordering — `detect.py`, `order.py`

Detection is stress-tested on simulated phone photos of printed chains under 29 environments plus random ones; see `docs/cv-robustness.md` for the results and what each step below fixed.

`detect_fiducials` dispatches on the chain's fiducial. Rings are described first; dots reuse the same pyramid, thresholds and edge fitting.

- **Ring candidates:** on an image pyramid (full, ½, ¼, ⅛ size), so every ring is also seen where it is a few dozen pixels across. There, area downsampling erases the extrusion lines of a 3D print, which catch a lamp as stripes 0.4 mm apart and can bridge a ring to its surroundings. Each level is binarized with 25 px windows, about one to two rings wide, so the threshold stays on the link instead of being dragged by the table beside it, a shadow edge or a glare hotspot: above/below the local mean (both polarities), and "well above" (mean + 0.8 local standard deviations, where there is contrast), which covers a black chain on a black table and a link that sheen has made lighter than the table. Connected components that could be a ring by size and fill are traced; a candidate has exactly one hole (noise specks under 2% of the ring are ignored) and roughly concentric elliptical edges, with the inner/outer ratio roughly the spec's (the accepted range is lopsided upward because thresholding, defocus, undersized printed windows and recess walls all thin the band).
- **Ring refinement:** on the full-resolution gray image, 32–180 rays through each candidate. On each ray the inner and outer edges are where the profile crosses halfway between that ray's own dark and bright levels, so a shadow or glare gradient across a ring does not shift them. Ellipses are fitted to the sub-pixel edge points with outlier rejection (seam blobs, dust). The center is the mean of the two ellipse centers weighted by fit quality and area: a print defect pulls a small circle's fitted center further than a large one's. On simulated prints the error is ~0.1 px median.
- **Dots** have no hole to tell them from other bright blobs, so each candidate (outer outline only: a dot wider than the threshold window comes out hollow) must show a dark margin all round, as a dot on a link does: along every ray, the margin 1.15–1.45 radii out must be a steady fraction of the dot's brightness (a ratio, so a shadow edge darkens both alike). The edge is fitted as for a ring. Candidates of different sizes on the same centre are all refined: a halo of table around a dark patch must not stand in for the dot.
- **Order:** the rings carry no IDs, so order comes from geometry. Walk from ring to ring stepping about one pitch, preferring the smallest turn, with joints limited to 80°. A ~2-pitch step counts as one missing ring (a *gap*). Each ring's ellipse predicts the step length (pitch = 2 ring diameters, foreshortened at most by the ellipse's own axis ratio), and neighbours must be within 35% of each other's size and have about the same ellipse shape (all pins lie on one plane), which keeps printed letters, washers and speckle off the chain. Light-on-dark and dark-on-light fiducials are never mixed (letters are dark on light). Every ring is tried as the start and the longest walk wins. Rings off the walk are rejected as strays.
- **Misfits:** after deskewing, a pin whose ring is more than 12% bigger or smaller (in mm) than its neighbours, or an end pin whose link is far from one pitch, is dropped and the chain re-solved: a washer that happened to sit one pitch past the end.
- Direction along the chain is arbitrary and doesn't matter for the curve.

Known limits: joints bending more than 80°, or two parts of the chain lying within about half a pitch of each other, can confuse ordering. A lamp reflected in shiny filament can wash out the fiducials entirely (matte filament avoids it; see `docs/cv-robustness.md`). The preview image shows the chosen order for checking.

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

Error grows roughly linearly with pin-center noise. Classical dot or ring detection (~0.2 px) is far inside the 1 mm target; LLM-style estimates (several px) are not. That is why an LLM does not do localization.

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

## App — `webapp/` (SplineWire.exe)

The app is a small local web server (standard library `http.server`) with its UI in the browser, so display scaling, layout and fonts are the browser's job. `SplineWire.exe` starts it in a console window (close the window to quit), opens `http://localhost:8765/`, and reuses an already-running copy if there is one.

- **Phone → computer:** the desktop page shows a QR code for `http://<this computer's LAN IP>:8765/phone`. The phone page has **Take photo** and **Choose from library**; uploads are sent as the raw file body and saved byte-for-byte, then measured by a worker thread. The phone and computer must share a network (a phone hotspot works). The desktop page follows new uploads automatically.
- **Camera metadata:** some iOS versions strip EXIF from web uploads. Every photo shows what arrived (format, size, camera, focal length). Focal length is taken from: an override, else EXIF, else the saved "iPhone focal length" setting, else estimated from the chain.
- **Desktop page:** photo list with thumbnails and status; for the selected photo, the measured curve (SVG in mm), the photo cropped to the chain with detections, **Copy points** (Ctrl+C also works), downloads (DXF, CSV, SVG, JSON, Fusion CSV) and diagnostics. Settings (chain, object side, focal lengths, test-part truth file) persist in `%APPDATA%\SplineWire\settings.json`. Photos and results go to `Documents\SplineWire`.
- **Security:** a random token (in the QR code, stored with the settings) is needed for every API call, as an `X-Token` header, which cross-site pages can't send without a CORS preflight. Images and downloads take it as `?t=`. The Host header must name this computer (against DNS rebinding). Settings, downloads, installing the Fusion add-in, opening folders and quitting only work from this computer.
- **Windows Firewall:** it silently drops the phone's connection (the phone just times out) until the program is allowed, and answering Cancel at its first-run prompt adds block rules, which beat any allow rule. `webapp/firewall.py` reads the firewall with PowerShell in the background; when phones would be blocked the desktop page shows **Allow phone connections**, which (after one admin prompt) removes this program's inbound block rules and adds one allow rule for TCP 8765–8774 on every network type. The rule isn't tied to the exe's path, so it survives downloading a newer build. The page also says when a phone has reached the app, and lists the other causes (guest Wi-Fi or client isolation, VPN, third-party firewalls).

## Testing without hardware

- `splinewire synth` renders a photo of a posed chain through a simulated tilted phone camera (with EXIF), plus a truth file. `--env PRESET` renders a realistic photo instead (`scene.py`): a printed plaque with FDM defects for a 0.4 mm nozzle, on one of several tables, under lamp light with glare and shadows, through a phone camera with defocus, shake, residual lens distortion, noise, sharpening and JPEG. `experiments/cv_benchmark.py` runs the pipeline over all presets and random environments; `experiments/fiducial_study.py` compares fiducial designs the same way (`docs/cv-robustness.md`).
- `splinewire test-part` writes a paper-printable SVG of a chain with exactly known pins, plus a truth file.
- `splinewire test-plaque` writes a 3D-printable version (STLs plus printing notes and truth): a 1.6 mm white base trimmed to the chain's outline plus a 1.5 mm rim (the caliper bar hangs off it on a short bridge), one manual filament swap, then a 0.4 mm black chain layer with a window over each pin (a dot, or a ring, as `chain.yaml` says) and a 50 mm caliper bar. About 3 cm³ of plastic; the earlier full rectangular plate took 14–25 cm³. The table shows around the part, so detection is tested on dark, grey and white tables. White goes underneath because white PLA is translucent; black is opaque in two layers, which keeps the window walls shallow.
- Truth comparisons report the error after a rigid fit and after a scale fit. The scale fit removes the test part's own print or paper scale error, and the fitted scale shows how far off-size the part (or the pitch setting) is.
- `experiments/relief_bias.py` ray-casts tilted photos of the plaque with its real window walls. At 0.4 mm depth all pins are found up to 40° tilt and the error matches a flat print (≈0.01–0.02 mm), because the walls shift every ring about the same way. At 0.8 mm, rings start being lost at 40°.
