# Open questions

Unresolved design questions. Cross-reference `next-steps.md`; several get answered by experiments there.

## Accuracy in the real world

- Does iOS keep EXIF on web uploads? Some iOS versions strip it (and convert HEIC to JPEG) for photos uploaded from Safari. The app reports what arrived and falls back to a saved per-phone focal length; if EXIF never survives, consider measuring the focal length once from a test-plaque photo.
- Is phone EXIF focal length accurate enough? It is an integer 35 mm equivalent (±2% rounding at 26 mm), and some phones crop or digitally zoom. In simulation (noiseless, 12-link chain), a 2% focal error costs ≤0.05 mm worst-case at 25° tilt and ≤0.11 mm at 40°; 5% costs ≤0.12 / ≤0.27 mm. So small errors are tolerable, especially with near-straight-on photos. Real phones still need checking for gross errors (digital zoom, crops).
- Is lens distortion on phone main cameras corrected well enough in-camera to ignore? Simulated (`experiments/lens_distortion.py`): a residual k1 up to 0.01 costs ≤0.04 mm, and even 0.04 with the chain in a corner of the frame costs ≤0.14 mm. Solving for k1 in the deskew makes things worse (it trades against curvature on arcs), so it stays off. Still open: what real phones leave behind, and ultra-wide lenses (which iPhones switch to below ~20 cm).
- How does detection cope with real lighting, shadows, glare on glossy fiducials, and cluttered backgrounds? Is "photograph on plain paper" an acceptable constraint? Simulated (`docs/cv-robustness.md`): shadows, textured and printed tables, clutter, defocus, shake, noise and JPEG are now handled; plain paper is not needed. The one remaining failure is a lamp reflected in shiny filament, which washes the rings out: matte filament avoids it. To confirm on real photos.
- The projected center of a circle is not exactly the center of its image ellipse. Simulated: the bias is about f·(r/Z)²·sin(2·tilt)/2, ~0.07 px from 300 mm at 25°, and changes the measured shape by ~0.002 mm because it is nearly the same shift for every pin. Confirm on real photos.

## Chain hardware

- Pitch: 10 mm is a guess. Chord sag at the contact is modelled, but a smaller pitch follows tight radii better, while a larger pitch means fewer, bigger, easier-to-detect rings. What is the smallest radius the chain must follow?
- How consistent is the pitch across links, and does it drift with wear? Pitch error feeds straight into the deskew.
- How much joint friction, and does it drift with use?
- Ring construction on real links: windows in a thin black top layer over white (as on the test plaque, ≤0.4 mm deep), hollow rivet pins, or a label ring? Contrast and edge sharpness matter more than exact size; keep any relief shallow. Simulated for a 0.4 mm nozzle (`docs/cv-robustness.md` §5): the plain printed ring beats a dot, a bullseye, a checker-corner centre and ArUco tags; keep ≥1.5 mm bands, a ≥2 mm centre and ≥1.5 mm of black around it, print it in one piece with the pin hole, and use matte black filament.
- Length: is ~120 mm right, or should there be short and long chains? The software handles any pin count; set `n_pins` in the YAML.
- Physical joint limit: ordering assumes no joint bends more than 80°.

## Contact model

- The model assumes stadium-shaped links (semicircular ends centered on the pins). If the real links differ, the concave-contact rule needs updating.
- Near inflection points, and where curvature changes quickly relative to the pitch, contact points are approximate. How large is the error on real molding profiles?
- `--side inside|outside` must be right. Can the UI make this obvious? (For example, the preview could show which side the offset went.)

## Workflow and CAD

- Fusion: script vs add-in — which is lower friction to install and re-run?
- Does Fusion's Insert SVG keep 1:1 mm scale for our SVG, or is a JSON-reading script required?
- Origin and orientation in CAD: currently the first pin, with axes following the photo. Should the user be able to pick them, or align them to something?
- Phone → computer transport: local web upload vs cloud folder; it must preserve EXIF.

## Naming

- Tentative rename from `spline-wire` to `spline-link` (noted 2026-04-22). Decide before anything ships.
