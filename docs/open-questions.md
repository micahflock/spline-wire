# Open questions

Unresolved design questions. Cross-reference `next-steps.md`; several get answered by experiments there.

## Accuracy in the real world

- Does iOS keep EXIF on web uploads? Some iOS versions strip it (and convert HEIC to JPEG) for photos uploaded from Safari. The app reports what arrived and falls back to a saved per-phone focal length; if EXIF never survives, consider measuring the focal length once from a test-plaque photo.
- Is phone EXIF focal length accurate enough? It is an integer 35 mm equivalent (±2% rounding at 26 mm), and some phones crop or digitally zoom. In simulation (noiseless, 12-link chain), a 2% focal error costs ≤0.05 mm worst-case at 25° tilt and ≤0.11 mm at 40°; 5% costs ≤0.12 / ≤0.27 mm. So small errors are tolerable, especially with near-straight-on photos. Real phones still need checking for gross errors (digital zoom, crops).
- Is lens distortion on phone main cameras corrected well enough in-camera to ignore? Simulated (`experiments/lens_distortion.py`): a residual k1 up to 0.01 costs ≤0.04 mm, and even 0.04 with the chain in a corner of the frame costs ≤0.14 mm. Solving for k1 in the deskew makes things worse (it trades against curvature on arcs), so it stays off. Still open: what real phones leave behind, and ultra-wide lenses (which iPhones switch to below ~20 cm).
- How does detection cope with real lighting, shadows, glare on glossy fiducials, and cluttered backgrounds? Is "photograph on plain paper" an acceptable constraint? Simulated (`docs/cv-robustness.md`): shadows, textured and printed tables, clutter, defocus, shake, noise and JPEG are now handled; plain paper is not needed. The one remaining failure is a lamp reflected in shiny filament, which washes the fiducials out: matte filament avoids it. To confirm on real photos.
- The projected center of a circle is not exactly the center of its image ellipse. Simulated: the bias is about f·(r/Z)²·sin(2·tilt)/2, ~0.07 px from 300 mm at 25°, and changes the measured shape by ~0.002 mm because it is nearly the same shift for every pin. Confirm on real photos.

## Chain hardware

- Pitch: 10 mm is a guess. Chord sag at the contact is modelled, but a smaller pitch follows tight radii better, while a larger pitch means fewer, bigger, easier-to-detect fiducials. What is the smallest radius the chain must follow?
- How consistent is the pitch across links, and does it drift with wear? Pitch error feeds straight into the deskew.
- Printed chain (`docs/printed-chain.md`): does 0.3 mm clearance print free on a typical printer, does the press-to-set seat every pin, and how much do friction and preload vary between joints and drift as PLA relaxes?
- O-ring chain (`docs/printed-chain.md`): does the countersink really out-hold the washer (no springback), does the self-cut thread centre the outer links well enough, and how evenly do the squeeze and friction come out across joints?
- How much joint friction, and does it drift with use?
- Fiducial construction on real links: a window in a thin black top layer over white (as on the test plaque, ≤0.4 mm deep), a light pin head, a label? Contrast and edge sharpness matter more than exact size; keep any relief shallow. Simulated for a 0.4 mm nozzle (`docs/cv-robustness.md` §5): a plain dot or ring beats a bullseye, a checker-corner centre and ArUco tags. The chain uses a 5 mm dot; keep ≥1.5 mm of black around it, make its centre the pin's (print it with the pin hole; a pin head sits wherever its clearance lets it), and use matte black filament.
- Length: is ~120 mm right, or should there be short and long chains? The software handles any pin count; set `n_pins` in the YAML.
- Physical joint limit: ordering assumes no joint bends more than 80°.

## Contact model

- The model assumes stadium-shaped links (semicircular ends centered on the pins). If the real links differ, the concave-contact rule needs updating.
- Near inflection points, and where curvature changes quickly relative to the pitch, contact points are approximate. How large is the error on real molding profiles?
- `--side inside|outside` must be right. Can the UI make this obvious? (For example, the preview could show which side the offset went.)

## Workflow and CAD

- Hand-placed pins: when no dot is visible to snap to (a lamp's reflection), a pin is its click fitted to the pitch of its links. Simulated, from a click up to 3 px off, the median error is 0.03 mm for one hidden pin and 0.1 mm for two in a row (worst of 12 trials: 0.23 and 0.19 mm), but how well do people click on real glare-damaged photos, and is "zoom in and click the middle of where the dot should be" enough guidance? Photos where every dot is washed out leave the plane fit with no measured links, so it falls back to fitting from the clicks.

- Fusion: script vs add-in — which is lower friction to install and re-run?
- Does Fusion's Insert SVG keep 1:1 mm scale for our SVG, or is a JSON-reading script required?
- Origin and orientation in CAD: currently the first pin, with axes following the photo. Should the user be able to pick them, or align them to something?
- Phone → computer transport: local web upload vs cloud folder; it must preserve EXIF.

## Naming

- Tentative rename from `spline-wire` to `spline-link` (noted 2026-04-22). Decide before anything ships.
