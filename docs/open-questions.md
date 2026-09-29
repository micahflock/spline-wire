# Open questions

Unresolved design questions. Cross-reference `next-steps.md`; several get answered by experiments there.

## Accuracy in the real world

- Does iOS keep EXIF on web uploads? Some iOS versions strip it (and convert HEIC to JPEG) for photos uploaded from Safari. The app reports what arrived and falls back to a saved per-phone focal length; if EXIF never survives, consider measuring the focal length once from a test-plaque photo.
- Is phone EXIF focal length accurate enough? It is an integer 35 mm equivalent (±2% rounding at 26 mm), and some phones crop or digitally zoom. In simulation (noiseless, 12-link chain), a 2% focal error costs ≤0.05 mm worst-case at 25° tilt and ≤0.11 mm at 40°; 5% costs ≤0.12 / ≤0.27 mm. So small errors are tolerable, especially with near-straight-on photos. Real phones still need checking for gross errors (digital zoom, crops).
- Is lens distortion on phone main cameras corrected well enough in-camera to ignore? Ultra-wide lenses probably are not.
- How does detection cope with real lighting, shadows, glare on glossy fiducials, and cluttered backgrounds? Is "photograph on plain paper" an acceptable constraint? (Shallow relief on its own is fine: see `experiments/relief_bias.py`.)
- The projected center of a circle is not exactly the center of its image ellipse. The bias is estimated below 0.1 px at these ring sizes and ignored; confirm on real photos.

## Chain hardware

- Pitch: 10 mm is a guess. Chord sag at the contact is modelled, but a smaller pitch follows tight radii better, while a larger pitch means fewer, bigger, easier-to-detect rings. What is the smallest radius the chain must follow?
- How consistent is the pitch across links, and does it drift with wear? Pitch error feeds straight into the deskew.
- How much joint friction, and does it drift with use?
- Ring construction on real links: windows in a thin black top layer over white (as on the test plaque, ≤0.4 mm deep), hollow rivet pins, or a label ring? Contrast and edge sharpness matter more than exact size; keep any relief shallow.
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
