# Open items

[← Back to the README](../README.md)

What should be done, in risk order. This list and [future-work.md](future-work.md) (what might be done) replace the earlier `next-steps.md` and `open-questions.md` (consolidated 2026-11-18). Everything here is about the snap-fit chain (`data/chain-snap.yaml`, [snap-chain.md](snap-chain.md)) and the website ([web-build.md](web-build.md)); the O-ring and print-in-place chains are not tracked.

## 1. Real-world accuracy: plan a revised test series  ← next

**Where it stands.** The owner's tests on the live site (2026-11, the TypeScript engine, phone photos) show that real-photo accuracy is good; the figures are to be recorded with the new series. The one scored real photo is older: a printed test plaque, 0.098 mm worst pin and 0.063 mm rms at a 25° tilt (2026-10-01, the retired Windows app). The old test matrix was written for the Windows app and the O-ring dot chain and has been dropped. A new series is to be planned for the snap-fit chain and the website. It should answer:

- **Accuracy on an object of known size.** No real measurement has yet been checked against an object of known size. Targets carried over: worst pin under 1 mm on every photo and under 0.5 mm on most; a known radius recovered within 0.5 mm.
- **Focal length.** How accurate the EXIF focal length is on real phones (the 35 mm tag is a whole number, so ±2% at 24 mm; digital zoom and crops), how the estimate from the chain compares, and the Firefox path, which measures a video frame with a remembered focal length. In simulation a 2% focal error costs ≤0.05 mm at 25° tilt and ≤0.11 mm at 40°; 5% costs ≤0.12 / ≤0.27 mm.
- **Lens distortion.** What real phones leave after their own correction. Simulated, a residual k1 up to 0.01 costs ≤0.04 mm (`experiments/lens_distortion.py`). The page never opens an ultra-wide camera (O10 in web-build.md).
- **Detection on real surfaces with the snap-fit rim.** Glare on the black top surfaces, the print quality of the two-perimeter white rim, white paper (nothing dark lies outside the rims there but the arms), matte against glossy filament.
- **Tilt and bend limits.** `max_bend_deg: 110` stays until real photos say otherwise; in simulation, chains at 35–40° tilt with a joint at 100–102° can read as cut (with a warning).
- **The circle-centre bias.** The projected centre of a circle is not the centre of its image ellipse; simulated, it moves the measured shape by ~0.002 mm. Confirm it stays negligible.

Every figure the series produces goes in this file with the photo's conditions (CLAUDE.md: every accuracy claim has a number behind it). Where a real photo fails or measures worse than simulated, reproduce the condition in `scene.py` and add it to the benchmark.

## 2. Snap-fit chain hardware

- **Print the redesigned link** (3DP_Link_Ver_MF, with the white rim). The software follows it since 2026-11-18, validated on simulated photos only; an earlier version of the link was printed in one colour and holds its shape.
- **Custom end links.** The owner is modelling end links so that both ends behave like the rest of the chain. Today the first pin touches the target only at its head's top edge, 12.65 mm above the table (the bare pin, 2–3 mm, lower down), and the last link's sleeve has no rim, so its position is not measured. When the end links exist: read their numbers from the STEP file (`experiments/snap_link_geometry.py`), update `data/chain-snap.yaml`, `synthetic.render_snap_photo` and the contact rule for end pins (`contact.py` and its TypeScript port, with new golden files), and drop the "keep the target off both ends" advice in snap-chain.md.
- **Bend limit.** The joint turns +102° / −127° and has no stop. Past about 105° a bend reads as a cut chain or with the bend's pin left out (with a warning). Add a stop to the link, or keep bends under about 105°.
- **Pitch and half width.** Measure the pitch over several links and a sleeve across with calipers, and put the means in `data/chain-snap.yaml`. How consistent is the pitch from link to link, and does it drift with wear? Pitch error feeds straight into the deskew.
- **Joint friction.** Does the snap clip hold a pose firmly enough, evenly across joints, and does its grip relax with use (PLA creep)?
- **Length and pitch.** Is the current pin count the right length, or should there be short and long chains? What is the smallest radius the chain must follow at a 13 mm pitch? Taking the perspective out needs at least 6 pins (7 without a focal length).

## 3. Contact model (snap-fit)

The snap-fit chain touches the target with the round sleeves and heads about every pin (`contact: round`): the curve point is the pin offset by 4.39 mm along the bisector of its two links, on convex and concave bends alike. The stadium-link rules of the other chains do not apply.

- **Quantify the model's error, in simulation first.** The target touches the chain at discrete round contacts 13 mm apart, and the curve between them is a spline. How far is that from the true curve on tight radii (down to the sleeve radius), at inflection points and where curvature changes quickly against the pitch? Then on real moldings and pipes, as part of item 1.
- **Ends.** Model the custom end links (item 2) so the end points are measured like the rest, rather than continuing the turn of the joint next to them.
- **Missed pins.** Where a pin is missed the curve bridges the gap (0.16 mm off on a 30 mm radius for the O-ring chain; not yet measured for this one). Should the missing pin be placed from its two neighbours, as a pin added by hand already is, or is "add it on the photo" enough?
- **Object side.** The phone page draws the contact edge on the photo and switches it with a tap. Does that make mistakes rare in practice?

## 4. Website: real-phone checks

From [web-build.md](web-build.md) (Status, Open decisions, Work queue).

- **Phone page checklist** (step 2's exit criterion; the owner's checklist is in its Log, 2026-11-17).
- **Camera view checklist** (step 5; Log, 2026-11-17), including which rear camera opens (**O10**: never an ultra-wide, one lens throughout).
- **Pairing and relay** with a real phone and Fusion (step 3's exit criterion).
- **Fusion add-in install path (O5).** Try the desktop page's three steps (download the zip, add it in Scripts and Add-Ins, Run on Startup) in a current Fusion; the menu names are unchecked.
- **One DXF imported in Fusion** (D19's simpler DXF).
- **The coach.** Recalibrate `web/src/overlay/coach.ts` with recorded phone video (its limits come from simulated renders).
- **A staging path** for deploying branches before `main`.

## 5. End to end

- Real chain, real object (a pipe or a molding profile), from the phone to a curve in a Fusion sketch, timed.
- A first run was made on 2026-10-19 with the O-ring chain and the retired Windows app (photo → Copy points → Paste points worked); it was not timed or checked against the object. None yet with the snap-fit chain and the website.
- **Exit criterion:** the curve tracks the object within item 1's targets, under 15 s from photo to points on the desktop and under 30 s from photo to the curve in the sketch.

## 6. Workflow and CAD

- **Hand-placed pins.** When no dot is visible to snap to (a lamp's reflection), a pin is its tap fitted to the pitch of its links. Simulated, from a tap up to 3 px off, the median error is 0.03 mm for one hidden pin and 0.1 mm for two in a row (worst of 12 trials: 0.23 and 0.19 mm). How well do people place them on real glare-damaged photos, and is "zoom in and tap the middle of where the rim should be" enough guidance?
- **Fit points in Fusion.** Does `sketchFittedSplines.add` take the pasted sketch points as its fit points, or does the add-in fall back to plain coordinates and leave a second point at each spot ([fusion-import.md](fusion-import.md))? Paste points works either way.
