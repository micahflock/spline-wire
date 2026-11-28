# Open items

[← Back to the README](../README.md)

What should be done, in risk order. What might be done later is in [future-work.md](future-work.md); the website's plan, decisions, ready-to-build work queue and session log are in [web-build.md](web-build.md). Everything here is about the snap-fit chain (`data/chain-snap.yaml`, [snap-chain.md](snap-chain.md)) and the website; the O-ring and print-in-place chains are not tracked. (This list and future-work.md replaced `next-steps.md` and `open-questions.md` on 2026-11-18.)

**Where things stand (2026-11-28).** The riskiest item, real-world accuracy (item 1), can resume: the owner has an updated chain that prints with an acceptable twist and bow, and an end pin (item 2); their files are to be uploaded. Meanwhile the next software work is item 4a (the colour channel, then chain cards; web-build.md work queue items 13 and 14), and the real-phone checks (item 4) are the owner's whenever a phone is at hand.

## 1. Real-world accuracy with the snap-fit chain

**Status 2026-11-28: to resume with the updated chain** (item 2), paused since 2026-11-19 on the old print's twist. The test series is planned in [real-world-tests.md](real-world-tests.md) and its tools are built (`splinewire fixed-chain`, `splinewire test-gauges`, the test page at `spikes/accuracy/`, `splinewire score`). Phase 0 (calipers) and phase 1 (plaques) are done; the gauge tests (phase 2) are to redo once the chain is fixed. What stands and what to redo: real-world-tests.md, "Where it stands".

**What is known.** The owner's tests on the live site (2026-11, the TypeScript engine, phone photos) show that real-photo accuracy is good; the figures are to be recorded with the series. Phase 1's rigid plaques (camera and detection, no chain): worst pin 0.12–0.24 mm centred, up to 37° tilt, all 13 pins. The one older scored photo: a printed test plaque, 0.098 mm worst pin and 0.063 mm rms at a 25° tilt (2026-10-01, the retired Windows app). The series should answer:

- **Accuracy on an object of known size.** No real measurement has yet been checked against an object of known size. Targets carried over: worst pin under 1 mm on every photo and under 0.5 mm on most; a known radius recovered within 0.5 mm.
- **Focal length.** How accurate the EXIF focal length is on real phones (the 35 mm tag is a whole number, so ±2% at 24 mm; digital zoom and crops), how the estimate from the chain compares, and the Firefox path, which measures a video frame with a remembered focal length. In simulation a 2% focal error costs ≤0.05 mm at 25° tilt and ≤0.11 mm at 40°; 5% costs ≤0.12 / ≤0.27 mm. **Found (2026-11-19, iPhone 16 Pro, the page's camera): the EXIF's 24 mm is about 5% short; the plaque fits best at a median 25.2 mm (24.5–26.0), and with it the steep photos' 0.66–0.90 mm drop to 0.12–0.19 mm** ([real-world-tests.md](real-world-tests.md), phase 1). To confirm on full-size photos, then correct: a per-lens focal calibration from a plaque photo, or the phone page's focal override (Settings in debug mode, `?debug`). **Session 2 (2026-11-19): on 16 centred plaque photos the best fit is 24.2–26.7 mm, median 25.0; not explained by placement; a fixed ~25.0 mm override is a working value, not a calibration (provisional: half-size uploads, one phone, plaque flatness unchecked). Simulated, at ≤20° tilt a ±5% focal error moves a concave R100 radius by ≤0.19 mm: shoot near straight on.**
- **Lens distortion.** What real phones leave after their own correction. Simulated, a residual k1 up to 0.01 costs ≤0.04 mm (`experiments/lens_distortion.py`). The page never opens an ultra-wide camera (O10 in web-build.md). **Found (2026-11-19): near the edge of the frame the S plaque measures 0.33–0.64 mm worst even with its best-fit focal length, against 0.08–0.20 mm centred: probably distortion left in takePhoto stills. Keep the chain near the middle until it is measured** (fit a k1 to plaque photos taken in the corners).
- **Detection on real surfaces with the snap-fit rim.** Glare on the black top surfaces, the print quality of the two-perimeter white rim, white paper (nothing dark lies outside the rims there but the arms), matte against glossy filament.
- **Tilt and bend limits.** `max_bend_deg: 110` stays until real photos say otherwise; in simulation, chains at 35–40° tilt with a joint at 100–102° can read as cut (with a warning).
- **The circle-centre bias.** The projected centre of a circle is not the centre of its image ellipse; simulated, it moves the measured shape by ~0.002 mm. Confirm it stays negligible.

Every figure the series produces goes in this file with the photo's conditions (CLAUDE.md: every accuracy claim has a number behind it). Where a real photo fails or measures worse than simulated, reproduce the condition in `scene.py` and add it to the benchmark.

## 2. Snap-fit chain hardware

- **The updated chain** (2026-11-28). The owner has an updated version of the chain that prints with an acceptable twist and bow; the earlier print twisted and did not sit fully against the concave gauges' walls (real-world-tests.md, session 2). Its files are to be uploaded; until then the software follows 3DP_Link_Ver_MF (since 2026-11-18, validated on simulated photos only). When they are: read its numbers from the STEP file (`experiments/snap_link_geometry.py`) and update what differs in `data/chain-snap.yaml`, `synthetic.render_snap_photo` and snap-chain.md (a changed preset that a chain card names takes a new name, D30), then measure its pitch again: phase 0's 12.98 mm was measured on the twisting chain.
- **Custom end links.** End links that make both ends behave like the rest of the chain. The owner has an end pin (2026-11-28, files to be uploaded) and still has to design an alternate first pin with a wider contact area, to match the rest of the chain. Today the first pin touches the target only at its head's top edge, 12.65 mm above the table (the bare pin, 2–3 mm, lower down), and the last link's sleeve has no rim, so its position is not measured. When the end links exist: read their numbers from the STEP file (`experiments/snap_link_geometry.py`), update `data/chain-snap.yaml`, `synthetic.render_snap_photo` and the contact rule for end pins (`contact.py` and its TypeScript port, with new golden files), and drop the "keep the target off both ends" advice in snap-chain.md.
- **Bend limit.** The joint turns +102° / −127° and has no stop. Past about 105° a bend reads as a cut chain or with the bend's pin left out (with a warning). Add a stop to the link, or keep bends under about 105°.
- **Pitch and half width.** Measure the pitch over several links and a sleeve across with calipers, and put the means in `data/chain-snap.yaml`. How consistent is the pitch from link to link, and does it drift with wear? Pitch error feeds straight into the deskew. Per-chain numbers reach the phone by a chain card once item 4a's calibration is built.
- **Joint friction.** Does the snap clip hold a pose firmly enough, evenly across joints, and does its grip relax with use (PLA creep)?
- **Length and pitch.** Is the current pin count the right length, or should there be short and long chains? What is the smallest radius the chain must follow at a 13 mm pitch? Taking the perspective out needs at least 6 pins (7 without a focal length).

## 3. Contact model (snap-fit)

The snap-fit chain touches the target with the round sleeves and heads about every pin (`contact: round`): the curve point is the pin offset by 4.39 mm along the bisector of its two links, on convex and concave bends alike. The stadium-link rules of the other chains do not apply.

- **The model's error.** Simulated (2026-11-18, `experiments/gauge_simulation.py`; [real-world-tests.md](real-world-tests.md) step 2): exact at the contact points on arcs R25–R100, the spline between them within 0.026 mm; at an S's inflection (R60 each way) 0.017 mm at the points and 0.19 mm in the spline between them. Still open: real gauges and objects (item 1), and whether the spline needs help at inflections.
- **Ends.** Model the custom end links (item 2) so the end points are measured like the rest, rather than continuing the turn of the joint next to them.
- **Missed pins.** Where a pin is missed the curve bridges the gap (0.16 mm off on a 30 mm radius for the O-ring chain; not yet measured for this one). Should the missing pin be placed from its two neighbours, as a pin added by hand already is, or is "add it on the photo" enough?
- **Object side.** The phone page draws the contact edge on the photo and switches it with a tap. Does that make mistakes rare in practice?

## 4. Website: real-phone checks

What only the owner can check, on real hardware; the steps' exit criteria are in [web-build.md](web-build.md) (Status, Steps). The checklists below were written on 2026-11-17; since then the phone page was redesigned (2026-11-25), Settings tidied with a debug mode (2026-11-26) and the camera view reworked (2026-11-24 to 11-28), all checked in Chromium only, so read the checklists against the pages as they are now.

- **Phone page checklist** (step 2's exit criterion; the owner's checklist is in its Log, 2026-11-17).
- **Camera view checklist** (step 5; Log, 2026-11-17), including which rear camera opens (**O10**: never an ultra-wide, one lens throughout; since 2026-11-28 the camera can also be chosen in Settings, Camera).
- **Pairing and relay** with a real phone and Fusion (step 3's exit criterion).
- **Fusion add-in install path (O5).** Try the desktop page's three steps (download the zip, add it in Scripts and Add-Ins, Run on Startup) in a current Fusion; the menu names are unchecked.
- **One DXF imported in Fusion** (D19's simpler DXF).
- **The coach.** Recalibrate `web/src/overlay/coach.ts` with recorded phone video (its limits come from simulated renders).
- **The live view on Android.** It works on the iPhone (Safari, from build 18: 13 of 13 pins traced, the grab in a worker; build 24 with the faster tracking: frame 20+3 ms against build 22's 48+9). How it was built and measured: [experiments/live_view/README.md](../experiments/live_view/README.md) (Findings); the phone readings, build by build: web-build.md's Log, 2026-11-19 to 2026-11-23. Still to read on the Pixel (Chrome, and Firefox) and the Galaxy (Chrome), in the camera view's info line (Debug, in debug mode): "frame X+Y ms" (X the grab, Y the detector), where frames are grabbed ("grab in a worker", or "grab on the page:" and why), the region (`track W×H`, `tile`, `full`) and "analysis Z ms"; at ~25 cm and ~45 cm (span ~0.4 and ~0.25), still and while moving slowly, and once with `?grab=main` to compare. Does the count stay at 13 of 13 from ~0.25 down, and does "Move closer" come at about 0.22? If X is over ~30 ms on a phone, the WebGPU note in future-work.md applies.
- **A staging path** for deploying branches before `main`.

## 4a. Website: new features, in debug mode first (the owner's list, 2026-11-28)

Built behind `?debug` and promoted to the ordinary pages only after the owner has tried them on a phone. Details and sizes in [web-build.md](web-build.md), work queue items 13 and 14.

- **Colour channel for detection** (item 13): Gray, Red, Green or Blue, so coloured rims (orange on off-white) are found. About one session.
- **Per-chain calibration from a link** (item 14): a QR card per measured chain (pitch, half width, colours; no serial number, a chain is its numbers) sets up the phone page; the same setting typed by hand for home-printed chains; the desktop page measures with it too. Link syntax: D30 (confirmed 2026-11-28), in [chain-link.md](chain-link.md). Part A done 2026-11-28: the link in Python and TypeScript, `splinewire chain-link` (link and card), `measure --chain-link`/`--pitch`, `synth --pitch`. Part B (the phone and desktop pages, debug mode) follows item 13's merge; about one session.
  - **Before launch (before cards are printed for others): settle the half-width band.** The spec left it open; Part A chose ±10% of the preset's half width (`HALF_WIDTH_BAND` in `splinewire/chainlink.py` and `chainlink.ts`). That allows a sleeve printed 0.2 mm oversize (2.3%) and refuses a moved decimal point. Widening it later is safe; narrowing it would refuse cards already printed, so decide it from measured sleeves (item 2's calipers) before any card leaves the owner's hands.

## 5. End to end

- Real chain, real object (a pipe or a molding profile), from the phone to a curve in a Fusion sketch, timed.
- A first run was made on 2026-10-19 with the O-ring chain and the retired Windows app (photo → Copy points → Paste points worked); it was not timed or checked against the object. None yet with the snap-fit chain and the website.
- **Exit criterion:** the curve tracks the object within item 1's targets, under 15 s from photo to points on the desktop and under 30 s from photo to the curve in the sketch.

## 6. Workflow and CAD

- **Add from phone in Fusion** (built 2026-11-19, [fusion-palette.md](fusion-palette.md), "Trying it in Fusion"): on Windows, then on a Mac (D25: the Mac is assumed to behave as Windows did in S6). Does a curve land in the open sketch within seconds, is one Ctrl+Z enough, and does the palette survive hiding, workspace switches and a restart?
- **Sharp corners** (D27, add-in 0.3.0): marked on the desktop page's drawing and rebuilt from the walls either side; not yet tried on a real stepped part. Steps shorter than about three pitches come out wrong by several mm (future-work.md, *Sharp corners between short walls*).
- **Hand-placed pins.** When no dot is visible to snap to (a lamp's reflection), a pin is its tap fitted to the pitch of its links. Simulated, from a tap up to 3 px off, the median error is 0.03 mm for one hidden pin and 0.1 mm for two in a row (worst of 12 trials: 0.23 and 0.19 mm). How well do people place them on real glare-damaged photos, and is "zoom in and tap the middle of where the rim should be" enough guidance?
- **Fit points in Fusion.** Does `sketchFittedSplines.add` take the pasted sketch points as its fit points, or does the add-in fall back to plain coordinates and leave a second point at each spot ([fusion-import.md](fusion-import.md))? Paste points works either way.
