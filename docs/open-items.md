# Open items

[← Back to the README](../README.md)

What should be done, in risk order. This list and [future-work.md](future-work.md) (what might be done) replace the earlier `next-steps.md` and `open-questions.md` (consolidated 2026-11-18). Everything here is about the snap-fit chain (`data/chain-snap.yaml`, [snap-chain.md](snap-chain.md)) and the website ([web-build.md](web-build.md)); the O-ring and print-in-place chains are not tracked.

## 1. Real-world accuracy: plan a revised test series  ← next

**Where it stands.** The owner's tests on the live site (2026-11, the TypeScript engine, phone photos) show that real-photo accuracy is good; the figures are to be recorded with the new series. The one scored real photo is older: a printed test plaque, 0.098 mm worst pin and 0.063 mm rms at a 25° tilt (2026-10-01, the retired Windows app). The old test matrix was written for the Windows app and the O-ring dot chain and has been dropped. The new series for the snap-fit chain and the website is planned in [real-world-tests.md](real-world-tests.md); its tools are built (`splinewire fixed-chain`, `splinewire test-gauges`, the test page at `spikes/accuracy/`, `splinewire score`), and printing and phase 0 are next. It should answer:

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

- **The model's error.** Simulated (2026-11-18, `experiments/gauge_simulation.py`; [real-world-tests.md](real-world-tests.md) step 2): exact at the contact points on arcs R25–R100, the spline between them within 0.026 mm; at an S's inflection (R60 each way) 0.017 mm at the points and 0.19 mm in the spline between them. Still open: real gauges and objects (item 1), and whether the spline needs help at inflections.
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
- **Live view resolution: built 2026-11-19, the real-phone figures missing.** The live detector read the whole frame scaled to 800 px on the long side, and that, not the still, limited how far the phone could be: the 13-pin snap-fit chain was lost below a span of ~0.40 (span: the pins' greatest distance plus a link's width, over the frame's short side), exactly the coach's old `minSpan`. Now (`experiments/live_view/`, synthetic frames, Node on a machine about 2× the docs' reference; Chromium headless, no GPU):
  - **One-pass threshold** (`blobs.ts`, `thresholds()`): both masks from running window sums in one pass, buffers kept between frames, padded masks so the components need no bounds checks, the components' stack preallocated. The masks are bit-identical to `localThreshold`'s (`blobs.test.ts`), and the detections identical on all 105 frame/fiducial combinations of the benchmark. `detectBlobs` per frame, before → after: 800 px 10–13 → 3–4 ms, 1200 px 23–29 → 8 ms, 1600 px 40–54 → 11–15 ms (the thresholds alone at 800 px: 7.9 → 2.7 ms).
  - **Regions** (`overlay/region.ts`): while the chain is tracked, only the box round it (1.5 links of margin, plus a link per pin not seen) is drawn, at the scale that makes its rims 18 work px across (all rims are found from ~14.5 px: 13/13 at 14.3–14.9, 9/13 at 13.1–13.7), never finer than 2016 px equivalent. With no chain: the whole frame at 1200 px, then each quarter (0.6 of each side) at 2016 px equivalent, in turn. A chain is followed from frame to frame (as trace.ts follows it) and dropped after 4 frames without 3 of its pins, or 2 analyses that find no chain.

    | Span | Pins found, 800 px frame (before) | After: found, from a cold start | Tracked region, detector per frame |
    |---|---|---|---|
    | 0.55 | 13 | 13 from the first frame | 0.17 MP, 2 ms |
    | 0.40 | 12 | 13 from the first frame | 0.09 MP, 1.4 ms |
    | 0.30 | 0 | 13 from the first frame | 0.07 MP, 1.2 ms |
    | 0.22 | 0 | 13 from the second frame (a quarter) | 0.07 MP, 1.3 ms |
    | 0.16 | 0 | 12 (rims 14 px at 2016 px) | 0.06 MP, 1.0 ms |

    Scanning costs 7–9 ms a frame (1.1 MP), against 10–13 ms for the old 800 px frame. Still or moving at a fifth of a link a frame (twice the coach's motion limit), the tracker keeps all 13 pins. Sharpness is read at the old full frame's scale (`Window.refScale`), so the coach's limits keep their meaning: on blurred renders the old frame and a tracked region read 0.60 / 0.62 (σ 4 px) and 0.41 / 0.41 (σ 7); only mild blur reads sharper (0.77 / 0.89 at σ 2).
  - **The grab in a worker** (`camera/grab.ts`, `grab-worker.ts`): `createImageBitmap(video, …)` costs the page as long as drawing it (37 of 37 ms for a 12 MP frame in headless Chromium), but taking a `VideoFrame` of the video costs ~0 ms, so the page hands the frame to a worker that draws the region, reads it back and finds the blobs. A first check compares the worker's copy of a region with the page's own (orientation and pixels); without VideoFrame (Firefox on Android may lack it), after a failed check, an error or no answer in 2 s, or with `?grab=main`, the page grabs as before. In Chromium (`e2e/camera.spec.ts`, the O-ring chain): the page's own time per frame ~39 ms (grab 34 + detector 4.8) → 0.1–0.2 ms, and a 16 ms timer late by median 26 / p95 45 / max 91 ms → 0.2 / 2.7 / 7 ms; the worker's grab is still ~35–40 ms per 12 MP frame there (no GPU), the detector 1.4–1.8 ms.
  - **Coach** (`coach.ts`): `minSpan` 0.40 / 0.33 → 0.26 / 0.22 (enter / keep, of the pins' bounding box, at least 0.23 in the span above). The still measures its worst pin to 0.003–0.004 mm from span 0.40 to 0.26 and 0.007 mm at 0.22, and loses a pin at 0.18 (`still_by_span.py`, four poses, clean renders). Nothing else in the coach depends on the work size.
  - **Tests**: `blobs.test.ts` (masks bit for bit; windows; spans 0.30 and 0.22 on three tables, ordered by the real analysis), `region.test.ts`, `session.test.ts` (the trace lands on the pins in video pixels), `grab.test.ts`; `e2e/camera-snap.spec.ts` at spans 0.55 (page grab), 0.30 (tracked, the still taken and measured) and 0.22 (tracked, all 13 pins, "Move closer", no still).
  - **For the owner to read off a real phone** (camera view, Live camera, snap-fit chain; the line under the view): "frame X+Y ms" (X the grab, Y the detector) and whether it says "in a worker", the region (`track W×H`, `tile`, `full`), and "analysis Z ms", on the iPhone (Safari), the Pixel (Chrome, and Firefox) and the Galaxy (Chrome): at ~25 cm and at ~45 cm (span ~0.4 and ~0.25), still and while moving slowly; and once with `?grab=main` in the address to compare. Also: does the count stay at 13 of 13 from ~0.25 down, and does "Move closer" come at about 0.22? If X is over ~30 ms on a phone, the GPU note in future-work.md (WebGPU detection) applies.
- **A staging path** for deploying branches before `main`.

## 5. End to end

- Real chain, real object (a pipe or a molding profile), from the phone to a curve in a Fusion sketch, timed.
- A first run was made on 2026-10-19 with the O-ring chain and the retired Windows app (photo → Copy points → Paste points worked); it was not timed or checked against the object. None yet with the snap-fit chain and the website.
- **Exit criterion:** the curve tracks the object within item 1's targets, under 15 s from photo to points on the desktop and under 30 s from photo to the curve in the sketch.

## 6. Workflow and CAD

- **Add from phone in Fusion** (built 2026-11-19, [fusion-palette.md](fusion-palette.md), "Trying it in Fusion"): on Windows, then on a Mac (D25: the Mac is assumed to behave as Windows did in S6). Does a curve land in the open sketch within seconds, is one Ctrl+Z enough, and does the palette survive hiding, workspace switches and a restart?

- **Hand-placed pins.** When no dot is visible to snap to (a lamp's reflection), a pin is its tap fitted to the pitch of its links. Simulated, from a tap up to 3 px off, the median error is 0.03 mm for one hidden pin and 0.1 mm for two in a row (worst of 12 trials: 0.23 and 0.19 mm). How well do people place them on real glare-damaged photos, and is "zoom in and tap the middle of where the rim should be" enough guidance?
- **Fit points in Fusion.** Does `sketchFittedSplines.add` take the pasted sketch points as its fit points, or does the add-in fall back to plain coordinates and leave a second point at each spot ([fusion-import.md](fusion-import.md))? Paste points works either way.
