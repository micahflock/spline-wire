# Accuracy tests (the real-world test series)

`index.html` is the test page for docs/real-world-tests.md, phases 1–3: photograph a fixed-chain
plaque (`splinewire fixed-chain`) or the chain on a contact gauge (`splinewire test-gauges`), and the
page measures the photo with the TypeScript engine and scores it against the target's known shape at
once, so a bad shot can be taken again on the spot. Its logic is `src/testbench/bench.ts`, the
scorer `src/pipeline/scoring.ts` (a port of `splinewire/scoring.py`, pinned to it by
`scoring.test.ts`); `e2e/accuracy.spec.ts` checks the page in Chromium.

Each test keeps its photo on the phone (IndexedDB). The report it downloads is what
`splinewire score --report` reads, so the saved photos can be scored again by Python, the
reference, with the EXIF, estimated and best-fit focal lengths (`--fit-focal`).

`test-photo.jpg` is a render of the S plaque for the end-to-end test
(`web/scripts/golden_scoring.py` writes it, with `src/pipeline/targets.json`).

## For the owner

Open `spikes/accuracy/` on the site, on the iPhone and on the Android phone.

1. **This phone and chain.** Name the phone (it goes into the file names), pick the chain (white
   rims on black links, or orange rims on white links) and set its number of links (rims).
   Leave the focal length empty unless a photo comes out with "estimated" as its focal source: on a
   Pixel, the page's camera gives no 35 mm focal length, so either open the phone page once and let it
   learn the lens from a library photo, or type the phone's main camera here (24 mm on recent
   iPhones and Pixels). **Chain pitch:** the pitch measured with calipers in phase 0 (12.98 mm
   for the first chain measured), used for the chain on a gauge or an object; a plaque is always
   measured at the 13.000 mm it was printed at. Leave it empty for the design pitch.
2. **The test.** Pick the target and what you are about to do: the tilt you aim for, where in the
   frame the chain is, the surface and the light. These go into the report as they are.
3. **Take the photo**, with the page's camera (the main route) or from the photo library (a
   camera-app photo). The result shows at once: green passes the proposed mark (a plaque's worst pin
   under 0.3 mm; a gauge's worst point under 0.5 mm and its radius within 0.5 mm, end points left
   out), red does not. A second line shows the same photo with the focal length estimated from the
   chain, for comparison.
4. **After a session:** Download the report (JSON) and Save all photos (on iPhone: Save to Files),
   keep them together in one folder, and Copy the report into a message. Then
   `uv run splinewire score --report spline-wire-tests-<phone>.json --fit-focal` scores them all
   again in Python.

The page keeps every test until **Delete every test on this phone**; save the photos and the report
first. Nothing leaves the phone unless you save or send it.
