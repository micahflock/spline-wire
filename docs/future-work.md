# Future work

[← Back to the README](../README.md)

What might be done, in no particular order. Nothing here is needed for the measuring workflow; what should be done is in [open-items.md](open-items.md).

## Website and engine

- **WebGPU detection.** Move the detector's pixel work to the GPU where it measurably helps (D18's order of work). Today a 12 MP photo measures in 1.1 s in Chromium and detection took 2.2 s on an iPhone before the faster detector, so this waits for a phone that shows the need.
  - *Looked at 2026-11-18 for the live camera view: not now.* The live view already analyses ~16 frames a second (the coach needs a few); its limit was resolution, which CPU work fixed (2026-11-19, open-items.md item 4, live view resolution: a one-pass threshold, regions round the tracked chain, the grab in a worker; the detector now costs 1–2 ms a tracked frame and 7–9 ms scanning). Only the local threshold suits a GPU (~60% of the live detector's time); the connected components (~38%) are serial, and readback adds a frame or two of latency. For the still, the GPU could save perhaps 0.2–0.5 s of 1–2 s, at the cost of a second implementation to keep in line with the golden files. WebGPU is in Safari on iOS 26+ and Chrome on Android (Samsung Xclipse GPUs uncertain), not in Firefox on Android, so a CPU fallback stays. If a GPU is ever needed for the live view, WebGL2 (a gray + threshold shader, async R8 readback) reaches every tested phone; the trigger is a phone's "frame X+Y ms" over ~30 ms with the regions (X is the grab, now in a worker where the browser can). Figures: `experiments/live_view/`.
- **Core memory.** The TypeScript detector peaks about 90 MB above the loaded 12 MP photo (Node, peak RSS), freed afterwards. What is left is the full-size float32 copy of the photo for refinement (48 MB) and the Int32 label image (48 MB). Work on it only if a phone needs it.
- **Thick ring fiducials in the TypeScript port**, if a chain ever uses them (the snap-fit rim is read by its hole and is supported).
- **A native phone app** with live capture and preview. The phone page's in-page camera and live overlay cover this for now.

## CAD

- **A Fusion add-in that pairs with the phone directly** (was O7): the add-in receives curves from the relay in an HTML palette, with no desktop page and no clipboard. Researched 2026-11-18: a palette loading the hosted desktop page in a `?fusion` mode, after a spike in real Fusion. Design, options and the spike: [fusion-palette.md](fusion-palette.md).
- **Setting the CAD sketch origin and orientation.** Today the origin is the first pin and the axes follow the photo. Let the person pick them, or align them to something in the photo or the sketch.
- **CAD packages beyond Fusion.** The DXF (mm) and the 1:1 SVG are a partial fallback for them today.

## Measurement

- **3D, non-planar curves.**
- **The fiducials' ellipse shapes as extra tilt information.** Not needed so far: the pitch and the EXIF focal length fix the plane.
