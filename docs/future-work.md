# Future work

[← Back to the README](../README.md)

What might be done, in no particular order. Nothing here is needed for the measuring workflow; what should be done is in [open-items.md](open-items.md).

## Website and engine

- **WebGPU detection.** Move the detector's pixel work to the GPU where it measurably helps (D18's order of work). Today a 12 MP photo measures in 1.1 s in Chromium and detection took 2.2 s on an iPhone before the faster detector, so this waits for a phone that shows the need.
- **Core memory.** The TypeScript detector peaks about 90 MB above the loaded 12 MP photo (Node, peak RSS), freed afterwards. What is left is the full-size float32 copy of the photo for refinement (48 MB) and the Int32 label image (48 MB). Work on it only if a phone needs it.
- **Thick ring fiducials in the TypeScript port**, if a chain ever uses them (the snap-fit rim is read by its hole and is supported).
- **A native phone app** with live capture and preview. The phone page's in-page camera and live overlay cover this for now.

## CAD

- **A Fusion add-in that pairs with the phone directly** (was O7): the add-in receives curves from the relay in an HTML palette, with no desktop page and no clipboard.
- **Setting the CAD sketch origin and orientation.** Today the origin is the first pin and the axes follow the photo. Let the person pick them, or align them to something in the photo or the sketch.
- **CAD packages beyond Fusion.** The DXF (mm) and the 1:1 SVG are a partial fallback for them today.

## Measurement

- **3D, non-planar curves.**
- **The fiducials' ellipse shapes as extra tilt information.** Not needed so far: the pitch and the EXIF focal length fix the plane.
