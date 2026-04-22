# Architecture (first pass)

The system is a pipeline: physical curve → chain → photo → points → CAD. Each stage can be developed and tested in isolation, with plain-text data contracts between stages.

## Stage 1 — Physical capture (hardware)

- Planar chain, roughly 125 mm long. Link pitch TBD; candidate range 5–10 mm.
- Each pin has enough friction to hold its posed angle under gravity and gentle handling, but low enough to pose by hand without tools.
- Fiducial on every link face. At least one end link carries a distinct skew fiducial for perspective rectification.
- Chain stays planar — no out-of-plane twist.

## Stage 2 — Image capture (phone)

- User lays the posed chain on a flat, contrasting surface.
- User takes one photo with the native camera app (no custom camera app in v1).
- The skew fiducial plus known chain geometry should be enough to rectify from a moderately off-axis shot, removing the need for a perfectly top-down capture.

## Stage 3 — Transport (phone → compute)

- Open question — see `open-questions.md`.
- MVP candidate: user uploads the photo to a local web app (mobile browser → laptop), or drops it in a shared cloud folder that the desktop side polls.
- Avoid requiring a custom mobile app for v1.

## Stage 4 — Computer vision

- Detect every link fiducial → list of `(u, v)` image coordinates + orientation.
- Detect the skew fiducial → 4+ correspondences to rectify the image plane to the chain plane.
- Apply a homography to recover `(x, y)` chain-plane coordinates.
- Order the links along the chain — either via an index encoded in each fiducial, or by spatial nearest-neighbor after rectification.
- **MVP implementation:** LLM-based multimodal vision, with a well-designed prompt that returns structured JSON.
- **Fallback:** classical CV (ArUco / AprilTag / custom) if LLM accuracy is insufficient.

## Stage 5 — CAD integration

- MVP output: a list of XY points loaded into an Autodesk Fusion sketch.
- Candidate integration paths:
  - Fusion Python add-in that reads a local file (JSON or CSV) and creates sketch points.
  - A script the user runs from Fusion's Scripts and Add-ins dialog.
  - Generic DXF or SVG export that Fusion can `Insert → Insert DXF/SVG`.
- Stretch: fit a spline or polyline through the points and emit that directly.

## Data contracts (sketched)

Between stages, keep formats boring and text-based so each stage can evolve independently.

CV output → CAD input:

```json
{
  "units": "mm",
  "points": [[x, y], [x, y], ...],
  "order": [0, 1, 2, ...],
  "skew_ref": { "link_index": 0 }
}
```

Version the schema in the repo (e.g., `schema/v1.json`) so stage 4 and stage 5 can evolve without silently breaking each other.

## Component diagram (text)

```
[Posed chain] → [Phone photo] → [Upload / sync] → [CV service]
                                                        │
                                                        ▼
                                              [points.json (schema v1)]
                                                        │
                                                        ▼
                                              [Fusion add-in] → [Sketch]
```
