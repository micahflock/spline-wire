# Next steps — feasibility validation

Goal: before building the full stack, verify the highest-risk transitions. Each item below should produce a concrete yes/no answer with a measurable exit criterion. Tackle them in order; the order reflects risk (highest risk first) and dependency (each step unblocks the next).

## 1. Fiducial + LLM vision accuracy test

**Question:** can an LLM-based vision pipeline read fiducial positions off a phone photo accurately enough to be useful?

- Design 2–3 candidate fiducial schemes (e.g., small ArUco, custom glyph, numbered squares).
- 3D print a flat test part: a rigid strip with fiducials spaced at known positions (say every 10 mm) and one skew fiducial at one end. This removes the chain variable entirely for now.
- Photograph the strip with a phone from several angles and lighting conditions.
- Feed the photos to an LLM vision model with a prompt requesting structured JSON output of each fiducial's XY position.
- Compare returned coordinates to ground truth.
- **Exit criterion:** mean positional error under **1 mm** across the test set, measured after homography rectification.

## 2. Phone → computer handoff

**Question:** can we move a photo from phone to desktop with minimal friction and no custom app?

- Prototype the least-custom path first: mobile browser upload to a small local web app, or a cloud folder that the desktop side polls.
- Measure friction in seconds and taps from "photo taken" to "photo on laptop and processed."
- **Exit criterion:** under **15 seconds** of user-visible wait, with no custom iOS/Android app installation required.

## 3. Points → Fusion

**Question:** can we reliably place a list of XY points into a Fusion sketch?

- Write a minimal Fusion add-in (or standalone script) that reads a JSON or CSV file of XY points from a known location and creates sketch points in the active sketch.
- Test with hand-authored input, bypassing steps 1 and 2 entirely.
- **Exit criterion:** points appear in the active sketch, correctly scaled in millimeters, within **5 seconds** of triggering.

## 4. End-to-end static POC

**Question:** do the three pieces compose?

- Combine items 1, 2, and 3 using a *printed* test part (not a posed chain) so the chain-hardware problem is still deferred.
- **Exit criterion:** take a phone photo of a printed curve-shaped strip with fiducials, and see the corresponding points show up in Fusion within ~30 seconds end-to-end.

## 5. Real chain

**Only after 1–4 succeed.**

- Fabricate or adapt a stiff planar chain with printed fiducials on each link and a skew fiducial at one end.
- Repeat the end-to-end test with the real chain posed around a real curve (e.g., a pipe or doorknob).
- **Exit criterion:** measured points in Fusion track the real curve within the tolerance established in item 1.

## Deferred (explicitly not blocking MVP)

- Curve fitting (spline through the points).
- Support for CAD packages beyond Fusion.
- Custom mobile app with live capture and preview.
- 3D / non-planar curves.
- Fine-grained UX polish.
