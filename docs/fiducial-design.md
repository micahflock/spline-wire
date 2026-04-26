# Fiducial design

The in-depth tracker for the fiducial workstream. `open-questions.md` carries the high-level status; this file carries the concrete spec, rationale, open parameters, validation plan, and decision log.

## Status

**2026-04-25** — design locked in. Ring + 7-segment hex glyph at 5 mm tile size, 16-tile strip. See `docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md` for the full numeric spec, including outline-level corner rounding. The earlier "preliminary design at 8 mm with TrueType digit" parameter table below is superseded by that spec.

**2026-04-22** — preliminary design at concept level (superseded).

## Design spec

Each link face carries **two recessed features aligned perpendicular to the chain axis**:

1. **Solid filled circle** — position reference. Centroid is the measured point for this link.
2. **Glyph** — ID reference. A digit or character; alphabet size TBD.

Both features recess to the same Z. Print mechanics: link body prints in color A; at the recess-floor layer, swap to contrast color B and lay down the floor of both features; swap back to A. Single swap event per link, no AMS required.

The vector from circle center to glyph center defines a per-link local orientation. This replaces the separate end-of-chain skew fiducial — every link carries enough information to anchor its own local frame.

### Numeric parameters (starting guesses; feasibility test finalizes)

| Parameter | Starting guess | Pinned down by |
|---|---|---|
| Link pitch | 8–10 mm | Feasibility test |
| Link face width | 8 mm | Feasibility test |
| Circle diameter | 3 mm | Feasibility test |
| Glyph height | ~3 mm | Legibility test |
| Circle-glyph center spacing | ~4 mm | Feasibility test |
| Recess depth | 0.4–0.8 mm (2–4 layers at 0.2 mm) | Print quality test |
| Glyph alphabet | Digits 0–9, extendable to hex | Legibility test |
| Color pair | Black link / white recess | Lighting robustness test |

## Rationale

Three reasons this beats the alternatives (dense ArUco, color-coded dots, digit-inside-circle):

- **Per-link orientation.** Every link contributes `(x, y, θ)` instead of just `(x, y)`. Homography rectification becomes over-determined with N links, improving noise robustness and removing the single-point-of-failure of a lone end-of-chain skew marker.
- **Separation of concerns.** Circle = position (sub-pixel centroid fitting is trivial on a filled disc). Glyph = ID. Each feature does one job. Packing an ID *into* a position marker made both worse.
- **LLM-first friendliness.** Multimodal LLMs read digits and simple shapes reliably; they do not read binary-grid fiducials (ArUco/AprilTag) reliably. The MVP CV path is LLM-based per `next-steps.md` item 1, so the fiducial must be LLM-friendly.

### Graceful degradation

If a glyph is unreadable (smudge, glare, print defect), the circle still gives position. Neighbor adjacency can interpolate the missing ID. The CV → CAD JSON schema should include a per-point confidence field so downstream stages can flag interpolated links.

## Candidate alternates (validation controls)

The feasibility test compares the preliminary design against two alternates to confirm we haven't missed a sharper option:

- **Small ArUco 4×4.** Classical-CV friendly, LLM-unfriendly. The de facto fallback fiducial if the LLM path fails entirely.
- **Color-coded dot.** One solid dot per link; color encodes index. Best detection robustness under lighting variation but requires multi-color filament (AMS) beyond a handful of links.

## Validation plan (feeds `next-steps.md` item 1)

1. Print a flat 125 mm test strip — strip rather than chain, so the chain-hardware problem stays deferred. Three tile sets side by side:
   - **Set A**: preliminary design at 8 mm tiles.
   - **Set B**: preliminary design at 5 mm tiles — probes the size floor.
   - **Set C**: small ArUco 4×4 at 8 mm tiles — alternate comparison.
2. Photograph each with a phone: 3 angles (top-down, 20°, 45°) × 2 lighting conditions (daylight, indoor bulb) × 3 repeats = 18 images per set.
3. For each image, run two extraction paths:
   - **LLM path** — prompt a multimodal model for JSON `{link_index, circle_xy, glyph_xy, confidence}` per link.
   - **Classical path** — circle detection + glyph OCR (Sets A and B), or ArUco detection (Set C).
4. Compare returned positions to ground truth (strip is designed with known positions).

**Exit criterion** (from `next-steps.md` item 1): mean positional error under **1 mm** after homography rectification, on at least one set × one extraction path combination.

## Known risks and open sub-questions

- **FDM edge accuracy at small glyph strokes.** A 0.4 mm nozzle on a 3 mm digit gives stroke widths of ~0.8 mm, about 2 nozzle widths. Marginal. May need resin printing or paper-label fallback specifically for glyph features.
- **Glyph legibility under phone photography.** Untested. The feasibility image set is the answer.
- **Color pair choice.** High-contrast candidates: black/white, black/yellow, dark-blue/white. One pair must survive indoor / daylight / shadow variation. Classical CV and LLM may prefer different pairs.
- **Glyph alphabet cap.** Digits 0–9 cap the chain at 10 links, which is below the likely ~15 links at 8 mm pitch across 125 mm. Need hex (0–F) or two-character codes. Two-character widens the tile; single hex is cleaner.

## Decision log

- **2026-04-25** — Iterated the fiducial: filled circle → ring (annulus); TrueType digit → 7-segment-hex with outline-level rounded corners; multi-set comparison strip (A/B/C) → single canonical strip at the 5 mm size. Drove the change by promoting the SVG output to a print artifact, which made print-print-correct rendering load-bearing. Full spec at `docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md`.
- **2026-04-22** — Chose circle + adjacent-glyph pattern. Rejected: digit-inside-circle (cramped dual-purpose feature), dense small ArUco (LLM-hostile), color-coded dots (needs AMS beyond a few links). Keeping small ArUco as the classical-CV fallback design if the LLM path fails validation.
