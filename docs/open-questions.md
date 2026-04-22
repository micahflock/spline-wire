# Open questions

Tracked gaps in the end-to-end workflow. Resolve as prototyping proceeds. Cross-reference `next-steps.md` — several of these get answered by the first few experiments.

## Fiducials

**Preliminary direction (2026-04-22):** per-link solid circle + glyph side-by-side, perpendicular to the chain axis, both recessed from the link face with a single filament swap for contrast. The per-link circle→glyph axis provides orientation, so no separate end-of-chain skew fiducial is needed. Full spec and validation plan in `fiducial-design.md`.

Still open:

- What size fits on a 5–10 mm link face and still prints and scans reliably? (Preliminary design pushes toward the 8–10 mm end of that range.)
- Can a hobbyist FDM printer hit the needed contrast and edge accuracy? If not, what's the fallback — resin printing, printed paper labels stuck on, laser-engraved inserts?
- Which glyph alphabet — digits 0–9 (10 links), hex 0–F (16 links), or two-character codes (100+)? Bounded by legibility at the chosen tile size.

Resolved:

- ~~What pattern per link?~~ LLM-friendly custom: solid circle + adjacent glyph.
- ~~Do fiducials need to encode a link index?~~ Yes — glyph encodes ID; spatial adjacency corroborates.
- ~~Skew fiducial shape?~~ No separate feature. Every link's circle→glyph axis serves as an orientation reference.

## Computer vision

- LLM-based vs. classical (OpenCV / AprilTag libraries) — accuracy comparison, cost per call, latency per photo?
- What positional accuracy (in mm) is "useful" as CAD reference geometry? Propose a starting target of **±1 mm**; tighten if downstream CAD use demands it.
- How robust does the pipeline need to be to varying lighting, shadows, and background clutter? Is a "please photograph on a white sheet of paper" constraint acceptable for v1?
- If LLM-based, which model, and is one photo enough or should we send several at once?

## Phone → computer

- Is there a smooth path that avoids a custom phone app in v1?
- If web-based: can a mobile browser upload work without friction? (Likely yes.)
- If cloud folder: Dropbox, iCloud, or Drive — which has the least setup?
- Is it acceptable to require the user's laptop and phone be on the same local network?

## CAD integration

- Fusion API: add-in vs. standalone script — which is lower friction for the user to install and re-run?
- What does "drop points into a sketch" look like in the Fusion SDK in practice? (Needs hands-on exploration.)
- Should the tool emit DXF or SVG as a universal fallback for non-Fusion users?
- Spline fitting — which algorithm? Candidates: natural cubic, centripetal Catmull-Rom, B-spline via least-squares.

## Units, scale, origin

- How does the system know real-world scale? The known link pitch can act as an intrinsic ruler — is that enough, or do we also want the user to include a printed reference scale in frame?
- Where is the origin — first link, centroid of the point cloud, or user-selectable?
- How is chain orientation (which end is "start"?) communicated — visually via the skew fiducial, or implicitly?

## Chain hardware

- Is ~125 mm the right length, or should there be a couple of sizes (short / long)?
- How is pin friction tuned, and does it drift with use?
- Does the chain need any visible scale reference printed on it (ruler marks) to aid rectification, or is the fiducial pitch enough?
