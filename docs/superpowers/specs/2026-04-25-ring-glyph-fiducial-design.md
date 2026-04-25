# Ring + 7-segment glyph fiducial design

**Date:** 2026-04-25
**Status:** Spec approved; ready for implementation planning.
**Supersedes:** `docs/fiducial-design.md` "Numeric parameters (starting guesses)" table and the multi-set test strip layout (Sets A, B, C).

## Goal

Replace the preliminary filled-circle + TrueType-rendered digit fiducial with a print-correct ring + 7-segment glyph fiducial, drive both an LLM-validation PNG and a print-ready SVG from a single geometry source, and burn the multi-set abstraction (Sets A/B/C) down to a single canonical strip.

The SVG output of this design is intended to drive the actual physical 3D-printed test strip — not just the LLM validation rasterizer. That promotes the synthetic generator from "test fixture" to "print artifact source."

## Why this iteration

The original design at `docs/fiducial-design.md` flagged three open issues that this spec resolves:

1. **FDM edge accuracy at small glyph strokes.** TrueType-rendered digits at 3 mm height have stroke widths of ~0.8 mm — about two perimeters on a 0.4 mm nozzle, marginal legibility. Solution: design glyphs with explicit 0.86 mm strokes (exactly two perimeters), and shape them as 7-segment-display digits, which the LLM has seen at scale and the printer can render cleanly.
2. **Glyph alphabet cap.** Digits 0–9 cap the chain at 10 links, below the projected ~15-link target. Solution: extend to mixed-case hex `0123456789AbCdEF` (16 tiles), the standard 7-seg hex convention.
3. **Multi-set scaffolding overhead.** Sets A (8 mm), B (5 mm), and C (ArUco) existed for a comparison experiment. With one fiducial design now committed to, the multi-set abstraction is dead weight.

## Design decisions (one-line log)

| # | Decision | Reasoning |
|---|---|---|
| 1 | Single fiducial design across the entire strip — no Sets A/B/C. | Comparison experiment is over; we're committing. |
| 2 | Ring (annulus), not filled disc, OD 5.00 mm, ID 1.66 mm, wall 1.67 mm. | User-supplied; 1.67 mm wall ≈ 2 strokes (printable). |
| 3 | 7-segment-display-style digits, 2.58 × 5.00 mm, stroke 0.86 mm. | Two perimeters on 0.4 mm nozzle; LLM-friendly. |
| 4 | Ring → digit gap 1.72 mm (= 2 strokes). | Matches stroke unit; LLM-segmentable with clear gutter. |
| 5 | Hex alphabet `0123456789AbCdEF` (16 tiles), mixed case. | Industry-standard 7-seg hex; LLM has seen this convention. |
| 6 | Tile pitch 11.02 mm, inter-tile gap 1.72 mm (matches within-tile gap). | Self-consistent: every distinct-feature gap is exactly two strokes. |
| 7 | Strip 184.60 × 11.00 mm; margins 5/5/3/3 (L/R/T/B). | Hand-spannable; minimal vertical real estate. |
| 8 | Glyph axis is +x (`orientation_deg = 0.0`) — digit to right of ring. | User preference; supersedes the prior +y "above" layout. |
| 9 | Outline-level corner rounding (convex + concave) at radius 0.20 mm. | SVG drives a 3D print; matches FDM nozzle-radius corners. |
| 10 | Implementation: `shapely` for polygon union + buffer-based rounding. | Standard Python lib; Windows-clean install; minimal code. |
| 11 | Output: SVG and PNG, both from a shared geometry primitive list. | Single source of truth between the print artifact and the LLM input. |
| 12 | Module split: `validation/fiducial_geometry.py` (geometry) + `validation/synthetic.py` (renderers). | Geometry is the SVG/PNG anti-drift mechanism. |

## Tile geometry

```
         tile center axis (y = 5.5 mm on strip, all tiles)
             │
   ┌─────────┼─────────┐
   │   Ring  │  Digit  │
   │  Ø5.00  │ 2.58 W  │
   │ ID 1.66 │ 5.00 H  │
   │  wall   │ stroke  │
   │  1.67   │  0.86   │
   └─────────┴─────────┘
   ← 5.00 →← 1.72 →← 2.58 →
        ring        digit
       center  ←5.51→ center      = GLYPH_OFFSET (derived from spec)
```

| Quantity | Value (mm) | Source |
|---|---|---|
| Ring outer diameter | 5.00 | spec |
| Ring inner diameter | 1.66 | spec |
| Ring wall thickness | 1.67 | derived |
| Digit width | 2.58 | spec |
| Digit height | 5.00 | spec |
| Digit stroke | 0.86 | spec |
| Digit corner radius | 0.20 | spec |
| Within-tile gap (ring edge → digit left edge) | 1.72 | spec |
| Ring center → digit center (`glyph_offset_mm`) | 5.51 | derived |
| Tile content width (ring left → digit right) | 9.30 | derived |
| Tile content height | 5.00 | derived |

## 7-segment glyph layout

Segment positions in mm relative to the digit's bottom-left corner (W=2.58, H=5.00, S=0.86):

```
  ┌──────── a ────────┐    a:  x∈[0, 2.58]    y∈[4.14, 5.00]
  │                   │    b:  x∈[1.72, 2.58] y∈[2.50, 5.00]
  f                   b    c:  x∈[1.72, 2.58] y∈[0.00, 2.50]
  │                   │    d:  x∈[0, 2.58]    y∈[0.00, 0.86]
  ├──────── g ────────┤    e:  x∈[0, 0.86]    y∈[0.00, 2.50]
  │                   │    f:  x∈[0, 0.86]    y∈[2.50, 5.00]
  e                   c    g:  x∈[0, 2.58]    y∈[2.07, 2.93]
  │                   │
  └──────── d ────────┘
```

Vertical segments are split at y=H/2 (not y=H/2±S/2). When `g` is unlit, `f`+`e` butt at y=2.50 and form a continuous left edge with no gap; same for `b`+`c`. When `g` is lit, it overlaps the verticals slightly and adjacent rectangles fuse cleanly. This rule is what makes "0" render as a clean rounded rectangle (matching the user's sketch) while still letting "8" have a clean middle bar.

**Segment lookup:**

| Char | Lit segments | Char | Lit segments |
|---|---|---|---|
| `0` | a b c d e f | `8` | a b c d e f g |
| `1` | b c | `9` | a b c d f g |
| `2` | a b d e g | `A` | a b c e f g |
| `3` | a b c d g | `b` | c d e f g |
| `4` | b c f g | `C` | a d e f |
| `5` | a c d f g | `d` | b c d e g |
| `6` | a c d e f g | `E` | a d e f g |
| `7` | a b c | `F` | a e f g |

**Outline construction.** For each character:

1. Compute the union of lit segment rectangles via `shapely.ops.unary_union`.
2. Round all corners (convex outward, concave inward) via `union.buffer(-r).buffer(r)` with `r = corner_radius_mm`.
3. The result is **expected** to be a single `shapely.geometry.Polygon` with possibly one or more interior rings (holes for `0`, `4`, `6`, `8`, `9`, `A`, `b`, `d`). If shapely returns a `MultiPolygon` instead — meaning the segments didn't connect into a single shape — the geometry module raises (see the validation section below).

The geometry module owns the shapely calls. Render backends consume only vertex lists.

## Strip layout

| Quantity | Value | Source |
|---|---|---|
| Tile count | 16 | spec |
| Alphabet | `0123456789AbCdEF` | spec |
| Tile pitch (ring center → next ring center) | 11.02 mm | spec |
| Inter-tile gap (digit right → next ring left) | 1.72 mm | derived |
| Margins (left, right, top, bottom) | 5.0, 5.0, 3.0, 3.0 mm | spec |
| First ring center | (7.5, 5.5) mm | derived |
| Last ring center | (172.8, 5.5) mm | derived |
| Strip width | 184.60 mm | derived |
| Strip height | 11.00 mm | derived |
| Origin convention | bottom-left, +y up, +x right | unchanged |
| Glyph axis (ring → digit) | +x, `orientation_deg = 0.0` | spec |

## YAML spec format (`data/test-strip-spec.yaml`)

```yaml
units: mm

tile:
  ring:
    outer_diameter_mm: 5.00
    inner_diameter_mm: 1.66
  digit:
    width_mm: 2.58
    height_mm: 5.00
    stroke_mm: 0.86
    corner_radius_mm: 0.20
  ring_to_digit_gap_mm: 1.72

strip:
  count: 16
  alphabet: "0123456789AbCdEF"
  tile_pitch_mm: 11.02
  orientation_deg: 0.0
  margin_mm:
    left: 5.0
    right: 5.0
    top: 3.0
    bottom: 3.0
```

Anything derivable is derived in code (`glyph_offset_mm`, strip dimensions, per-tile ring centers). The YAML carries only inputs.

## Code architecture

### New: `validation/fiducial_geometry.py`

Pure-data primitive computation. Owns `shapely`. Exports:

- Constants: segment positions, segment lookup table.
- `Annulus(center_mm, outer_r_mm, inner_r_mm)` — frozen dataclass.
- `Polygon(exterior_mm: list[(x,y)], interiors_mm: list[list[(x,y)]])` — frozen dataclass.
- `digit_polygon(char, origin_mm, digit_dims, corner_radius_mm) -> Polygon`
- `tile_primitives(tile_id, ring_center_mm, spec) -> list[Annulus | Polygon]`
- `strip_primitives(spec) -> list[Annulus | Polygon]` — all 16 tiles, in tile-id order.

### Rewritten: `validation/synthetic.py`

Render layer only — no geometry math. Two functions, both consume `list[Annulus | Polygon]`:

- `render_png(primitives, out_path, px_per_mm) -> None` — drives PIL. Renders onto a freshly-created background-filled image (so cutouts work by overdraw). For each primitive in order: an `Annulus` is drawn as a filled outer ellipse in ink, then a smaller inner ellipse in background to cut out the hole. A `Polygon` is drawn as a filled exterior polygon in ink, then each interior polygon in background. PIL does not support fill-rule paths-with-holes natively; this overdraw approach is correct as long as no later primitive's bounding region overlaps a hole of an earlier primitive — which it cannot, given strip layout (rings and digits don't overlap each other).
- `render_svg(primitives, out_path) -> None` — emits an SVG document with `width`/`height` in mm. Fill is solid black (`#000000`) on white background (`#ffffff`); the SVG's purpose is print-driving, where colors are ignored by slicers but black-on-white is the conventional inspection appearance. Annuli render as a `<path>` with two arc subpaths and `fill-rule="evenodd"`. Digits render as `<path>` elements tracing exterior + holes, also `evenodd`.

### Simplified: `validation/schemas.py`

Drop `FiducialType` enum and `TileSet` wrapper. New shape:

```python
@dataclass(frozen=True)
class RingDims:
    outer_diameter_mm: float
    inner_diameter_mm: float

@dataclass(frozen=True)
class DigitDims:
    width_mm: float
    height_mm: float
    stroke_mm: float
    corner_radius_mm: float

@dataclass(frozen=True)
class StripSpec:
    units: str
    ring: RingDims
    digit: DigitDims
    ring_to_digit_gap_mm: float
    orientation_deg: float
    margin_mm: dict[str, float]
    tile_pitch_mm: float
    tiles: list[GroundTruthTile]

    @property
    def glyph_offset_mm(self) -> float: ...
    @property
    def strip_width_mm(self) -> float: ...
    @property
    def strip_height_mm(self) -> float: ...
```

`ImageDetection` drops the `set_name` field.

### Simplified: `validation/ground_truth.py`

`load_strip_spec(path) -> StripSpec` returns the flat shape. No multi-set dict, no fiducial-type branching, no ArUco loader.

### Simplified: `validation/prompts.py`

`prompt_for_set(set_name, fiducial_type)` becomes `prompt_for_strip()`. No branching.

### Updated: `validation/extraction.py`

`extract(image_path)` (drop `set_name`). Calls `prompt_for_strip()`.

### Updated: `validation/cli.py`

Drop the `img_path.stem.split("_", 1)[0]` set-name parse. Run extraction + scoring against the single `spec.tiles`.

### Updated: `validation/scoring.py`

Hardcoded `GLYPH_OFFSET_MM = 3.0` constant removed. Single source of truth moves to `spec.glyph_offset_mm`. Function signature changes:

```python
# Before:
def score_detection(detection: ImageDetection, ground_truth: list[GroundTruthTile]) -> ScoringResult: ...

# After:
def score_detection(detection: ImageDetection, spec: StripSpec) -> ScoringResult: ...
```

Inside the function, `spec.tiles` replaces `ground_truth`, and `spec.glyph_offset_mm` replaces the module constant. `validation/cli.py` updates its call site to pass `spec` instead of `spec.tiles`.

### Updated: `scripts/generate_synthetic.py`

Single call. Emit both:

```python
prims = strip_primitives(spec)
render_png(prims, Path("data/photos/strip.png"), px_per_mm=40)
render_svg(prims, Path("data/photos/strip.svg"))
```

### Config

- `pyproject.toml` — add `shapely>=2.0` to dependencies.

### Cleanup

- Delete `data/photos/A_top_day_01.png`, `data/photos/B_top_day_01.png`.

## Validation at the YAML loader boundary

Spec-load-time checks (raised as `ValueError` with specific messages):

- Every char in `strip.alphabet` exists in the 7-segment lookup table.
- `len(strip.alphabet) == strip.count`.
- `tile.ring.inner_diameter_mm < tile.ring.outer_diameter_mm`.
- `tile.digit.corner_radius_mm * 2 ≤ tile.digit.stroke_mm` (otherwise the buffer-shrink eats a stroke entirely).
- `tile.digit.width_mm ≥ 3 * tile.digit.stroke_mm`.
- `tile.digit.height_mm ≥ 5 * tile.digit.stroke_mm`.
- `tile_pitch_mm ≥ tile.ring.outer_diameter_mm/2 + ring_to_digit_gap_mm + tile.digit.width_mm + tile.ring.outer_diameter_mm/2`. (Tiles must not overlap. The right-hand side is the minimum pitch for digit-of-N right edge to touch ring-of-N+1 left edge.)

If shapely's union+buffer pipeline produces a `MultiPolygon` (a character whose segments don't connect — not the case for any of `0–F`, but a sanity guard for future glyph additions), the geometry module raises a clear error.

## Testing strategy

### New: `tests/test_fiducial_geometry.py`

- `digit_polygon('0', ...)` returns one exterior + one interior.
- `digit_polygon('1', ...)` returns one exterior + zero interiors.
- `digit_polygon('8', ...)` returns one exterior + two interiors.
- Each character's polygon bounds fit inside the design box (with at most `corner_radius_mm` inset on convex corners).
- `tile_primitives(...)` returns one `Annulus` followed by one `Polygon`.
- `strip_primitives(spec)` returns 16 annuli + 16 polygons in tile-id order.
- `spec.glyph_offset_mm == 5.51`.

### Updated: `tests/test_synthetic.py`

Raster:
- `render_png(...)` produces an RGB image with width × height matching `spec.strip_width_mm × spec.strip_height_mm × px_per_mm`.
- For each of 16 tiles: a dark pixel exists at half-radius from the ring center (annulus is inked).
- For each ring: the very center pixel is light/background (ring is hollow, not a disc).
- Total dark pixel count is in the right order of magnitude.

SVG:
- `render_svg(...)` produces well-formed XML.
- Document `width`/`height` attributes match `spec.strip_width_mm` / `spec.strip_height_mm` in mm.
- Document contains 16 ring elements + 16 digit elements.

Cross-check:
- Both backends consume the same `strip_primitives(spec)`. Every ring center coordinate (mm) maps to a dark pixel in the PNG and to an enclosing `<path>` element in the SVG.

### Updated: `tests/test_ground_truth.py`, `tests/test_schemas.py`, `tests/test_scoring.py`

Drop `TileSet`/`FiducialType` references. `tests/test_scoring.py` replaces hardcoded `GLYPH_OFFSET_MM = 3.0` with `spec.glyph_offset_mm = 5.51`.

### Not tested

- Visual fidelity of rounded outlines at the pixel level (trust shapely; "looks right" is brittle).
- LLM extraction accuracy on the new strip — that's the validation experiment, not a unit test.

## Out of scope

- ArUco-based fallback fiducial. The fallback path was Set C; with the multi-set abstraction gone, ArUco is on hold. If the LLM path fails validation on the new strip, ArUco can be reintroduced later as a separate workstream.
- Print-color simulation. The synthetic image stays monochrome (`_INK = (20,20,20)`, `_BG = (240,240,240)`). The SVG is single-fill-color too. The eventual physical print uses a contrast color pair (TBD), but the synthetic doesn't try to mock that.
- True-arc SVG output. Buffered shapely polygons emit sampled-arc segments (many short lines per arc). Slicers handle this fine. If a future iteration needs true `<path>` arc commands, that's a render-layer enhancement, not a geometry change.
- Updating `docs/fiducial-design.md`. The original doc's "Numeric parameters (starting guesses)" table is superseded by this spec. The `fiducial-design.md` file should get a status update + decision-log entry pointing here, but that's a documentation task in the implementation plan, not part of this spec.
