# Ring + 7-Segment Glyph Fiducial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the multi-set (A/B/C) circle+TrueType-digit fiducial design with a single-set ring + 7-segment-hex-digit fiducial. Both PNG (LLM input) and SVG (3D-print artifact) render from a shared geometry primitive list.

**Architecture:** Geometry computation lives in a new `validation/fiducial_geometry.py` module that owns `shapely`. Render backends (`validation/synthetic.py`) consume primitive lists and emit PNG via PIL or SVG via string formatting. The multi-set abstraction (`TileSet`, `FiducialType`) is removed; `StripSpec` becomes flat. Spec dimensions live in YAML; everything derivable (glyph offset, strip width/height, per-tile centers) is computed by the loader.

**Tech Stack:** Python 3.12+, Pillow, PyYAML, NumPy (existing). Adding `shapely>=2.0` for polygon union and corner rounding.

**Spec reference:** `docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md`. The plan implements that spec exactly.

**Note on intermediate test state:** Tasks 5–11 form a refactor across schemas, ground truth, prompts, extraction, scoring, synthetic, and CLI. Each task leaves *its own* tests passing; tests in modules not yet updated will be temporarily broken (import errors or signature mismatches) between Task 5 and Task 11. The full test suite passes again at the end of Task 11.

---

## File structure (post-implementation)

| File | Status | Responsibility |
|---|---|---|
| `validation/fiducial_geometry.py` | NEW | Geometry primitives (`Annulus`, `Polygon` dataclasses), segment table, `digit_polygon`, `tile_primitives`, `strip_primitives`. Owns `shapely`. |
| `validation/synthetic.py` | REWRITE | `render_png` and `render_svg` consuming primitive lists. No geometry math. |
| `validation/schemas.py` | SIMPLIFY | Drop `FiducialType`, `TileSet`. Add `RingDims`, `DigitDims`. Flat `StripSpec` with derived properties. `ImageDetection` drops `set_name`. |
| `validation/ground_truth.py` | SIMPLIFY | Single-strip loader with validation checks. |
| `validation/prompts.py` | SIMPLIFY | `prompt_for_strip()`. No set/fiducial branching. |
| `validation/extraction.py` | UPDATE | Drop `set_name` parameter. |
| `validation/scoring.py` | UPDATE | Take `spec: StripSpec` instead of `list[GroundTruthTile]`. Use `spec.glyph_offset_mm`. |
| `validation/cli.py` | UPDATE | Drop `set_name` parsing. |
| `scripts/generate_synthetic.py` | UPDATE | Single call; emit both PNG and SVG. |
| `data/test-strip-spec.yaml` | REWRITE | Flat schema. |
| `pyproject.toml` | UPDATE | Add `shapely>=2.0`. |
| `docs/fiducial-design.md` | UPDATE | Status update + decision log entry. |
| `tests/test_schemas.py` | REWRITE | Test new dataclasses. |
| `tests/test_ground_truth.py` | REWRITE | Test new flat loader. |
| `tests/test_synthetic.py` | REWRITE | Raster + SVG tests against new design. |
| `tests/test_scoring.py` | UPDATE | New signature. |
| `tests/test_extraction.py` | UPDATE | No `set_name`. |
| `tests/test_fiducial_geometry.py` | NEW | Geometry unit tests. |
| `data/photos/A_top_day_01.png`, `B_top_day_01.png` | DELETE | Stale; superseded by `strip.png`. |

---

## Task 1: Add `shapely` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add shapely to dependencies**

Edit `pyproject.toml`. The current dependencies block is:

```toml
dependencies = [
    "anthropic>=0.96.0",
    "numpy>=2.4.4",
    "opencv-python>=4.13.0.92",
    "pillow>=12.2.0",
    "pyyaml>=6.0.3",
]
```

Add `"shapely>=2.0",` (alphabetical placement after pyyaml is fine). Result:

```toml
dependencies = [
    "anthropic>=0.96.0",
    "numpy>=2.4.4",
    "opencv-python>=4.13.0.92",
    "pillow>=12.2.0",
    "pyyaml>=6.0.3",
    "shapely>=2.0",
]
```

- [ ] **Step 2: Install the dependency**

Run: `uv sync`

Expected: completes without error; `shapely` appears in `uv.lock`.

- [ ] **Step 3: Verify shapely imports**

Run: `uv run python -c "from shapely.geometry import box; from shapely.ops import unary_union; print('ok')"`

Expected output: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add shapely dependency for fiducial geometry"
```

---

## Task 2: Create `fiducial_geometry.py` — primitive dataclasses and segment table

**Files:**
- Create: `validation/fiducial_geometry.py`
- Create: `tests/test_fiducial_geometry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fiducial_geometry.py`:

```python
from validation.fiducial_geometry import (
    Annulus,
    Polygon,
    SEGMENT_LAYOUT,
    SEGMENTS_FOR,
)


def test_annulus_holds_center_and_radii():
    a = Annulus(center_mm=(7.5, 5.5), outer_radius_mm=2.5, inner_radius_mm=0.83)
    assert a.center_mm == (7.5, 5.5)
    assert a.outer_radius_mm == 2.5
    assert a.inner_radius_mm == 0.83


def test_polygon_holds_exterior_and_interiors():
    p = Polygon(
        exterior_mm=[(0, 0), (1, 0), (1, 1), (0, 1)],
        interiors_mm=[[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]],
    )
    assert len(p.exterior_mm) == 4
    assert len(p.interiors_mm) == 1
    assert len(p.interiors_mm[0]) == 4


def test_segments_for_covers_all_hex_chars():
    expected = "0123456789AbCdEF"
    assert set(SEGMENTS_FOR.keys()) == set(expected)


def test_segment_layout_returns_axis_aligned_box_for_each_segment():
    # SEGMENT_LAYOUT[seg] is a callable (W, H, S) -> (x0, y0, x1, y1)
    for seg in "abcdefg":
        x0, y0, x1, y1 = SEGMENT_LAYOUT[seg](2.58, 5.00, 0.86)
        assert x0 < x1
        assert y0 < y1


def test_segment_a_is_top_horizontal():
    x0, y0, x1, y1 = SEGMENT_LAYOUT["a"](2.58, 5.00, 0.86)
    assert (x0, y0, x1, y1) == (0.0, 4.14, 2.58, 5.00)


def test_segment_g_is_centered_horizontal():
    x0, y0, x1, y1 = SEGMENT_LAYOUT["g"](2.58, 5.00, 0.86)
    assert (x0, x1) == (0.0, 2.58)
    # y centered at H/2 = 2.5, with stroke 0.86 → [2.07, 2.93]
    assert y0 == 2.07
    assert y1 == 2.93


def test_vertical_segments_split_at_half_height_for_seamless_fusion():
    # f and e meet at y=H/2 (not y=H/2±S/2). When g is unlit, f+e form a
    # continuous left edge — this is what makes "0" render as a clean
    # outline. See spec section "7-segment glyph layout".
    H = 5.00
    _, fy0, _, fy1 = SEGMENT_LAYOUT["f"](2.58, H, 0.86)
    _, ey0, _, ey1 = SEGMENT_LAYOUT["e"](2.58, H, 0.86)
    assert fy0 == H / 2
    assert ey1 == H / 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Create the module with primitives and segment table**

Create `validation/fiducial_geometry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Annulus:
    """Ring fiducial: filled outer disc with a concentric inner cutout."""
    center_mm: tuple[float, float]
    outer_radius_mm: float
    inner_radius_mm: float


@dataclass(frozen=True)
class Polygon:
    """A planar polygon in mm, possibly with holes (interiors)."""
    exterior_mm: list[tuple[float, float]]
    interiors_mm: list[list[tuple[float, float]]]


# 7-segment glyph layout. Each entry maps a segment label to a function
# (W, H, S) -> (x0, y0, x1, y1) returning the segment's bounding box in
# mm relative to the digit's bottom-left corner.
#
# Vertical segments split at y=H/2 (not y=H/2 ± S/2) so f+e and b+c fuse
# seamlessly when g is unlit. See the spec ("7-segment glyph layout") for
# the rationale.
SegmentBox = Callable[[float, float, float], tuple[float, float, float, float]]

SEGMENT_LAYOUT: dict[str, SegmentBox] = {
    "a": lambda W, H, S: (0.0,   H - S,         W,   H),                # top horizontal
    "b": lambda W, H, S: (W - S, H / 2.0,       W,   H),                # upper right vertical
    "c": lambda W, H, S: (W - S, 0.0,           W,   H / 2.0),          # lower right vertical
    "d": lambda W, H, S: (0.0,   0.0,           W,   S),                # bottom horizontal
    "e": lambda W, H, S: (0.0,   0.0,           S,   H / 2.0),          # lower left vertical
    "f": lambda W, H, S: (0.0,   H / 2.0,       S,   H),                # upper left vertical
    "g": lambda W, H, S: (0.0,   (H - S) / 2.0, W,   (H + S) / 2.0),    # middle horizontal
}


# Lit segments per character (mixed-case hex; standard 7-seg convention).
SEGMENTS_FOR: dict[str, str] = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    "A": "abcefg",
    "b": "cdefg",
    "C": "adef",
    "d": "bcdeg",
    "E": "adefg",
    "F": "aefg",
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/fiducial_geometry.py tests/test_fiducial_geometry.py
git commit -m "Add fiducial_geometry primitives and 7-segment layout table"
```

---

## Task 3: `digit_polygon` — shapely-based outline computation

**Files:**
- Modify: `validation/fiducial_geometry.py`
- Modify: `tests/test_fiducial_geometry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fiducial_geometry.py`:

```python
import pytest

from validation.fiducial_geometry import digit_polygon


# Reusable digit dimensions for tests.
DIGIT_W = 2.58
DIGIT_H = 5.00
DIGIT_STROKE = 0.86
DIGIT_RADIUS = 0.20


def _digit(char, origin=(0.0, 0.0)):
    return digit_polygon(
        char,
        origin_mm=origin,
        width_mm=DIGIT_W,
        height_mm=DIGIT_H,
        stroke_mm=DIGIT_STROKE,
        corner_radius_mm=DIGIT_RADIUS,
    )


def test_zero_has_one_exterior_and_one_interior():
    p = _digit("0")
    assert len(p.interiors_mm) == 1


def test_one_has_no_interior():
    p = _digit("1")
    assert len(p.interiors_mm) == 0


def test_eight_has_two_interiors():
    # "8" lights all segments; the upper and lower bowls form two holes.
    p = _digit("8")
    assert len(p.interiors_mm) == 2


def test_digit_bounds_fit_inside_design_box():
    # Every character's outline must lie inside the nominal digit cell
    # (with at most corner_radius_mm of inset on convex outer corners).
    for char in "0123456789AbCdEF":
        p = _digit(char)
        xs = [x for x, _ in p.exterior_mm]
        ys = [y for _, y in p.exterior_mm]
        assert min(xs) >= -1e-9
        assert min(ys) >= -1e-9
        assert max(xs) <= DIGIT_W + 1e-9
        assert max(ys) <= DIGIT_H + 1e-9


def test_origin_offset_translates_polygon():
    p = _digit("1", origin=(10.0, 20.0))
    xs = [x for x, _ in p.exterior_mm]
    ys = [y for _, y in p.exterior_mm]
    # "1" lights b+c only — a single rectangle on the right side.
    # x range should be ~[10 + W - S, 10 + W] = [11.72, 12.58]; y range ~[20, 25].
    assert min(xs) >= 10.0 + DIGIT_W - DIGIT_STROKE - 1e-9
    assert max(xs) <= 10.0 + DIGIT_W + 1e-9
    assert min(ys) >= 20.0 - 1e-9
    assert max(ys) <= 20.0 + DIGIT_H + 1e-9


def test_unknown_character_raises():
    with pytest.raises(KeyError):
        _digit("Z")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: ImportError (or AttributeError) on `digit_polygon`.

- [ ] **Step 3: Implement `digit_polygon`**

Add to `validation/fiducial_geometry.py` (at the bottom):

```python
from shapely.geometry import box, MultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


def digit_polygon(
    char: str,
    origin_mm: tuple[float, float],
    width_mm: float,
    height_mm: float,
    stroke_mm: float,
    corner_radius_mm: float,
) -> Polygon:
    """Build the rounded outline of a 7-segment hex character.

    Returns a Polygon whose coordinates are in mm, translated by origin_mm.
    The outline has all corners rounded (convex outward, concave inward) at
    corner_radius_mm via shapely's morphological-opening pipeline.
    """
    if char not in SEGMENTS_FOR:
        raise KeyError(f"no 7-segment glyph defined for character {char!r}")

    rects = [
        box(*SEGMENT_LAYOUT[seg](width_mm, height_mm, stroke_mm))
        for seg in SEGMENTS_FOR[char]
    ]
    union = unary_union(rects)
    rounded = union.buffer(-corner_radius_mm).buffer(corner_radius_mm)

    if isinstance(rounded, MultiPolygon):
        raise ValueError(
            f"digit {char!r}: corner rounding produced disconnected pieces — "
            f"the segment layout is wrong for this character"
        )
    if not isinstance(rounded, ShapelyPolygon):
        raise ValueError(
            f"digit {char!r}: unexpected shapely result {type(rounded).__name__}"
        )

    ox, oy = origin_mm
    exterior = [(x + ox, y + oy) for x, y in rounded.exterior.coords]
    interiors = [
        [(x + ox, y + oy) for x, y in interior.coords]
        for interior in rounded.interiors
    ]
    return Polygon(exterior_mm=exterior, interiors_mm=interiors)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: all tests pass (the original 7 plus the new 6 = 13 tests).

- [ ] **Step 5: Commit**

```bash
git add validation/fiducial_geometry.py tests/test_fiducial_geometry.py
git commit -m "Add digit_polygon with shapely-based outline rounding"
```

---

## Task 4: `tile_primitives` and `strip_primitives`

**Files:**
- Modify: `validation/fiducial_geometry.py`
- Modify: `tests/test_fiducial_geometry.py`

These functions need a `StripSpec`-like object to read dimensions from. To avoid circular imports, accept dimensions as plain parameters; the caller (in Task 7) will unpack `spec` fields.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fiducial_geometry.py`:

```python
from validation.fiducial_geometry import (
    tile_primitives,
    strip_primitives,
    StripGeometry,
)


def _strip_geom(count=16, alphabet="0123456789AbCdEF"):
    return StripGeometry(
        ring_outer_diameter_mm=5.00,
        ring_inner_diameter_mm=1.66,
        digit_width_mm=2.58,
        digit_height_mm=5.00,
        digit_stroke_mm=0.86,
        digit_corner_radius_mm=0.20,
        ring_to_digit_gap_mm=1.72,
        tile_pitch_mm=11.02,
        margin_left_mm=5.0,
        margin_bottom_mm=3.0,
        count=count,
        alphabet=alphabet,
    )


def test_tile_primitives_returns_one_annulus_then_one_polygon():
    geom = _strip_geom()
    prims = tile_primitives(tile_id="0", ring_center_mm=(7.5, 5.5), geom=geom)
    assert len(prims) == 2
    assert isinstance(prims[0], Annulus)
    assert isinstance(prims[1], Polygon)


def test_tile_primitives_ring_matches_ring_dimensions():
    geom = _strip_geom()
    prims = tile_primitives(tile_id="3", ring_center_mm=(7.5, 5.5), geom=geom)
    annulus = prims[0]
    assert annulus.center_mm == (7.5, 5.5)
    assert annulus.outer_radius_mm == pytest.approx(2.5)
    assert annulus.inner_radius_mm == pytest.approx(0.83)


def test_tile_primitives_digit_centered_on_ring_y_axis():
    # Digit's vertical center should equal ring center y (digit centered on ring axis).
    # Digit's horizontal center is offset to the right by glyph_offset_mm.
    geom = _strip_geom()
    prims = tile_primitives(tile_id="3", ring_center_mm=(7.5, 5.5), geom=geom)
    digit = prims[1]
    xs = [x for x, _ in digit.exterior_mm]
    ys = [y for _, y in digit.exterior_mm]
    digit_center_y = (min(ys) + max(ys)) / 2
    assert digit_center_y == pytest.approx(5.5, abs=0.001)
    # glyph_offset = 2.5 + 1.72 + 1.29 = 5.51
    digit_center_x = (min(xs) + max(xs)) / 2
    assert digit_center_x == pytest.approx(7.5 + 5.51, abs=0.001)


def test_strip_primitives_emits_count_annuli_and_count_polygons():
    geom = _strip_geom(count=16)
    prims = strip_primitives(geom)
    annuli = [p for p in prims if isinstance(p, Annulus)]
    polygons = [p for p in prims if isinstance(p, Polygon)]
    assert len(annuli) == 16
    assert len(polygons) == 16


def test_strip_primitives_first_ring_at_left_margin_plus_radius():
    # First ring center x should be margin_left + ring_outer_radius.
    geom = _strip_geom()
    prims = strip_primitives(geom)
    first_annulus = next(p for p in prims if isinstance(p, Annulus))
    assert first_annulus.center_mm[0] == pytest.approx(7.5)


def test_strip_primitives_ring_centers_step_by_pitch():
    geom = _strip_geom()
    prims = strip_primitives(geom)
    annuli = [p for p in prims if isinstance(p, Annulus)]
    assert annuli[1].center_mm[0] - annuli[0].center_mm[0] == pytest.approx(11.02)
    assert annuli[15].center_mm[0] == pytest.approx(7.5 + 15 * 11.02)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: ImportError on `tile_primitives`/`strip_primitives`/`StripGeometry`.

- [ ] **Step 3: Implement the functions**

Add to `validation/fiducial_geometry.py`:

```python
@dataclass(frozen=True)
class StripGeometry:
    """Plain-parameter view of a StripSpec for the geometry layer.

    Decoupled from validation.schemas.StripSpec to avoid circular imports.
    The ground_truth loader (Task 7) builds a StripGeometry from a StripSpec.
    """
    ring_outer_diameter_mm: float
    ring_inner_diameter_mm: float
    digit_width_mm: float
    digit_height_mm: float
    digit_stroke_mm: float
    digit_corner_radius_mm: float
    ring_to_digit_gap_mm: float
    tile_pitch_mm: float
    margin_left_mm: float
    margin_bottom_mm: float
    count: int
    alphabet: str

    @property
    def glyph_offset_mm(self) -> float:
        return (
            self.ring_outer_diameter_mm / 2.0
            + self.ring_to_digit_gap_mm
            + self.digit_width_mm / 2.0
        )

    @property
    def tile_center_y_mm(self) -> float:
        return self.margin_bottom_mm + max(
            self.ring_outer_diameter_mm, self.digit_height_mm
        ) / 2.0

    @property
    def first_ring_center_x_mm(self) -> float:
        return self.margin_left_mm + self.ring_outer_diameter_mm / 2.0


def tile_primitives(
    tile_id: str,
    ring_center_mm: tuple[float, float],
    geom: StripGeometry,
) -> list[Annulus | Polygon]:
    """Build the primitives for one tile: ring + rounded digit polygon."""
    rx, ry = ring_center_mm
    annulus = Annulus(
        center_mm=(rx, ry),
        outer_radius_mm=geom.ring_outer_diameter_mm / 2.0,
        inner_radius_mm=geom.ring_inner_diameter_mm / 2.0,
    )
    digit_origin = (
        rx + geom.glyph_offset_mm - geom.digit_width_mm / 2.0,
        ry - geom.digit_height_mm / 2.0,
    )
    digit = digit_polygon(
        tile_id,
        origin_mm=digit_origin,
        width_mm=geom.digit_width_mm,
        height_mm=geom.digit_height_mm,
        stroke_mm=geom.digit_stroke_mm,
        corner_radius_mm=geom.digit_corner_radius_mm,
    )
    return [annulus, digit]


def strip_primitives(geom: StripGeometry) -> list[Annulus | Polygon]:
    """Build the full strip's primitives in tile-id order."""
    if len(geom.alphabet) != geom.count:
        raise ValueError(
            f"alphabet length {len(geom.alphabet)} does not match count {geom.count}"
        )
    prims: list[Annulus | Polygon] = []
    cy = geom.tile_center_y_mm
    for i, char in enumerate(geom.alphabet):
        cx = geom.first_ring_center_x_mm + i * geom.tile_pitch_mm
        prims.extend(tile_primitives(char, (cx, cy), geom))
    return prims
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fiducial_geometry.py -v`

Expected: 19 tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/fiducial_geometry.py tests/test_fiducial_geometry.py
git commit -m "Add tile_primitives and strip_primitives for full-strip geometry"
```

---

## Task 5: Replace `schemas.py` with new dataclass shape

**WARNING:** This task drops `FiducialType` and `TileSet`. After this commit, `ground_truth.py`, `synthetic.py`, `extraction.py`, and `cli.py` (which import them) will fail at collection time. Tasks 6–11 fix them in order. Run only the targeted test in Step 4 to validate this task.

**Files:**
- Modify: `validation/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the new failing tests**

Replace `tests/test_schemas.py` with:

```python
import pytest

from validation.schemas import (
    DigitDims,
    DetectedTile,
    GroundTruthTile,
    ImageDetection,
    RingDims,
    ScoringResult,
    StripSpec,
)


def test_ground_truth_tile_holds_position_and_id():
    tile = GroundTruthTile(tile_id="0", center_mm=(7.5, 5.5), orientation_deg=0.0)
    assert tile.tile_id == "0"
    assert tile.center_mm == (7.5, 5.5)
    assert tile.orientation_deg == 0.0


def test_ring_dims_holds_outer_and_inner_diameter():
    r = RingDims(outer_diameter_mm=5.00, inner_diameter_mm=1.66)
    assert r.outer_diameter_mm == 5.00
    assert r.inner_diameter_mm == 1.66


def test_digit_dims_holds_size_stroke_and_radius():
    d = DigitDims(width_mm=2.58, height_mm=5.00, stroke_mm=0.86, corner_radius_mm=0.20)
    assert d.width_mm == 2.58
    assert d.corner_radius_mm == 0.20


def _build_spec(count=16, alphabet="0123456789AbCdEF"):
    tiles = [
        GroundTruthTile(tile_id=alphabet[i], center_mm=(7.5 + i * 11.02, 5.5), orientation_deg=0.0)
        for i in range(count)
    ]
    return StripSpec(
        units="mm",
        ring=RingDims(outer_diameter_mm=5.00, inner_diameter_mm=1.66),
        digit=DigitDims(width_mm=2.58, height_mm=5.00, stroke_mm=0.86, corner_radius_mm=0.20),
        ring_to_digit_gap_mm=1.72,
        orientation_deg=0.0,
        margin_mm={"left": 5.0, "right": 5.0, "top": 3.0, "bottom": 3.0},
        tile_pitch_mm=11.02,
        tiles=tiles,
    )


def test_strip_spec_glyph_offset_is_derived():
    spec = _build_spec()
    # 2.5 + 1.72 + 1.29 = 5.51
    assert spec.glyph_offset_mm == pytest.approx(5.51)


def test_strip_spec_strip_width_is_derived():
    spec = _build_spec()
    # 5.0 + 15*11.02 + 5.51 + 1.29 + 5.0 = 184.60
    assert spec.strip_width_mm == pytest.approx(184.60, abs=0.001)


def test_strip_spec_strip_height_is_derived():
    spec = _build_spec()
    # bottom 3 + max(5,5) + top 3 = 11
    assert spec.strip_height_mm == pytest.approx(11.0)


def test_strip_spec_has_count_tiles():
    spec = _build_spec()
    assert len(spec.tiles) == 16


def test_detected_tile_stores_pixel_positions_and_confidence():
    d = DetectedTile(
        tile_id="3",
        circle_xy_px=(120.5, 200.0),
        glyph_xy_px=(160.0, 200.0),
        confidence=0.92,
    )
    assert d.tile_id == "3"
    assert d.confidence == 0.92


def test_image_detection_drops_set_name():
    det = DetectedTile(
        tile_id="0",
        circle_xy_px=(10.0, 10.0),
        glyph_xy_px=(20.0, 10.0),
        confidence=1.0,
    )
    img = ImageDetection(
        image_path="data/photos/strip.png",
        model="claude-opus-4-7",
        prompt_version="v1",
        detections=[det],
        raw_response="...",
        latency_seconds=3.1,
    )
    assert img.model == "claude-opus-4-7"
    assert img.detections[0].tile_id == "0"


def test_scoring_result_captures_mean_and_per_point_errors():
    r = ScoringResult(
        image_path="data/photos/strip.png",
        per_point_error_mm={"0": 0.4, "1": 0.6},
        mean_error_mm=0.5,
        homography_rmse_px=1.2,
    )
    assert r.mean_error_mm == 0.5
    assert r.per_point_error_mm["1"] == 0.6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: ImportError on `RingDims`/`DigitDims` (or other failures).

- [ ] **Step 3: Replace `validation/schemas.py`**

Replace the entire file contents with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruthTile:
    tile_id: str
    center_mm: tuple[float, float]
    orientation_deg: float


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
    margin_mm: dict[str, float]   # keys: left, right, top, bottom
    tile_pitch_mm: float
    tiles: list[GroundTruthTile]

    @property
    def glyph_offset_mm(self) -> float:
        return (
            self.ring.outer_diameter_mm / 2.0
            + self.ring_to_digit_gap_mm
            + self.digit.width_mm / 2.0
        )

    @property
    def strip_width_mm(self) -> float:
        # margin.left + (count-1)*pitch + glyph_offset + digit.width/2 + margin.right
        count = len(self.tiles)
        return (
            self.margin_mm["left"]
            + (count - 1) * self.tile_pitch_mm
            + self.glyph_offset_mm
            + self.digit.width_mm / 2.0
            + self.margin_mm["right"]
        )

    @property
    def strip_height_mm(self) -> float:
        tile_h = max(self.ring.outer_diameter_mm, self.digit.height_mm)
        return self.margin_mm["bottom"] + tile_h + self.margin_mm["top"]


@dataclass(frozen=True)
class DetectedTile:
    tile_id: str
    circle_xy_px: tuple[float, float]
    glyph_xy_px: tuple[float, float] | None
    confidence: float


@dataclass
class ImageDetection:
    image_path: str
    model: str
    prompt_version: str
    detections: list[DetectedTile]
    raw_response: str
    latency_seconds: float


@dataclass
class ScoringResult:
    image_path: str
    per_point_error_mm: dict[str, float]
    mean_error_mm: float
    homography_rmse_px: float
```

- [ ] **Step 4: Run the schema tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: 10 tests pass.

(Other test files are expected to fail collection at this point; that's fixed by Tasks 6–11.)

- [ ] **Step 5: Commit**

```bash
git add validation/schemas.py tests/test_schemas.py
git commit -m "Simplify schemas: flat StripSpec with derived dimensions, drop multi-set"
```

---

## Task 6: Replace `data/test-strip-spec.yaml`

**Files:**
- Modify: `data/test-strip-spec.yaml`

- [ ] **Step 1: Replace the YAML**

Replace the entire contents of `data/test-strip-spec.yaml` with:

```yaml
# Canonical ground-truth specification for the printed test strip.
# All positions are in millimeters; origin is the lower-left corner of the strip.
# Single fiducial design: ring (annulus) + 7-segment hex digit beside it.
# Reference: docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md

units: mm

# --- Per-tile geometry ---
tile:
  ring:
    outer_diameter_mm: 5.00
    inner_diameter_mm: 1.66        # wall = 1.67 mm
  digit:
    width_mm: 2.58                 # 3 × stroke
    height_mm: 5.00                # vertically centered on ring center
    stroke_mm: 0.86                # 0.4 mm nozzle, 2 perimeters
    corner_radius_mm: 0.20         # outline-level rounding via shapely buffer
  ring_to_digit_gap_mm: 1.72       # 2 strokes; ring outer edge → digit left edge

# Derived by ground_truth.load_strip_spec (not in spec):
#   glyph_offset_mm = 2.5 + 1.72 + 1.29 = 5.51 mm
#   strip_width_mm  = 5.0 + 15*11.02 + 5.51 + 1.29 + 5.0 = 184.60 mm
#   strip_height_mm = 3.0 + 5.0 + 3.0 = 11.00 mm

# --- Strip layout ---
strip:
  count: 16
  alphabet: "0123456789AbCdEF"     # mixed-case hex (7-seg display convention)
  tile_pitch_mm: 11.02             # ring-center to ring-center; inter-tile gap = 1.72 mm
  orientation_deg: 0.0             # ring → digit axis is +x for every tile
  margin_mm:
    left: 5.0
    right: 5.0
    top: 3.0
    bottom: 3.0
```

- [ ] **Step 2: Verify YAML parses**

Run: `uv run python -c "import yaml; print(yaml.safe_load(open('data/test-strip-spec.yaml')))"`

Expected: a Python dict prints; no traceback.

- [ ] **Step 3: Commit**

```bash
git add data/test-strip-spec.yaml
git commit -m "Replace test-strip-spec.yaml with flat single-set schema"
```

---

## Task 7: Rewrite `ground_truth.py` with validation checks

**Files:**
- Modify: `validation/ground_truth.py`
- Modify: `tests/test_ground_truth.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_ground_truth.py` with:

```python
from pathlib import Path

import pytest

from validation.ground_truth import build_strip_geometry, load_strip_spec


def test_load_strip_spec_returns_flat_strip_with_16_tiles(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.units == "mm"
    assert len(spec.tiles) == 16
    assert spec.tiles[0].tile_id == "0"
    assert spec.tiles[15].tile_id == "F"


def test_load_strip_spec_first_tile_at_expected_position(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    # margin.left + ring.outer/2 = 5 + 2.5 = 7.5
    # margin.bottom + max(ring.outer, digit.height)/2 = 3 + 2.5 = 5.5
    assert spec.tiles[0].center_mm == pytest.approx((7.5, 5.5))


def test_load_strip_spec_pitch_steps_ring_centers(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.tiles[1].center_mm[0] - spec.tiles[0].center_mm[0] == pytest.approx(11.02)
    assert spec.tiles[15].center_mm[0] == pytest.approx(7.5 + 15 * 11.02)


def test_load_strip_spec_orientation_propagated_to_tiles(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    for tile in spec.tiles:
        assert tile.orientation_deg == 0.0


def test_load_strip_spec_derived_glyph_offset(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.glyph_offset_mm == pytest.approx(5.51)


def test_build_strip_geometry_matches_spec(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    geom = build_strip_geometry(spec)
    assert geom.ring_outer_diameter_mm == spec.ring.outer_diameter_mm
    assert geom.alphabet == "0123456789AbCdEF"
    assert geom.count == 16


def test_load_rejects_bad_alphabet_char(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(alphabet="0123456789AbcdEF"), encoding="utf-8")
    # lowercase 'c' is not in SEGMENTS_FOR (we use uppercase 'C' for hex)
    with pytest.raises(ValueError, match="character 'c'"):
        load_strip_spec(bad_yaml)


def test_load_rejects_alphabet_count_mismatch(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(alphabet="012345"), encoding="utf-8")
    with pytest.raises(ValueError, match="alphabet"):
        load_strip_spec(bad_yaml)


def test_load_rejects_inverted_ring_diameters(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(ring_inner=6.0, ring_outer=5.0), encoding="utf-8")
    with pytest.raises(ValueError, match="inner_diameter"):
        load_strip_spec(bad_yaml)


def test_load_rejects_oversized_corner_radius(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(corner_radius=0.5), encoding="utf-8")
    # corner_radius * 2 > stroke (0.86) → invalid
    with pytest.raises(ValueError, match="corner_radius"):
        load_strip_spec(bad_yaml)


def test_load_rejects_pitch_too_small(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(_yaml_with(pitch=8.0), encoding="utf-8")
    # min pitch = 9.30; 8.0 causes tile overlap
    with pytest.raises(ValueError, match="tile_pitch"):
        load_strip_spec(bad_yaml)


def _yaml_with(
    ring_outer=5.00, ring_inner=1.66,
    digit_w=2.58, digit_h=5.00, stroke=0.86, corner_radius=0.20,
    gap=1.72, count=16, alphabet="0123456789AbCdEF",
    pitch=11.02,
):
    return f"""
units: mm
tile:
  ring:
    outer_diameter_mm: {ring_outer}
    inner_diameter_mm: {ring_inner}
  digit:
    width_mm: {digit_w}
    height_mm: {digit_h}
    stroke_mm: {stroke}
    corner_radius_mm: {corner_radius}
  ring_to_digit_gap_mm: {gap}
strip:
  count: {count}
  alphabet: "{alphabet}"
  tile_pitch_mm: {pitch}
  orientation_deg: 0.0
  margin_mm:
    left: 5.0
    right: 5.0
    top: 3.0
    bottom: 3.0
"""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ground_truth.py -v`

Expected: fail (most likely import errors or "build_strip_geometry not found").

- [ ] **Step 3: Replace `validation/ground_truth.py`**

Replace the entire file with:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from validation.fiducial_geometry import SEGMENTS_FOR, StripGeometry
from validation.schemas import (
    DigitDims,
    GroundTruthTile,
    RingDims,
    StripSpec,
)


def load_strip_spec(path: Path) -> StripSpec:
    """Load and validate the test strip YAML spec."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ring = RingDims(
        outer_diameter_mm=raw["tile"]["ring"]["outer_diameter_mm"],
        inner_diameter_mm=raw["tile"]["ring"]["inner_diameter_mm"],
    )
    digit = DigitDims(
        width_mm=raw["tile"]["digit"]["width_mm"],
        height_mm=raw["tile"]["digit"]["height_mm"],
        stroke_mm=raw["tile"]["digit"]["stroke_mm"],
        corner_radius_mm=raw["tile"]["digit"]["corner_radius_mm"],
    )
    ring_to_digit_gap_mm = raw["tile"]["ring_to_digit_gap_mm"]

    strip = raw["strip"]
    count = strip["count"]
    alphabet = strip["alphabet"]
    pitch = strip["tile_pitch_mm"]
    orientation = strip["orientation_deg"]
    margin = {k: float(v) for k, v in strip["margin_mm"].items()}

    _validate_ring(ring)
    _validate_digit(digit)
    _validate_alphabet(alphabet, count)
    _validate_pitch(pitch, ring, digit, ring_to_digit_gap_mm)

    first_x = margin["left"] + ring.outer_diameter_mm / 2.0
    tile_y = margin["bottom"] + max(ring.outer_diameter_mm, digit.height_mm) / 2.0
    tiles = [
        GroundTruthTile(
            tile_id=alphabet[i],
            center_mm=(first_x + i * pitch, tile_y),
            orientation_deg=orientation,
        )
        for i in range(count)
    ]

    return StripSpec(
        units=raw["units"],
        ring=ring,
        digit=digit,
        ring_to_digit_gap_mm=ring_to_digit_gap_mm,
        orientation_deg=orientation,
        margin_mm=margin,
        tile_pitch_mm=pitch,
        tiles=tiles,
    )


def build_strip_geometry(spec: StripSpec) -> StripGeometry:
    """Build the geometry-layer view from a StripSpec for renderers."""
    return StripGeometry(
        ring_outer_diameter_mm=spec.ring.outer_diameter_mm,
        ring_inner_diameter_mm=spec.ring.inner_diameter_mm,
        digit_width_mm=spec.digit.width_mm,
        digit_height_mm=spec.digit.height_mm,
        digit_stroke_mm=spec.digit.stroke_mm,
        digit_corner_radius_mm=spec.digit.corner_radius_mm,
        ring_to_digit_gap_mm=spec.ring_to_digit_gap_mm,
        tile_pitch_mm=spec.tile_pitch_mm,
        margin_left_mm=spec.margin_mm["left"],
        margin_bottom_mm=spec.margin_mm["bottom"],
        count=len(spec.tiles),
        alphabet="".join(t.tile_id for t in spec.tiles),
    )


def _validate_ring(ring: RingDims) -> None:
    if ring.inner_diameter_mm >= ring.outer_diameter_mm:
        raise ValueError(
            f"ring inner_diameter_mm ({ring.inner_diameter_mm}) must be less than "
            f"outer_diameter_mm ({ring.outer_diameter_mm})"
        )


def _validate_digit(digit: DigitDims) -> None:
    if digit.corner_radius_mm * 2 > digit.stroke_mm:
        raise ValueError(
            f"digit corner_radius_mm ({digit.corner_radius_mm}) * 2 must not exceed "
            f"stroke_mm ({digit.stroke_mm}); rounding would eat the stroke"
        )
    if digit.width_mm < 3 * digit.stroke_mm:
        raise ValueError(
            f"digit width_mm ({digit.width_mm}) must be at least 3 * stroke_mm "
            f"({3 * digit.stroke_mm}) to fit segments horizontally"
        )
    if digit.height_mm < 5 * digit.stroke_mm:
        raise ValueError(
            f"digit height_mm ({digit.height_mm}) must be at least 5 * stroke_mm "
            f"({5 * digit.stroke_mm}) to fit segments vertically"
        )


def _validate_alphabet(alphabet: str, count: int) -> None:
    if len(alphabet) != count:
        raise ValueError(
            f"alphabet length {len(alphabet)} does not match strip count {count}"
        )
    for ch in alphabet:
        if ch not in SEGMENTS_FOR:
            raise ValueError(
                f"alphabet contains character {ch!r} with no 7-segment glyph defined"
            )


def _validate_pitch(
    pitch: float,
    ring: RingDims,
    digit: DigitDims,
    gap: float,
) -> None:
    min_pitch = ring.outer_diameter_mm + gap + digit.width_mm
    if pitch < min_pitch:
        raise ValueError(
            f"tile_pitch_mm ({pitch}) is below minimum {min_pitch:.3f} (tiles would overlap)"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ground_truth.py -v`

Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/ground_truth.py tests/test_ground_truth.py
git commit -m "Rewrite ground_truth loader for flat single-strip spec with validation"
```

---

## Task 8: Simplify `prompts.py`

**Files:**
- Modify: `validation/prompts.py`

- [ ] **Step 1: Replace `validation/prompts.py`**

Replace the entire file with:

```python
from __future__ import annotations

PROMPT_V2_RING_GLYPH = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of fiducial tiles.

Each tile has TWO features:
1. A RING (an annulus / hollow circle) — the position reference. Treat \
   the ring's center as the tile's position.
2. A 7-SEGMENT-DISPLAY-STYLE HEX DIGIT to the right of the ring — the ID \
   reference. Characters are from 0-9, A, b, C, d, E, F (mixed case, \
   industry-standard 7-segment hex convention).

The ring → digit axis is horizontal (digit to the right of the ring) for \
every tile.

For EVERY tile visible in the image, return a JSON array with one object \
per tile:

[
  {
    "tile_id": "<character as string, e.g. '0' or 'A' or 'b'>",
    "circle_xy_px": [<ring center x>, <ring center y>],
    "glyph_xy_px": [<digit center x>, <digit center y>],
    "confidence": <float 0-1>
  },
  ...
]

Coordinates are pixel coordinates in the original image, with (0,0) at \
top-left. Use your best estimate of each feature's center. If you cannot \
read the digit, set "tile_id" to "?" and set "confidence" below 0.5.

Return ONLY the JSON array, no prose, no markdown fences.\
"""


def prompt_for_strip() -> str:
    """Return the extraction prompt for the canonical ring + 7-seg-hex strip."""
    return PROMPT_V2_RING_GLYPH
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from validation.prompts import prompt_for_strip; print(prompt_for_strip()[:50])"`

Expected: prints the first 50 characters of the prompt.

- [ ] **Step 3: Commit**

```bash
git add validation/prompts.py
git commit -m "Simplify prompts to single ring+7seg-hex prompt"
```

---

## Task 9: Update `extraction.py`

**Files:**
- Modify: `validation/extraction.py`
- Modify: `tests/test_extraction.py`

- [ ] **Step 1: Update the tests**

In `tests/test_extraction.py`, make these four changes:

1. Change line `extractor = ClaudeExtractor(client=mock_client, model="claude-opus-4-7", prompt_version="v1")` to use `prompt_version="v2"`.
2. Change line `result = extractor.extract(img_path, set_name="A")` to `result = extractor.extract(img_path)` (drop `set_name`).
3. Change line `assert result.prompt_version == "v1"` to `assert result.prompt_version == "v2"`.
4. Delete the line `assert result.set_name == "A"` entirely.

Result of those changes — the affected portion of the test should look like:

```python
    extractor = ClaudeExtractor(
        client=mock_client, model="claude-opus-4-7", prompt_version="v2"
    )
    result = extractor.extract(img_path)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert any(
        isinstance(block, dict) and block.get("type") == "image"
        for msg in kwargs["messages"]
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
    )
    assert result.model == "claude-opus-4-7"
    assert result.prompt_version == "v2"
    assert len(result.detections) == 1
    assert result.detections[0].tile_id == "0"
```

The other two test functions in the file (`test_parse_extraction_json_returns_tile_list` and `test_parse_extraction_json_strips_code_fences`) need no changes.

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_extraction.py -v`

Expected: failures because `extraction.py` still requires `set_name`.

- [ ] **Step 4: Update `validation/extraction.py`**

Replace the file with:

```python
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Protocol

from validation.prompts import prompt_for_strip
from validation.schemas import DetectedTile, ImageDetection


class Extractor(Protocol):
    def extract(self, image_path: Path) -> ImageDetection: ...


class ClaudeExtractor:
    def __init__(
        self,
        client,
        model: str = "claude-opus-4-7",
        prompt_version: str = "v2",
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_version = prompt_version

    def extract(self, image_path: Path) -> ImageDetection:
        prompt = prompt_for_strip()
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        media_type = _media_type_for(image_path)

        t0 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        )
        latency = time.monotonic() - t0

        raw = response.content[0].text
        detections = parse_extraction_json(raw)
        return ImageDetection(
            image_path=str(image_path),
            model=self.model,
            prompt_version=self.prompt_version,
            detections=detections,
            raw_response=raw,
            latency_seconds=latency,
        )


def parse_extraction_json(raw: str) -> list[DetectedTile]:
    cleaned = _strip_code_fences(raw).strip()
    data = json.loads(cleaned)
    return [
        DetectedTile(
            tile_id=item["tile_id"],
            circle_xy_px=tuple(item["circle_xy_px"]),
            glyph_xy_px=(
                tuple(item["glyph_xy_px"])
                if item.get("glyph_xy_px") is not None
                else None
            ),
            confidence=float(item["confidence"]),
        )
        for item in data
    ]


def _strip_code_fences(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1)
    return raw


def _media_type_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".heic": "image/heic",
    }.get(ext, "image/jpeg")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_extraction.py -v`

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add validation/extraction.py tests/test_extraction.py
git commit -m "Drop set_name from extraction; use single prompt_for_strip"
```

---

## Task 10: Update `scoring.py` to take `StripSpec`

**Files:**
- Modify: `validation/scoring.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Replace `tests/test_scoring.py`**

Replace the entire file with:

```python
import numpy as np

from validation.homography import fit_homography
from validation.schemas import (
    DetectedTile,
    DigitDims,
    GroundTruthTile,
    ImageDetection,
    RingDims,
    ScoringResult,
    StripSpec,
)
from validation.scoring import score_detection


def _make_detection(tile_id: str, px: tuple[float, float]) -> DetectedTile:
    return DetectedTile(
        tile_id=tile_id,
        circle_xy_px=px,
        glyph_xy_px=None,
        confidence=1.0,
    )


def _spec_for(tiles: list[GroundTruthTile]) -> StripSpec:
    """Build a minimal StripSpec wrapping the given tiles. Geometry values
    are the canonical ring+digit dimensions from data/test-strip-spec.yaml,
    but the tests below use only spec.tiles and spec.glyph_offset_mm — they
    don't depend on the surrounding strip layout."""
    return StripSpec(
        units="mm",
        ring=RingDims(outer_diameter_mm=5.00, inner_diameter_mm=1.66),
        digit=DigitDims(width_mm=2.58, height_mm=5.00, stroke_mm=0.86, corner_radius_mm=0.20),
        ring_to_digit_gap_mm=1.72,
        orientation_deg=0.0,
        margin_mm={"left": 5.0, "right": 5.0, "top": 3.0, "bottom": 3.0},
        tile_pitch_mm=11.02,
        tiles=tiles,
    )


def _make_img(detections: list[DetectedTile]) -> ImageDetection:
    return ImageDetection(
        image_path="x", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )


def test_score_perfect_detection_has_zero_mean_error():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert result.mean_error_mm < 1e-6


def test_score_reports_per_point_error_when_detection_is_offset():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
        GroundTruthTile("4", (5.0, 5.0), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("4", (55.0, 55.0)),  # would be mm (5.5, 5.5) — 0.5 mm off
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert "4" in result.per_point_error_mm
    assert 0.2 < result.per_point_error_mm["4"] < 1.0
    assert result.mean_error_mm > 0.0


def test_score_only_uses_tiles_detected_and_in_truth():
    tiles = [
        GroundTruthTile("0", (0, 0), 0),
        GroundTruthTile("1", (10, 0), 0),
        GroundTruthTile("2", (10, 10), 0),
        GroundTruthTile("3", (0, 10), 0),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("99", (0.0, 0.0)),  # bogus detection not in truth
    ]
    result = score_detection(_make_img(detections), _spec_for(tiles))
    assert "99" not in result.per_point_error_mm
```

Key changes from the old test file:
- Helper `_spec_for(tiles)` builds a `StripSpec` wrapping the tile list (the scorer now takes `spec`, not `tiles`).
- Helper `_make_img(detections)` builds `ImageDetection` without `set_name`.
- Tile orientation changed from `90` to `0` (matches the new strip-level orientation, where the glyph axis is +x). The tests in this file don't pass glyph_xy_px in their detections, so glyph correspondences aren't generated and orientation_deg is unused — but using `0` matches the project's actual ground truth.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`

Expected: failures (signature mismatch).

- [ ] **Step 3: Update `validation/scoring.py`**

Replace the entire file with:

```python
from __future__ import annotations

import math

import numpy as np

from validation.homography import apply_homography, fit_homography
from validation.schemas import (
    ImageDetection,
    ScoringResult,
    StripSpec,
)


def score_detection(
    detection: ImageDetection,
    spec: StripSpec,
) -> ScoringResult:
    """Score one image's detections against the strip ground truth.

    Per-tile error is the circle-center reprojection error after fitting a
    homography from detected pixel positions to known mm positions. Glyph
    positions, when reported, contribute additional correspondences offset
    by spec.glyph_offset_mm along each tile's orientation axis to break
    collinearity.
    """
    truth_by_id = {t.tile_id: t for t in spec.tiles}
    matched = [d for d in detection.detections if d.tile_id in truth_by_id]

    glyph_offset = spec.glyph_offset_mm

    corr_px: list[tuple[float, float]] = []
    corr_mm: list[tuple[float, float]] = []
    for d in matched:
        tile = truth_by_id[d.tile_id]
        corr_px.append(d.circle_xy_px)
        corr_mm.append(tile.center_mm)
        if d.glyph_xy_px is not None:
            rad = math.radians(tile.orientation_deg)
            glyph_mm = (
                tile.center_mm[0] + glyph_offset * math.cos(rad),
                tile.center_mm[1] + glyph_offset * math.sin(rad),
            )
            corr_px.append(d.glyph_xy_px)
            corr_mm.append(glyph_mm)

    if len(corr_px) < 4:
        raise ValueError(
            f"need at least 4 correspondences to fit homography; "
            f"got {len(corr_px)} (from {len(matched)} matched detections)"
        )

    pts_px = np.array(corr_px, dtype=float)
    pts_mm = np.array(corr_mm, dtype=float)
    H = fit_homography(pts_px, pts_mm)

    circle_px = np.array([d.circle_xy_px for d in matched], dtype=float)
    circle_mm = np.array(
        [truth_by_id[d.tile_id].center_mm for d in matched], dtype=float
    )
    projected_mm = apply_homography(H, circle_px)

    per_point: dict[str, float] = {}
    errors_mm: list[float] = []
    for d, proj, truth_xy in zip(matched, projected_mm, circle_mm):
        err = float(np.linalg.norm(proj - truth_xy))
        per_point[d.tile_id] = err
        errors_mm.append(err)

    mean_err = float(np.mean(errors_mm)) if errors_mm else 0.0

    H_inv = np.linalg.inv(H)
    reproj_px = apply_homography(H_inv, pts_mm)
    rmse_px = float(np.sqrt(np.mean(np.sum((reproj_px - pts_px) ** 2, axis=1))))

    return ScoringResult(
        image_path=detection.image_path,
        per_point_error_mm=per_point,
        mean_error_mm=mean_err,
        homography_rmse_px=rmse_px,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`

Expected: all scoring tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/scoring.py tests/test_scoring.py
git commit -m "Score against StripSpec; pull glyph_offset_mm from spec"
```

---

## Task 11: Rewrite `synthetic.py` — `render_png`

**Files:**
- Modify: `validation/synthetic.py`
- Modify: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_synthetic.py` with:

```python
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from validation.fiducial_geometry import strip_primitives
from validation.ground_truth import build_strip_geometry, load_strip_spec
from validation.synthetic import render_png


PX_PER_MM = 20  # tests run faster at 20 than at 40


@pytest.fixture
def primitives_and_spec(data_dir):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    geom = build_strip_geometry(spec)
    prims = strip_primitives(geom)
    return prims, spec


def test_render_png_produces_rgb_image(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = Image.open(out)
    assert img.mode == "RGB"


def test_render_png_dimensions_match_strip(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out))
    expected_w = int(round(spec.strip_width_mm * PX_PER_MM))
    expected_h = int(round(spec.strip_height_mm * PX_PER_MM))
    assert img.shape[1] == expected_w
    assert img.shape[0] == expected_h


def test_render_png_ring_is_hollow(tmp_path, primitives_and_spec):
    """A pixel at the very center of a ring should be background (the hole),
    while a pixel halfway between center and outer edge should be inked."""
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out).convert("L"))
    h_px = img.shape[0]

    ring_center_mm = spec.tiles[0].center_mm
    cx_px = int(round(ring_center_mm[0] * PX_PER_MM))
    cy_px = h_px - int(round(ring_center_mm[1] * PX_PER_MM))
    # center pixel: light (background, inside the hole)
    assert img[cy_px, cx_px] > 200
    # halfway between center and outer edge (outer radius 2.5 mm)
    half_radius_mm = 1.25
    edge_x_px = cx_px + int(round(half_radius_mm * PX_PER_MM))
    assert img[cy_px, edge_x_px] < 80


def test_render_png_inks_each_ring(tmp_path, primitives_and_spec):
    """Every tile's ring should leave dark pixels in the image."""
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.png"
    render_png(prims, out, spec, px_per_mm=PX_PER_MM)
    img = np.array(Image.open(out).convert("L"))
    h_px = img.shape[0]
    # Halfway from ring center toward outer edge along +x.
    half_r_mm = 1.25
    for tile in spec.tiles:
        cx, cy = tile.center_mm
        px = int(round((cx + half_r_mm) * PX_PER_MM))
        py = h_px - int(round(cy * PX_PER_MM))
        assert img[py, px] < 80, f"tile {tile.tile_id}: expected dark pixel at ({px},{py})"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: ImportError or signature mismatch.

- [ ] **Step 3: Replace `validation/synthetic.py`**

Replace the file with:

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from validation.fiducial_geometry import Annulus, Polygon
from validation.schemas import StripSpec


_BG_RGB = (240, 240, 240)
_INK_RGB = (20, 20, 20)


def render_png(
    primitives: list[Annulus | Polygon],
    out_path: Path,
    spec: StripSpec,
    px_per_mm: int = 40,
) -> None:
    """Rasterize primitives onto a fresh background and write a PNG.

    Spec is required to size the canvas. Primitives are rendered in order;
    annulus holes and polygon holes are produced by overdraw with background.
    The strip layout guarantees rings and digits do not overlap, so overdraw
    is safe.
    """
    w_px = int(round(spec.strip_width_mm * px_per_mm))
    h_px = int(round(spec.strip_height_mm * px_per_mm))
    img = Image.new("RGB", (w_px, h_px), _BG_RGB)
    draw = ImageDraw.Draw(img)

    for prim in primitives:
        if isinstance(prim, Annulus):
            _draw_annulus(draw, prim, px_per_mm, h_px)
        elif isinstance(prim, Polygon):
            _draw_polygon(draw, prim, px_per_mm, h_px)
        else:
            raise TypeError(f"unsupported primitive type: {type(prim).__name__}")

    img.save(out_path)


def _mm_to_px(x_mm: float, y_mm: float, px_per_mm: int, h_px: int) -> tuple[int, int]:
    """Convert strip-mm coords (y-up origin-bottom-left) to image px (y-down)."""
    return int(round(x_mm * px_per_mm)), h_px - int(round(y_mm * px_per_mm))


def _draw_annulus(draw: ImageDraw.ImageDraw, a: Annulus, px_per_mm: int, h_px: int) -> None:
    cx, cy = a.center_mm
    outer_px = a.outer_radius_mm * px_per_mm
    inner_px = a.inner_radius_mm * px_per_mm
    cx_px, cy_px = _mm_to_px(cx, cy, px_per_mm, h_px)
    # Outer disc in ink, then inner disc in background to cut the hole.
    draw.ellipse(
        (cx_px - outer_px, cy_px - outer_px, cx_px + outer_px, cy_px + outer_px),
        fill=_INK_RGB,
    )
    draw.ellipse(
        (cx_px - inner_px, cy_px - inner_px, cx_px + inner_px, cy_px + inner_px),
        fill=_BG_RGB,
    )


def _draw_polygon(draw: ImageDraw.ImageDraw, p: Polygon, px_per_mm: int, h_px: int) -> None:
    exterior_px = [_mm_to_px(x, y, px_per_mm, h_px) for x, y in p.exterior_mm]
    draw.polygon(exterior_px, fill=_INK_RGB)
    for interior in p.interiors_mm:
        interior_px = [_mm_to_px(x, y, px_per_mm, h_px) for x, y in interior]
        draw.polygon(interior_px, fill=_BG_RGB)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: 4 tests pass.

- [ ] **Step 5: Run the full test suite to confirm the refactor is healed**

Run: `uv run pytest -v`

Expected: all tests pass (`test_extraction`, `test_fiducial_geometry`, `test_ground_truth`, `test_homography`, `test_schemas`, `test_scoring`, `test_synthetic`).

- [ ] **Step 6: Commit**

```bash
git add validation/synthetic.py tests/test_synthetic.py
git commit -m "Rewrite synthetic.render_png to consume geometry primitives"
```

---

## Task 12: Add `render_svg`

**Files:**
- Modify: `validation/synthetic.py`
- Modify: `tests/test_synthetic.py`

- [ ] **Step 1: Add SVG tests**

Append to `tests/test_synthetic.py`:

```python
import xml.etree.ElementTree as ET

from validation.synthetic import render_svg


def test_render_svg_is_well_formed(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    # Parses without error.
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag.endswith("svg")


def test_render_svg_dimensions_in_mm_match_strip(tmp_path, primitives_and_spec):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    tree = ET.parse(out)
    root = tree.getroot()
    width_str = root.attrib["width"]
    height_str = root.attrib["height"]
    assert width_str.endswith("mm")
    assert height_str.endswith("mm")
    assert float(width_str[:-2]) == pytest.approx(spec.strip_width_mm)
    assert float(height_str[:-2]) == pytest.approx(spec.strip_height_mm)


def test_render_svg_contains_one_ring_path_and_one_digit_path_per_tile(
    tmp_path, primitives_and_spec
):
    prims, spec = primitives_and_spec
    out = tmp_path / "strip.svg"
    render_svg(prims, out, spec)
    tree = ET.parse(out)
    root = tree.getroot()
    # Both rings and digits are <path> elements (ring uses two-circle path
    # with evenodd; digit uses traced exterior+interiors).
    ns = {"svg": "http://www.w3.org/2000/svg"}
    paths = root.findall("svg:path", ns) + root.findall("path")
    # 16 tiles × 2 paths each = 32
    assert len(paths) == 2 * len(spec.tiles)


def test_render_svg_and_png_use_same_primitives(tmp_path, primitives_and_spec):
    """Cross-backend smoke test: both backends consume the same primitive
    list and produce non-empty output. The deeper agreement between them
    (e.g. ring positions match) is asserted indirectly by the per-backend
    tests above plus the geometry tests in test_fiducial_geometry.py."""
    prims, spec = primitives_and_spec
    png_out = tmp_path / "strip.png"
    svg_out = tmp_path / "strip.svg"
    render_png(prims, png_out, spec, px_per_mm=PX_PER_MM)
    render_svg(prims, svg_out, spec)

    assert png_out.stat().st_size > 0
    assert svg_out.stat().st_size > 0

    # SVG references the first ring's leftmost-edge x coordinate (rx - outer_radius).
    # rx = 7.5, outer_radius = 2.5 → 5.0. render_svg formats with %.4f → "5.0000".
    svg_text = svg_out.read_text(encoding="utf-8")
    rx = spec.tiles[0].center_mm[0]
    leftmost_edge = rx - spec.ring.outer_diameter_mm / 2.0
    assert f"{leftmost_edge:.4f}" in svg_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: ImportError on `render_svg`.

- [ ] **Step 3: Add `render_svg` to `validation/synthetic.py`**

Append to `validation/synthetic.py`:

```python
def render_svg(
    primitives: list[Annulus | Polygon],
    out_path: Path,
    spec: StripSpec,
) -> None:
    """Emit an SVG document of the strip in mm.

    Width/height attributes are in mm so a slicer or printer can interpret
    the document at physical scale. Annuli render as a path with two arc
    subpaths and fill-rule=evenodd; polygons render as a path with a
    traced exterior and interior subpaths, also evenodd.
    """
    width = spec.strip_width_mm
    height = spec.strip_height_mm
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}mm" height="{height}mm" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    for prim in primitives:
        if isinstance(prim, Annulus):
            parts.append(_svg_annulus(prim, height))
        elif isinstance(prim, Polygon):
            parts.append(_svg_polygon(prim, height))
        else:
            raise TypeError(f"unsupported primitive type: {type(prim).__name__}")

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _svg_y(y_mm: float, height_mm: float) -> float:
    """Flip y so the SVG y-down coordinate matches the strip's y-up convention."""
    return height_mm - y_mm


def _svg_annulus(a: Annulus, height_mm: float) -> str:
    cx = a.center_mm[0]
    cy = _svg_y(a.center_mm[1], height_mm)
    ro = a.outer_radius_mm
    ri = a.inner_radius_mm
    # Two circles via SVG arc commands. evenodd makes the inner circle a hole.
    d = (
        f"M {cx - ro:.4f} {cy:.4f} "
        f"A {ro} {ro} 0 1 0 {cx + ro:.4f} {cy:.4f} "
        f"A {ro} {ro} 0 1 0 {cx - ro:.4f} {cy:.4f} Z "
        f"M {cx - ri:.4f} {cy:.4f} "
        f"A {ri} {ri} 0 1 0 {cx + ri:.4f} {cy:.4f} "
        f"A {ri} {ri} 0 1 0 {cx - ri:.4f} {cy:.4f} Z"
    )
    return f'<path d="{d}" fill="#000000" fill-rule="evenodd"/>'


def _svg_polygon(p: Polygon, height_mm: float) -> str:
    parts: list[str] = []
    parts.append(_svg_subpath(p.exterior_mm, height_mm))
    for interior in p.interiors_mm:
        parts.append(_svg_subpath(interior, height_mm))
    d = " ".join(parts)
    return f'<path d="{d}" fill="#000000" fill-rule="evenodd"/>'


def _svg_subpath(vertices: list[tuple[float, float]], height_mm: float) -> str:
    if not vertices:
        return ""
    cmds: list[str] = []
    x0, y0 = vertices[0]
    cmds.append(f"M {x0:.4f} {_svg_y(y0, height_mm):.4f}")
    for x, y in vertices[1:]:
        cmds.append(f"L {x:.4f} {_svg_y(y, height_mm):.4f}")
    cmds.append("Z")
    return " ".join(cmds)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: 8 tests pass (4 PNG + 4 SVG).

- [ ] **Step 5: Commit**

```bash
git add validation/synthetic.py tests/test_synthetic.py
git commit -m "Add render_svg with shared geometry primitives"
```

---

## Task 13: Update `cli.py`

**Files:**
- Modify: `validation/cli.py`

- [ ] **Step 1: Replace `validation/cli.py`**

Replace the file with:

```python
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic

from validation.extraction import ClaudeExtractor
from validation.ground_truth import load_strip_spec
from validation.schemas import ImageDetection, ScoringResult
from validation.scoring import score_detection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM fiducial extraction on a batch of images"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--photos", type=Path, required=True,
        help="directory of images; every image is treated as a photo of the canonical strip",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--prompt-version", default="v2")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_strip_spec(args.spec)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    extractor = ClaudeExtractor(
        client=client, model=args.model, prompt_version=args.prompt_version
    )

    rows: list[dict] = []
    for img_path in sorted(args.photos.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".heic"}:
            continue

        print(f"extracting {img_path.name}...")
        detection = extractor.extract(img_path)
        _dump_detection(args.out, detection)

        try:
            result = score_detection(detection, spec)
            rows.append(_flatten_result(detection, result))
            print(
                f"  {len(detection.detections)} tiles detected, "
                f"mean error {result.mean_error_mm:.3f} mm"
            )
        except ValueError as err:
            print(f"  scoring failed: {err}")
            rows.append({
                "image": img_path.name,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "n_detected": len(detection.detections),
                "mean_error_mm": None,
                "error_message": str(err),
            })

    _write_results_csv(args.out / "results.csv", rows)
    print(f"\nresults written to {args.out / 'results.csv'}")


def _dump_detection(out_dir: Path, detection: ImageDetection) -> None:
    stem = Path(detection.image_path).stem
    with (out_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(detection), f, indent=2)


def _flatten_result(d: ImageDetection, r: ScoringResult) -> dict:
    return {
        "image": Path(d.image_path).name,
        "model": d.model,
        "prompt_version": d.prompt_version,
        "n_detected": len(d.detections),
        "mean_error_mm": round(r.mean_error_mm, 4),
        "homography_rmse_px": round(r.homography_rmse_px, 3),
        "latency_seconds": round(d.latency_seconds, 2),
    }


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from validation.cli import main; print('ok')"`

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add validation/cli.py
git commit -m "Drop set-name parsing from CLI; pass spec to scorer"
```

---

## Task 14: Update `generate_synthetic.py` and produce strip artifacts

**Files:**
- Modify: `scripts/generate_synthetic.py`

- [ ] **Step 1: Replace `scripts/generate_synthetic.py`**

Replace the file with:

```python
from pathlib import Path

from validation.fiducial_geometry import strip_primitives
from validation.ground_truth import build_strip_geometry, load_strip_spec
from validation.synthetic import render_png, render_svg


def main() -> None:
    spec = load_strip_spec(Path("data/test-strip-spec.yaml"))
    geom = build_strip_geometry(spec)
    prims = strip_primitives(geom)

    out_dir = Path("data/photos")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "strip.png"
    svg_path = out_dir / "strip.svg"

    render_png(prims, png_path, spec, px_per_mm=40)
    render_svg(prims, svg_path, spec)

    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `uv run python scripts/generate_synthetic.py`

Expected output:
```
wrote data/photos/strip.png
wrote data/photos/strip.svg
```

- [ ] **Step 3: Eyeball the output**

Open `data/photos/strip.png` in an image viewer. Visually verify:
- 16 tiles in a single row
- Each tile is a hollow ring + a 7-segment-style digit to its right
- The alphabet reads `0 1 2 3 4 5 6 7 8 9 A b C d E F` left to right
- Outer corners of digits look slightly rounded

Open `data/photos/strip.svg` in an SVG viewer (e.g., a web browser). Verify the same content is visible.

- [ ] **Step 4: Commit the regenerated artifacts**

```bash
git add scripts/generate_synthetic.py data/photos/strip.png data/photos/strip.svg
git commit -m "Regenerate synthetic strip with new ring + 7-seg-hex design"
```

---

## Task 15: Cleanup stale photos and update fiducial-design.md

**Files:**
- Delete: `data/photos/A_top_day_01.png`
- Delete: `data/photos/B_top_day_01.png`
- Modify: `docs/fiducial-design.md`

- [ ] **Step 1: Delete the stale renders**

Run:
```bash
git rm data/photos/A_top_day_01.png data/photos/B_top_day_01.png
```

- [ ] **Step 2: Update `docs/fiducial-design.md`**

Replace the `## Status` section with:

```markdown
## Status

**2026-04-25** — design locked in. Ring + 7-segment hex glyph at 5 mm tile size, 16-tile strip. See `docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md` for the full numeric spec, including outline-level corner rounding. The earlier "preliminary design at 8 mm with TrueType digit" parameter table below is superseded by that spec.

**2026-04-22** — preliminary design at concept level (superseded).
```

Append a new entry at the top of the `## Decision log` section:

```markdown
- **2026-04-25** — Iterated the fiducial: filled circle → ring (annulus); TrueType digit → 7-segment-hex with outline-level rounded corners; multi-set comparison strip (A/B/C) → single canonical strip at the 5 mm size. Drove the change by promoting the SVG output to a print artifact, which made print-print-correct rendering load-bearing. Full spec at `docs/superpowers/specs/2026-04-25-ring-glyph-fiducial-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/fiducial-design.md
git commit -m "Pin ring+7seg-hex design status; log decision; remove stale photos"
```

---

## Final verification

- [ ] **Run the full test suite**

Run: `uv run pytest -v`

Expected: all tests pass (no skips, no failures).

- [ ] **Inspect the generated strip artifacts**

Open `data/photos/strip.png` and `data/photos/strip.svg`. Confirm visually that:
- Strip dimensions are roughly 184.6 × 11.0 mm (the SVG header shows this in mm).
- 16 hollow rings with 7-segment hex digits to the right.
- Alphabet reads `0 1 2 3 4 5 6 7 8 9 A b C d E F`.
- Digit corners visibly rounded but not so rounded that the 7-segment look is lost.

- [ ] **Confirm git status is clean**

Run: `git status`

Expected: "nothing to commit, working tree clean."
