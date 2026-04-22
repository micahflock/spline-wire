# LLM CV Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python validation harness that ingests phone photos of a printed fiducial test strip, extracts per-tile positions via a multimodal LLM, rectifies to chain-plane mm via homography, and scores positional error against a ground-truth spec. Pass criterion: mean positional error < 1 mm on at least one (image set × extraction path) combination.

**Architecture:** A single Python package `validation/` with a clear data flow — `test-strip-spec.yaml` defines ground truth; `synthetic.py` renders fake test images for pre-real-photos iteration; `extraction.py` drives a pluggable LLM backend; `homography.py` fits the pixel → chain-plane transform from known reference tiles; `scoring.py` computes positional error; `cli.py` runs the full batch and emits a results table. Pure-math modules (homography, scoring, schemas) are TDD-first; LLM calls are tested via mocked responses. Real API calls happen only in the final validation task.

**Tech Stack:** Python 3.12, uv for dependency management, OpenCV for homography, Pillow for synthetic image rendering, Anthropic Python SDK for LLM extraction, pytest for tests, PyYAML for the ground-truth spec, dataclasses for schemas.

**Design notes for the executor:**
- If the `superpowers:claude-api` skill is available, use it during Task 7 (LLM extractor). The extractor should use prompt caching because the instruction prompt is stable across images and we'll send 54+ images per run. It should set the model to `claude-opus-4-7` by default but accept a model-name override.
- Everything under `data/photos/` and `data/results/` is gitignored. Only the spec and code get committed.
- The 10 tasks form four chunks: setup (T1), pure-math core (T2–T6), LLM layer (T7–T8), validation runs (T9–T10). Chunks 2 and 3 are independent — T7 does not depend on T5/T6 being done.

---

## File structure

**Created by this plan:**

- `pyproject.toml` — uv project manifest.
- `validation/__init__.py` — package marker.
- `validation/schemas.py` — dataclasses for ground truth, detections, results.
- `validation/ground_truth.py` — YAML loader for the test-strip spec.
- `validation/synthetic.py` — renders a synthetic photo-like image from a ground-truth spec.
- `validation/homography.py` — fits pixel → mm transform; applies it to detected points.
- `validation/scoring.py` — computes per-point and mean error against ground truth.
- `validation/prompts.py` — prompt templates (versioned).
- `validation/extraction.py` — LLM extractor interface + Claude backend.
- `validation/cli.py` — batch runner entry point.
- `data/test-strip-spec.yaml` — canonical ground-truth spec for all three tile sets.
- `tests/conftest.py` — shared pytest fixtures.
- `tests/test_schemas.py`, `tests/test_ground_truth.py`, `tests/test_synthetic.py`, `tests/test_homography.py`, `tests/test_scoring.py`, `tests/test_extraction.py` — unit tests.
- `.gitignore` — add `data/photos/`, `data/results/`, `.venv/`, `__pycache__/`, `.pytest_cache/`.

**Modified:**

- `.gitignore` — add new ignores.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `validation/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Initialize uv project**

Run: `uv init --package --name spline-wire-validation --no-readme`

This creates `pyproject.toml` and a `src/` layout. Delete the generated `src/` — we're using a flat `validation/` layout.

Run: `rm -rf src`

- [ ] **Step 2: Add dependencies**

Run:
```bash
uv add anthropic opencv-python numpy pillow pyyaml
uv add --dev pytest pytest-mock
```

Expected: `pyproject.toml` is updated, `.venv/` is created, `uv.lock` is written.

- [ ] **Step 3: Replace pyproject.toml package config**

Edit `pyproject.toml` — replace whatever uv-init generated for `[tool.hatch.build.targets.wheel]` or `[project]` package discovery with:

```toml
[tool.hatch.build.targets.wheel]
packages = ["validation"]
```

And under `[project]`, ensure:

```toml
name = "spline-wire-validation"
requires-python = ">=3.12"
```

- [ ] **Step 4: Create package skeleton**

Create `validation/__init__.py` (empty file).
Create `tests/__init__.py` (empty file).
Create `tests/conftest.py`:

```python
from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(repo_root: Path) -> Path:
    return repo_root / "data"
```

- [ ] **Step 5: Update .gitignore**

Append to `.gitignore`:

```
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
data/photos/
data/results/
```

- [ ] **Step 6: Verify the package imports**

Run: `uv run python -c "import validation; import anthropic; import cv2; import numpy; import PIL; import yaml; print('ok')"`

Expected: `ok`.

- [ ] **Step 7: Verify pytest runs (no tests yet)**

Run: `uv run pytest`

Expected: exit code 5 (no tests collected) — this confirms pytest is wired up.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock validation/ tests/ .gitignore
git commit -m "Scaffold validation package with uv and pytest"
```

---

## Task 2: Dataclass schemas

**Files:**
- Create: `validation/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for schemas**

Create `tests/test_schemas.py`:

```python
from validation.schemas import (
    FiducialType,
    GroundTruthTile,
    TileSet,
    StripSpec,
    DetectedTile,
    ImageDetection,
    ScoringResult,
)


def test_ground_truth_tile_holds_position_and_id():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    assert tile.tile_id == "0"
    assert tile.center_mm == (10.0, 10.0)
    assert tile.orientation_deg == 90.0


def test_tile_set_has_name_type_and_tiles():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    ts = TileSet(
        name="A",
        fiducial_type=FiducialType.CIRCLE_GLYPH,
        tile_face_mm=8.0,
        tile_pitch_mm=10.0,
        tiles=[tile],
    )
    assert ts.name == "A"
    assert ts.fiducial_type is FiducialType.CIRCLE_GLYPH
    assert len(ts.tiles) == 1


def test_strip_spec_indexes_sets_by_name():
    tile = GroundTruthTile(tile_id="0", center_mm=(10.0, 10.0), orientation_deg=90.0)
    ts = TileSet(
        name="A",
        fiducial_type=FiducialType.CIRCLE_GLYPH,
        tile_face_mm=8.0,
        tile_pitch_mm=10.0,
        tiles=[tile],
    )
    spec = StripSpec(units="mm", sets={"A": ts})
    assert spec.sets["A"].tiles[0].tile_id == "0"


def test_detected_tile_stores_pixel_positions_and_confidence():
    d = DetectedTile(
        tile_id="3",
        circle_xy_px=(120.5, 200.0),
        glyph_xy_px=(160.0, 200.0),
        confidence=0.92,
    )
    assert d.tile_id == "3"
    assert d.confidence == 0.92


def test_image_detection_carries_metadata():
    det = DetectedTile(
        tile_id="0",
        circle_xy_px=(10.0, 10.0),
        glyph_xy_px=(20.0, 10.0),
        confidence=1.0,
    )
    img = ImageDetection(
        image_path="data/photos/a_top_day_01.jpg",
        set_name="A",
        model="claude-opus-4-7",
        prompt_version="v1",
        detections=[det],
        raw_response="...",
        latency_seconds=3.1,
    )
    assert img.set_name == "A"
    assert img.model == "claude-opus-4-7"
    assert img.detections[0].tile_id == "0"


def test_scoring_result_captures_mean_and_per_point_errors():
    r = ScoringResult(
        image_path="data/photos/a_top_day_01.jpg",
        per_point_error_mm={"0": 0.4, "1": 0.6},
        mean_error_mm=0.5,
        homography_rmse_px=1.2,
    )
    assert r.mean_error_mm == 0.5
    assert r.per_point_error_mm["1"] == 0.6
```

- [ ] **Step 2: Run the test and watch it fail**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'validation.schemas'`.

- [ ] **Step 3: Implement schemas**

Create `validation/schemas.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FiducialType(str, Enum):
    CIRCLE_GLYPH = "circle_glyph"
    ARUCO_4X4 = "aruco_4x4"


@dataclass(frozen=True)
class GroundTruthTile:
    tile_id: str
    center_mm: tuple[float, float]
    orientation_deg: float


@dataclass(frozen=True)
class TileSet:
    name: str
    fiducial_type: FiducialType
    tile_face_mm: float
    tile_pitch_mm: float
    tiles: list[GroundTruthTile]


@dataclass(frozen=True)
class StripSpec:
    units: str
    sets: dict[str, TileSet]


@dataclass(frozen=True)
class DetectedTile:
    tile_id: str
    circle_xy_px: tuple[float, float]
    glyph_xy_px: tuple[float, float] | None
    confidence: float


@dataclass
class ImageDetection:
    image_path: str
    set_name: str
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

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/schemas.py tests/test_schemas.py
git commit -m "Add dataclass schemas for ground truth, detections, and scoring"
```

---

## Task 3: Ground-truth spec and loader

**Files:**
- Create: `data/test-strip-spec.yaml`
- Create: `validation/ground_truth.py`
- Create: `tests/test_ground_truth.py`

- [ ] **Step 1: Write the test-strip spec**

Create `data/test-strip-spec.yaml`:

```yaml
# Canonical ground-truth specification for the printed test strip.
# All positions are in millimeters, measured on the planar strip surface.
# Origin is the lower-left corner of the strip.
# Tile centers are evenly spaced along the strip x-axis.
# Each tile's orientation is the direction of the circle→glyph axis,
# in degrees, measured counterclockwise from the +x axis of the strip.

units: mm

sets:
  A:
    fiducial_type: circle_glyph
    tile_face_mm: 8.0
    tile_pitch_mm: 10.0
    origin_mm: [15.0, 20.0]   # center of tile 0
    count: 10
    orientation_deg: 90.0     # circle→glyph axis perpendicular to strip
    glyph_alphabet: "0123456789"

  B:
    fiducial_type: circle_glyph
    tile_face_mm: 5.0
    tile_pitch_mm: 7.0
    origin_mm: [15.0, 40.0]
    count: 10
    orientation_deg: 90.0
    glyph_alphabet: "0123456789"

  C:
    fiducial_type: aruco_4x4
    tile_face_mm: 8.0
    tile_pitch_mm: 10.0
    origin_mm: [15.0, 60.0]
    count: 10
    orientation_deg: 0.0      # ArUco has no orientation axis in this sense
    aruco_dict: DICT_4X4_50
    aruco_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

- [ ] **Step 2: Write failing tests for the loader**

Create `tests/test_ground_truth.py`:

```python
from pathlib import Path

import pytest

from validation.ground_truth import load_strip_spec
from validation.schemas import FiducialType


def test_load_strip_spec_parses_three_sets(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    assert spec.units == "mm"
    assert set(spec.sets.keys()) == {"A", "B", "C"}


def test_set_a_generates_10_tiles_with_correct_pitch(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    assert len(set_a.tiles) == 10
    assert set_a.tiles[0].center_mm == (15.0, 20.0)
    assert set_a.tiles[1].center_mm == (25.0, 20.0)
    assert set_a.tiles[9].center_mm == (105.0, 20.0)
    assert set_a.fiducial_type is FiducialType.CIRCLE_GLYPH


def test_set_a_tile_ids_match_digit_alphabet(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    ids = [t.tile_id for t in set_a.tiles]
    assert ids == list("0123456789")


def test_set_c_uses_aruco_type(data_dir: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_c = spec.sets["C"]
    assert set_c.fiducial_type is FiducialType.ARUCO_4X4


def test_load_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_strip_spec(tmp_path / "does-not-exist.yaml")
```

- [ ] **Step 3: Run tests and watch them fail**

Run: `uv run pytest tests/test_ground_truth.py -v`

Expected: `ModuleNotFoundError: No module named 'validation.ground_truth'`.

- [ ] **Step 4: Implement the loader**

Create `validation/ground_truth.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from validation.schemas import FiducialType, GroundTruthTile, StripSpec, TileSet


def load_strip_spec(path: Path) -> StripSpec:
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    sets: dict[str, TileSet] = {}
    for name, entry in raw["sets"].items():
        sets[name] = _build_tile_set(name, entry)
    return StripSpec(units=raw["units"], sets=sets)


def _build_tile_set(name: str, entry: dict) -> TileSet:
    fid_type = FiducialType(entry["fiducial_type"])
    origin_x, origin_y = entry["origin_mm"]
    pitch = entry["tile_pitch_mm"]
    count = entry["count"]
    orientation = entry["orientation_deg"]

    if fid_type is FiducialType.CIRCLE_GLYPH:
        alphabet = entry["glyph_alphabet"]
        assert len(alphabet) >= count, (
            f"Set {name}: glyph alphabet shorter than count"
        )
        ids = list(alphabet[:count])
    else:
        ids = [str(i) for i in entry["aruco_ids"]]

    tiles = [
        GroundTruthTile(
            tile_id=ids[i],
            center_mm=(origin_x + i * pitch, origin_y),
            orientation_deg=orientation,
        )
        for i in range(count)
    ]
    return TileSet(
        name=name,
        fiducial_type=fid_type,
        tile_face_mm=entry["tile_face_mm"],
        tile_pitch_mm=pitch,
        tiles=tiles,
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_ground_truth.py -v`

Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/test-strip-spec.yaml validation/ground_truth.py tests/test_ground_truth.py
git commit -m "Add test strip spec and ground-truth loader"
```

---

## Task 4: Synthetic image renderer

**Files:**
- Create: `validation/synthetic.py`
- Create: `tests/test_synthetic.py`

**Purpose:** Render a photo-like image of a tile set from the ground-truth spec, so the whole pipeline can be validated before real phone photos exist. The synthetic image is what a perfect top-down photo would look like.

- [ ] **Step 1: Write failing tests**

Create `tests/test_synthetic.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from validation.ground_truth import load_strip_spec
from validation.synthetic import render_set


def test_render_set_produces_rgb_image(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    out = tmp_path / "set_a.png"
    render_set(spec.sets["A"], out, px_per_mm=20)
    assert out.exists()
    img = Image.open(out)
    assert img.mode == "RGB"


def test_render_set_is_wide_enough_for_all_tiles(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out))
    # strip width must cover last tile's x + half pitch, with 20 px/mm
    min_width_px = int((set_a.tiles[-1].center_mm[0] + set_a.tile_pitch_mm) * 20)
    assert img.shape[1] >= min_width_px


def test_render_set_places_dark_circle_at_each_tile_center(
    data_dir: Path, tmp_path: Path
):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out).convert("L"))
    # at the pixel of the first tile's circle center, expect a dark pixel
    # circle is offset from tile center toward -y in the rendered orientation
    # (circle→glyph axis = 90deg means glyph is +y of circle)
    # so circle sits at tile_center with an offset of -1.5mm in y
    tile0 = set_a.tiles[0]
    circle_x_mm = tile0.center_mm[0]
    circle_y_mm = tile0.center_mm[1] - 1.5
    px = int(circle_x_mm * 20)
    py = int(circle_y_mm * 20)
    # image y grows downward, spec y grows upward — synthetic must handle this
    # we expect a *low* pixel value (dark ink on light background)
    assert img[img.shape[0] - py, px] < 80


def test_render_set_draws_all_tiles(data_dir: Path, tmp_path: Path):
    spec = load_strip_spec(data_dir / "test-strip-spec.yaml")
    set_a = spec.sets["A"]
    out = tmp_path / "set_a.png"
    render_set(set_a, out, px_per_mm=20)
    img = np.array(Image.open(out).convert("L"))
    # sum dark pixels — should have roughly (circle area + glyph strokes) × 10 tiles
    dark = (img < 80).sum()
    assert dark > 1000
```

- [ ] **Step 2: Run tests — they should fail**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the renderer**

Create `validation/synthetic.py`:

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from validation.schemas import FiducialType, TileSet

_BG = (240, 240, 240)
_INK = (20, 20, 20)


def render_set(tile_set: TileSet, out_path: Path, px_per_mm: int = 20) -> None:
    last = tile_set.tiles[-1]
    width_mm = last.center_mm[0] + tile_set.tile_pitch_mm
    height_mm = last.center_mm[1] + tile_set.tile_pitch_mm
    w_px = int(width_mm * px_per_mm)
    h_px = int(height_mm * px_per_mm)

    img = Image.new("RGB", (w_px, h_px), _BG)
    draw = ImageDraw.Draw(img)

    for tile in tile_set.tiles:
        _draw_tile(draw, tile_set, tile, px_per_mm, h_px)

    img.save(out_path)


def _draw_tile(draw, tile_set, tile, px_per_mm, h_px):
    cx_mm, cy_mm = tile.center_mm
    if tile_set.fiducial_type is FiducialType.CIRCLE_GLYPH:
        _draw_circle_glyph(draw, cx_mm, cy_mm, tile, px_per_mm, h_px)
    elif tile_set.fiducial_type is FiducialType.ARUCO_4X4:
        # synthetic ArUco: a simple binary pattern placeholder — the real
        # validation run uses actual cv2.aruco.generateImageMarker patterns.
        # For synthetic-pipeline testing, a solid dark square is enough to
        # confirm the pipeline moves data through; ArUco decoding is tested
        # separately against real photos.
        _draw_aruco_placeholder(draw, cx_mm, cy_mm, tile_set, px_per_mm, h_px)


def _mm_to_px(x_mm: float, y_mm: float, px_per_mm: int, h_px: int) -> tuple[int, int]:
    """Convert strip-mm coords (y-up origin-bottom-left) to image px (y-down)."""
    return int(x_mm * px_per_mm), h_px - int(y_mm * px_per_mm)


def _draw_circle_glyph(draw, cx_mm, cy_mm, tile, px_per_mm, h_px):
    # circle at -1.5mm y from tile center, glyph at +1.5mm y
    # (matches orientation_deg=90.0 axis)
    circle_mm = (cx_mm, cy_mm - 1.5)
    glyph_mm = (cx_mm, cy_mm + 1.5)
    radius_px = int(1.5 * px_per_mm)

    cx, cy = _mm_to_px(*circle_mm, px_per_mm, h_px)
    draw.ellipse(
        (cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px),
        fill=_INK,
    )

    gx, gy = _mm_to_px(*glyph_mm, px_per_mm, h_px)
    font = _load_font(px_per_mm)
    draw.text((gx, gy), tile.tile_id, fill=_INK, anchor="mm", font=font)


def _draw_aruco_placeholder(draw, cx_mm, cy_mm, tile_set, px_per_mm, h_px):
    half = tile_set.tile_face_mm / 2
    x0, y0 = _mm_to_px(cx_mm - half, cy_mm + half, px_per_mm, h_px)
    x1, y1 = _mm_to_px(cx_mm + half, cy_mm - half, px_per_mm, h_px)
    draw.rectangle((x0, y0, x1, y1), outline=_INK, width=2)


def _load_font(px_per_mm: int) -> ImageFont.ImageFont:
    # 3mm glyph height target → ~3 * px_per_mm font size
    size = max(8, int(3 * px_per_mm))
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_synthetic.py -v`

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add validation/synthetic.py tests/test_synthetic.py
git commit -m "Add synthetic image renderer for pipeline validation"
```

---

## Task 5: Homography estimation

**Files:**
- Create: `validation/homography.py`
- Create: `tests/test_homography.py`

**Purpose:** Given detected pixel coordinates for reference tiles whose true mm positions are known, fit a 3×3 homography mapping pixels → mm, and apply it to arbitrary points.

- [ ] **Step 1: Write failing tests**

Create `tests/test_homography.py`:

```python
import numpy as np
import pytest

from validation.homography import fit_homography, apply_homography


def test_identity_pixel_to_mm_with_known_scale():
    """10 px/mm uniform scale, no rotation, no translation."""
    pts_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    pts_mm = pts_px / 10.0
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[50, 50]], dtype=float))
    np.testing.assert_allclose(out, [[5.0, 5.0]], atol=1e-6)


def test_rotated_90deg_mapping():
    """90deg rotation: pixel (1,0) maps to mm (0,1)."""
    pts_px = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    pts_mm = np.array([[0, 0], [0, 1], [-1, 1], [-1, 0]], dtype=float)
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[0.5, 0.0]], dtype=float))
    np.testing.assert_allclose(out, [[0.0, 0.5]], atol=1e-6)


def test_requires_at_least_four_correspondences():
    pts_px = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    pts_mm = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
    with pytest.raises(ValueError):
        fit_homography(pts_px, pts_mm)


def test_apply_handles_multiple_points():
    pts_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    pts_mm = pts_px / 10.0
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, np.array([[0, 0], [50, 50], [100, 100]], dtype=float))
    np.testing.assert_allclose(
        out, [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]], atol=1e-6
    )


def test_perspective_mapping_roundtrips():
    """Construct a known perspective, verify fit recovers it."""
    # quadrilateral in pixels, square in mm
    pts_px = np.array(
        [[10, 10], [200, 50], [210, 190], [20, 210]], dtype=float
    )
    pts_mm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    H = fit_homography(pts_px, pts_mm)
    out = apply_homography(H, pts_px)
    np.testing.assert_allclose(out, pts_mm, atol=1e-6)
```

- [ ] **Step 2: Run tests — fail**

Run: `uv run pytest tests/test_homography.py -v`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement homography**

Create `validation/homography.py`:

```python
from __future__ import annotations

import cv2
import numpy as np


def fit_homography(pts_px: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    """Fit 3x3 homography mapping pixel coordinates to mm coordinates.

    Both inputs must be Nx2 arrays of corresponding points, N >= 4.
    """
    if pts_px.shape[0] < 4 or pts_mm.shape[0] < 4:
        raise ValueError("need at least 4 correspondences")
    if pts_px.shape != pts_mm.shape:
        raise ValueError("input shape mismatch")
    H, _mask = cv2.findHomography(pts_px, pts_mm, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography returned None")
    return H


def apply_homography(H: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Apply homography to Nx2 pixel points, return Nx2 mm points."""
    if pts_px.ndim != 2 or pts_px.shape[1] != 2:
        raise ValueError("pts_px must be shape (N, 2)")
    reshaped = pts_px.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(reshaped, H)
    return out.reshape(-1, 2)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_homography.py -v`

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add validation/homography.py tests/test_homography.py
git commit -m "Add homography fit and apply with OpenCV"
```

---

## Task 6: Scoring metrics

**Files:**
- Create: `validation/scoring.py`
- Create: `tests/test_scoring.py`

**Purpose:** Given detections + ground truth + a fitted homography, compute per-point error and mean error in mm. Also compute the homography's own reprojection RMSE so we can tell "bad detection" apart from "bad homography."

- [ ] **Step 1: Write failing tests**

Create `tests/test_scoring.py`:

```python
import numpy as np

from validation.homography import fit_homography
from validation.schemas import (
    DetectedTile,
    GroundTruthTile,
    ImageDetection,
    ScoringResult,
)
from validation.scoring import score_detection


def _make_detection(tile_id: str, px: tuple[float, float]) -> DetectedTile:
    return DetectedTile(
        tile_id=tile_id,
        circle_xy_px=px,
        glyph_xy_px=None,
        confidence=1.0,
    )


def test_score_perfect_detection_has_zero_mean_error():
    # build 4-point identity mapping at 10 px/mm
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
    ]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles)
    assert result.mean_error_mm < 1e-6


def test_score_reports_per_point_error_when_detection_is_offset():
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    # perfect first 4, then a 5th tile detected 0.5mm off
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("4", (55.0, 55.0)),  # would be mm (5.5, 5.5)
    ]
    tiles_with_5th = tiles + [GroundTruthTile("4", (5.0, 5.0), 90)]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles_with_5th)
    # tile 4: detected at (5.5, 5.5), truth (5.0, 5.0), error = sqrt(0.5)
    assert abs(result.per_point_error_mm["4"] - np.sqrt(0.5)) < 1e-6


def test_score_only_uses_tiles_detected_and_in_truth():
    tiles = [
        GroundTruthTile("0", (0, 0), 90),
        GroundTruthTile("1", (10, 0), 90),
        GroundTruthTile("2", (10, 10), 90),
        GroundTruthTile("3", (0, 10), 90),
    ]
    detections = [
        _make_detection("0", (0.0, 0.0)),
        _make_detection("1", (100.0, 0.0)),
        _make_detection("2", (100.0, 100.0)),
        _make_detection("3", (0.0, 100.0)),
        _make_detection("99", (0.0, 0.0)),  # bogus detection not in truth
    ]
    img = ImageDetection(
        image_path="x", set_name="A", model="m", prompt_version="v",
        detections=detections, raw_response="", latency_seconds=0.0,
    )
    result = score_detection(img, tiles)
    assert "99" not in result.per_point_error_mm
```

- [ ] **Step 2: Run tests — fail**

Run: `uv run pytest tests/test_scoring.py -v`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement scoring**

Create `validation/scoring.py`:

```python
from __future__ import annotations

import numpy as np

from validation.homography import apply_homography, fit_homography
from validation.schemas import (
    GroundTruthTile,
    ImageDetection,
    ScoringResult,
)


def score_detection(
    detection: ImageDetection,
    ground_truth: list[GroundTruthTile],
) -> ScoringResult:
    truth_by_id = {t.tile_id: t for t in ground_truth}
    matched = [d for d in detection.detections if d.tile_id in truth_by_id]
    if len(matched) < 4:
        raise ValueError(
            f"need at least 4 matched detections to fit homography; "
            f"got {len(matched)}"
        )

    pts_px = np.array([d.circle_xy_px for d in matched], dtype=float)
    pts_mm = np.array(
        [truth_by_id[d.tile_id].center_mm for d in matched], dtype=float
    )
    H = fit_homography(pts_px, pts_mm)

    projected_mm = apply_homography(H, pts_px)

    per_point: dict[str, float] = {}
    errors_mm: list[float] = []
    for d, proj in zip(matched, projected_mm):
        truth_xy = np.array(truth_by_id[d.tile_id].center_mm)
        err = float(np.linalg.norm(proj - truth_xy))
        per_point[d.tile_id] = err
        errors_mm.append(err)

    mean_err = float(np.mean(errors_mm)) if errors_mm else 0.0

    # homography reprojection RMSE in px: project truth-mm back to pixels
    # via inverse H and compare to detected pixels
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

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_scoring.py -v`

Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add validation/scoring.py tests/test_scoring.py
git commit -m "Add scoring with per-point and mean positional error"
```

---

## Task 7: LLM extractor with Claude backend

**Files:**
- Create: `validation/prompts.py`
- Create: `validation/extraction.py`
- Create: `tests/test_extraction.py`

**Purpose:** Send an image to a multimodal LLM and parse its reply into `DetectedTile` objects. The interface is model-agnostic; the first backend is Anthropic Claude.

**Notes for the executor:**
- If the `superpowers:claude-api` skill is available, invoke it — it knows the current SDK idioms, prompt caching, and model IDs. Default model: `claude-opus-4-7`.
- Use prompt caching on the stable instruction portion of the prompt. The image is the per-call portion.
- The LLM call itself is mocked in the unit test. Task 9 covers the real integration run.

- [ ] **Step 1: Write failing tests**

Create `tests/test_extraction.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock

from validation.extraction import ClaudeExtractor, parse_extraction_json
from validation.schemas import DetectedTile


def test_parse_extraction_json_returns_tile_list():
    raw = json.dumps([
        {"tile_id": "0", "circle_xy_px": [120.5, 200.0],
         "glyph_xy_px": [160.0, 200.0], "confidence": 0.92},
        {"tile_id": "1", "circle_xy_px": [220.0, 200.0],
         "glyph_xy_px": [260.0, 200.0], "confidence": 0.88},
    ])
    tiles = parse_extraction_json(raw)
    assert len(tiles) == 2
    assert tiles[0].tile_id == "0"
    assert tiles[0].circle_xy_px == (120.5, 200.0)
    assert tiles[1].confidence == 0.88


def test_parse_extraction_json_strips_code_fences():
    raw = "```json\n" + json.dumps([
        {"tile_id": "0", "circle_xy_px": [1, 2],
         "glyph_xy_px": None, "confidence": 1.0},
    ]) + "\n```"
    tiles = parse_extraction_json(raw)
    assert tiles[0].glyph_xy_px is None


def test_claude_extractor_builds_request_with_image_and_prompt(tmp_path: Path):
    # prepare a 1x1 PNG
    from PIL import Image
    img_path = tmp_path / "test.png"
    Image.new("RGB", (1, 1), (255, 255, 255)).save(img_path)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps([
        {"tile_id": "0", "circle_xy_px": [0, 0],
         "glyph_xy_px": None, "confidence": 1.0}
    ]))]
    mock_client.messages.create.return_value = mock_response

    extractor = ClaudeExtractor(
        client=mock_client, model="claude-opus-4-7", prompt_version="v1"
    )
    result = extractor.extract(img_path, set_name="A")

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert any(
        isinstance(block, dict) and block.get("type") == "image"
        for msg in kwargs["messages"]
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
    )
    assert result.model == "claude-opus-4-7"
    assert result.prompt_version == "v1"
    assert result.set_name == "A"
    assert len(result.detections) == 1
    assert result.detections[0].tile_id == "0"
```

- [ ] **Step 2: Run tests — fail**

Run: `uv run pytest tests/test_extraction.py -v`

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement prompt templates**

Create `validation/prompts.py`:

```python
from __future__ import annotations

PROMPT_V1_CIRCLE_GLYPH = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of fiducial tiles.

Each tile has TWO features:
1. A solid dark FILLED CIRCLE (the position reference).
2. A DIGIT from 0-9 printed next to the circle (the ID reference).

Within each tile, the circle and digit are aligned along an axis \
perpendicular to the strip's long direction. Treat the circle's center \
as the tile's position. Treat the digit as the tile's ID.

For EVERY tile visible in the image, return a JSON array with one object \
per tile:

[
  {
    "tile_id": "<digit as string, e.g. '0'>",
    "circle_xy_px": [<circle center x>, <circle center y>],
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

PROMPT_V1_ARUCO = """\
You are looking at a photograph of a planar test strip containing a \
horizontal row of ArUco 4x4 fiducial markers.

For EACH visible ArUco marker, return a JSON array with one object per \
marker:

[
  {
    "tile_id": "<marker id as string, e.g. '0'>",
    "circle_xy_px": [<marker center x>, <marker center y>],
    "glyph_xy_px": null,
    "confidence": <float 0-1>
  },
  ...
]

The marker ID is encoded in the black-and-white grid. Coordinates are \
pixel coordinates in the original image, with (0,0) at top-left. Return \
ONLY the JSON array.\
"""


def prompt_for_set(set_name: str, fiducial_type: str) -> str:
    if fiducial_type == "circle_glyph":
        return PROMPT_V1_CIRCLE_GLYPH
    if fiducial_type == "aruco_4x4":
        return PROMPT_V1_ARUCO
    raise ValueError(f"no prompt for fiducial type {fiducial_type}")
```

- [ ] **Step 4: Implement the extractor**

Create `validation/extraction.py`:

```python
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Protocol

from validation.prompts import prompt_for_set
from validation.schemas import DetectedTile, ImageDetection


class Extractor(Protocol):
    def extract(self, image_path: Path, set_name: str) -> ImageDetection: ...


class ClaudeExtractor:
    def __init__(
        self,
        client,
        model: str = "claude-opus-4-7",
        prompt_version: str = "v1",
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_version = prompt_version

    def extract(self, image_path: Path, set_name: str) -> ImageDetection:
        # TODO when executing: route fiducial_type from the strip spec
        # rather than hard-coding circle_glyph. For set C, pass aruco_4x4.
        prompt = prompt_for_set(set_name, "circle_glyph")
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
            set_name=set_name,
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

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_extraction.py -v`

Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add validation/prompts.py validation/extraction.py tests/test_extraction.py
git commit -m "Add LLM extractor interface with Claude backend and v1 prompt"
```

---

## Task 8: Batch runner and report

**Files:**
- Create: `validation/cli.py`

**Purpose:** End-to-end driver. Reads the strip spec, iterates over a manifest of images, calls the extractor, scores, and writes a CSV results table plus per-image JSON dumps.

- [ ] **Step 1: Implement the CLI**

Create `validation/cli.py`:

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
        help="directory of images; filename format: <set>_<angle>_<light>_<rep>.<ext>",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--prompt-version", default="v1")
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
        set_name = img_path.stem.split("_", 1)[0].upper()
        if set_name not in spec.sets:
            print(f"skipping {img_path.name}: unknown set {set_name}")
            continue

        print(f"extracting {img_path.name} (set {set_name})...")
        detection = extractor.extract(img_path, set_name=set_name)

        _dump_detection(args.out, detection)

        try:
            result = score_detection(detection, spec.sets[set_name].tiles)
            rows.append(_flatten_result(detection, result))
            print(
                f"  {len(detection.detections)} tiles detected, "
                f"mean error {result.mean_error_mm:.3f} mm"
            )
        except ValueError as err:
            print(f"  scoring failed: {err}")
            rows.append({
                "image": img_path.name,
                "set": set_name,
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
        "set": d.set_name,
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

- [ ] **Step 2: Add a `[project.scripts]` entry**

Edit `pyproject.toml` — under `[project]`, add:

```toml
[project.scripts]
run-validation = "validation.cli:main"
```

- [ ] **Step 3: Verify the CLI loads**

Run: `uv run run-validation --help`

Expected: argparse help text appears and exit code 0.

- [ ] **Step 4: Commit**

```bash
git add validation/cli.py pyproject.toml
git commit -m "Add CLI batch runner with CSV results output"
```

---

## Task 9: Validate on synthetic images

**Purpose:** Prove the pipeline works end-to-end **before** real phone photos exist. If the synthetic run does not hit <1 mm error, there's a bug in the code, not in the LLM or the print.

- [ ] **Step 1: Generate a synthetic image for each set**

Create `scripts/generate_synthetic.py`:

```python
from pathlib import Path

from validation.ground_truth import load_strip_spec
from validation.synthetic import render_set


def main() -> None:
    spec = load_strip_spec(Path("data/test-strip-spec.yaml"))
    out = Path("data/photos")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("A", "B"):  # skip C — synthetic ArUco is a placeholder
        img_path = out / f"{name}_top_day_01.png"
        render_set(spec.sets[name], img_path, px_per_mm=40)
        print(f"wrote {img_path}")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/generate_synthetic.py`

Expected: `data/photos/A_top_day_01.png` and `data/photos/B_top_day_01.png` are created.

- [ ] **Step 2: Eyeball the synthetic images**

Open both PNG files. Confirm:
- A row of 10 filled circles with digits beside them.
- Set A at 8 mm tile size, Set B at 5 mm.
- No obvious rendering glitches.

If anything looks wrong, fix `validation/synthetic.py` and regenerate before continuing.

- [ ] **Step 3: Set the Anthropic API key**

Ensure `ANTHROPIC_API_KEY` is set in the shell environment. If not:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # unix/macOS
# or on Windows bash:
export ANTHROPIC_API_KEY="sk-ant-..."
```

Verify: `echo $ANTHROPIC_API_KEY | head -c 10` prints the prefix.

- [ ] **Step 4: Run the batch against synthetic images**

Run:
```bash
uv run run-validation \
  --spec data/test-strip-spec.yaml \
  --photos data/photos \
  --out data/results/synthetic
```

Expected:
- Two images processed.
- Per-image JSON dumps in `data/results/synthetic/`.
- `data/results/synthetic/results.csv` with mean_error_mm columns populated.

- [ ] **Step 5: Verify synthetic error is near zero**

Read `data/results/synthetic/results.csv`. Both rows should have `mean_error_mm < 0.2` (synthetic images are noise-free — the only error is LLM pixel-guessing precision).

If errors are large:
- Check the per-image JSON's `detections`. If tile IDs are wrong, the prompt needs fixing.
- If pixel coords look way off, the synthetic renderer and the ground truth are mis-aligned — check `_mm_to_px` in `validation/synthetic.py`.
- If the homography RMSE is huge but tile-level error is small, the ordering of points passed to `fit_homography` may be inconsistent.

Do not proceed to Task 10 until synthetic mean error is consistently <0.2 mm. This is the non-negotiable correctness gate for the math.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_synthetic.py
git commit -m "Add synthetic image generator script and validate pipeline"
```

---

## Task 10: Validate on real phone photos

**Purpose:** Run the same harness on real printed + photographed fiducials. This is the answer to `docs/next-steps.md` item 1.

- [ ] **Step 1: Capture the photos**

Prerequisites: printed test strip exists, covering Sets A, B, and C. (User is doing this separately.)

Photo protocol per set:
- 3 angles: top-down (as flat to the strip as possible), ~20° tilt, ~45° tilt.
- 2 lighting conditions: daylight near a window, indoor bulb.
- 3 repeats each.
- 18 photos per set × 3 sets = 54 photos.

Filename convention: `<set>_<angle>_<light>_<rep>.<ext>`
- `<set>`: `A`, `B`, or `C`
- `<angle>`: `top`, `20`, `45`
- `<light>`: `day`, `bulb`
- `<rep>`: `01`, `02`, `03`
- Example: `A_top_day_01.jpg`

Place all photos in `data/photos/real/`.

- [ ] **Step 2: Dry-run with one photo to sanity-check**

Run:
```bash
uv run run-validation \
  --spec data/test-strip-spec.yaml \
  --photos data/photos/real \
  --out data/results/real-smoke
```

Note: the batch runner processes every image in the directory. For a dry run, either move all but one photo aside, or temporarily restrict with a subdirectory. Confirm:
- Extraction returns JSON successfully.
- Detected tile IDs match the printed strip's IDs.
- Pixel coordinates look plausible (manually overlay on the image if needed).

If extraction fails or JSON is malformed, iterate on the prompt in `validation/prompts.py`. Commit each prompt variant under a new version name (`v2`, `v3`, ...) so results stay reproducible.

- [ ] **Step 3: Run the full batch**

Run:
```bash
uv run run-validation \
  --spec data/test-strip-spec.yaml \
  --photos data/photos/real \
  --out data/results/real-full
```

Expected: 54 images processed. Cost estimate for Claude Opus 4.7: ~$0.02–0.05 per image, so ~$1–3 total.

- [ ] **Step 4: Analyze the results**

Open `data/results/real-full/results.csv`. Pivot / group by `set` and `mean_error_mm`:

For each set, compute the mean of `mean_error_mm` across its 18 images. Also note:
- How many images failed outright (error_message populated)?
- Does error correlate with angle (top vs. 20° vs. 45°)?
- Does error correlate with lighting (day vs. bulb)?

**Exit criterion** (from `docs/next-steps.md` item 1): at least one set has mean error < 1 mm across all 18 images.

- [ ] **Step 5: Write up the result**

Append to `docs/fiducial-design.md` under the Decision log:

```markdown
- **YYYY-MM-DD** — LLM validation run on real photos completed. Set A mean error: X.XX mm. Set B: Y.YY mm. Set C: Z.ZZ mm. [Passed/Failed] exit criterion.
```

If passed: celebrate and plan Task 4 in `docs/next-steps.md` (end-to-end static POC).

If failed: the decision log entry explains which axis failed (too-small tiles? specific lighting? specific angle?) and the team picks a next iteration — retry with larger tiles, or fall back to the classical-CV path using Set C's ArUco detections.

- [ ] **Step 6: Commit the writeup and results summary**

```bash
git add docs/fiducial-design.md
git commit -m "Record LLM validation run results"
```

---

## Self-review notes

- Every step has runnable code or a runnable command — no "add appropriate X" placeholders.
- Types are consistent: `GroundTruthTile`, `DetectedTile`, `ImageDetection`, `ScoringResult` are defined once in `schemas.py` and used consistently across all downstream tasks.
- TDD is applied where it fits (schemas, ground truth loader, synthetic renderer, homography, scoring, extraction parsing). It is skipped where the task is experimental (capturing real photos, analyzing results).
- The pipeline is validated against synthetic images in Task 9 before consuming real-photo budget and before the print-hardware variable enters.
- The plan ends with a concrete decision outcome: the answer to `docs/next-steps.md` item 1 (does LLM vision hit <1 mm?), and a clear branch for either outcome.
