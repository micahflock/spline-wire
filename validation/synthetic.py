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
