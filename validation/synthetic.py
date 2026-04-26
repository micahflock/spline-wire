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
