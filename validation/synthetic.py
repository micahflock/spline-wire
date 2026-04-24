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
