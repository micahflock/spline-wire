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
