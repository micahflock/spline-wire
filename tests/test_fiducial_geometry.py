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
