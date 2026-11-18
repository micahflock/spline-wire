"""The curve as a small DXF of our own: one cubic spline through the curve
points (layer CURVE) and the points themselves (layer CURVE_POINTS), in mm.

DXF R2000 (AC1015), with what CAD programs expect of one and nothing more:
the header ($ACADVER, $HANDSEED, $INSUNITS = mm so importers don't guess
the scale), empty CLASSES, the tables (line types ByBlock, ByLayer and
Continuous; layers 0, CURVE and CURVE_POINTS; the Standard text and dimension
styles; the ACAD application id; model and paper space), their two blocks,
the entities, and the root dictionary. Every handle is fixed and nothing
depends on the time, so the same curve gives the same file, here and in the
browser (web/src/pipeline/dxf.ts writes it byte for byte; decision D19 in
docs/web-build.md). It replaced ezdxf's output (an 18 KB file with dates and
GUIDs) on 2026-11-18.

The spline is built as CAD programs build one from fit points, so it passes
through every point: a cubic B-spline by global interpolation, knots at the
points' chord-length parameters, the second derivative zero at both ends
(Piegl & Tiller, The NURBS Book, 9.2.1 with (3.9)/(3.10); the construction
ezdxf's add_cad_spline_control_frame uses). Coordinates are written to 1 nm
(6 decimals), knots to 9 decimals.
"""
from __future__ import annotations

import numpy as np

# Fixed handles (hex). Tables and their records, then the blocks, then the
# entities from ENTITY_HANDLE on.
_H = {
    "vport_table": "8", "ltype_table": "5", "layer_table": "2", "style_table": "3", "view_table": "6",
    "ucs_table": "7", "appid_table": "9", "dimstyle_table": "A", "block_record_table": "1",
    "ltype_byblock": "14", "ltype_bylayer": "15", "ltype_continuous": "16",
    "layer_0": "10", "layer_curve": "11", "layer_points": "12",
    "style_standard": "13", "appid_acad": "17", "dimstyle_standard": "18",
    "model_record": "1F", "paper_record": "1B",
    "model_block": "20", "model_endblk": "21", "paper_block": "1C", "paper_endblk": "1D",
    "root_dict": "C", "group_dict": "D",
}
ENTITY_HANDLE = 0x100


def dxf_text(curve_points_mm) -> str:
    """The DXF file's text for a curve through the points (mm)."""
    pts = np.asarray(curve_points_mm, dtype=float).reshape(-1, 2)
    tags: list[tuple[int, str]] = []
    add = tags.append
    h = _H

    def section(name: str) -> None:
        add((0, "SECTION"))
        add((2, name))

    def table(name: str, handle: str, count: int) -> None:
        add((0, "TABLE"))
        add((2, name))
        add((5, handle))
        add((330, "0"))
        add((100, "AcDbSymbolTable"))
        add((70, str(count)))

    def record(kind: str, handle: str, owner: str, subclass: str) -> None:
        add((0, kind))
        add((5, handle))
        add((330, owner))
        add((100, "AcDbSymbolTableRecord"))
        add((100, subclass))

    entity_count = 1 + len(pts) if len(pts) >= 2 else len(pts)
    section("HEADER")
    for name, code, value in (("$ACADVER", 1, "AC1015"), ("$HANDSEED", 5, f"{ENTITY_HANDLE + entity_count:X}"),
                              ("$INSUNITS", 70, "4"), ("$MEASUREMENT", 70, "1")):
        add((9, name))
        add((code, value))
    add((0, "ENDSEC"))
    section("CLASSES")
    add((0, "ENDSEC"))

    section("TABLES")
    table("VPORT", h["vport_table"], 0)
    add((0, "ENDTAB"))
    table("LTYPE", h["ltype_table"], 3)
    for name, handle, text in (("ByBlock", h["ltype_byblock"], ""), ("ByLayer", h["ltype_bylayer"], ""),
                               ("Continuous", h["ltype_continuous"], "Solid line")):
        record("LTYPE", handle, h["ltype_table"], "AcDbLinetypeTableRecord")
        add((2, name))
        add((70, "0"))
        add((3, text))
        add((72, "65"))
        add((73, "0"))
        add((40, "0.0"))
    add((0, "ENDTAB"))
    table("LAYER", h["layer_table"], 3)
    for name, handle, color in (("0", h["layer_0"], "7"), ("CURVE", h["layer_curve"], "7"),
                                ("CURVE_POINTS", h["layer_points"], "1")):
        record("LAYER", handle, h["layer_table"], "AcDbLayerTableRecord")
        add((2, name))
        add((70, "0"))
        add((62, color))
        add((6, "Continuous"))
        add((370, "-3"))
    add((0, "ENDTAB"))
    table("STYLE", h["style_table"], 1)
    record("STYLE", h["style_standard"], h["style_table"], "AcDbTextStyleTableRecord")
    for code, value in ((2, "Standard"), (70, "0"), (40, "0.0"), (41, "1.0"), (50, "0.0"), (71, "0"),
                        (42, "2.5"), (3, "txt"), (4, "")):
        add((code, value))
    add((0, "ENDTAB"))
    for name, key in (("VIEW", "view_table"), ("UCS", "ucs_table")):
        table(name, h[key], 0)
        add((0, "ENDTAB"))
    table("APPID", h["appid_table"], 1)
    record("APPID", h["appid_acad"], h["appid_table"], "AcDbRegAppTableRecord")
    add((2, "ACAD"))
    add((70, "0"))
    add((0, "ENDTAB"))
    table("DIMSTYLE", h["dimstyle_table"], 1)
    add((100, "AcDbDimStyleTable"))
    add((0, "DIMSTYLE"))
    add((105, h["dimstyle_standard"]))
    add((330, h["dimstyle_table"]))
    add((100, "AcDbSymbolTableRecord"))
    add((100, "AcDbDimStyleTableRecord"))
    add((2, "Standard"))
    add((70, "0"))
    add((0, "ENDTAB"))
    table("BLOCK_RECORD", h["block_record_table"], 2)
    for name, handle, block in (("*Model_Space", h["model_record"], h["model_block"]),
                                ("*Paper_Space", h["paper_record"], h["paper_block"])):
        record("BLOCK_RECORD", handle, h["block_record_table"], "AcDbBlockTableRecord")
        add((2, name))
    add((0, "ENDTAB"))
    add((0, "ENDSEC"))

    section("BLOCKS")
    for name, rec, block, endblk in (("*Model_Space", h["model_record"], h["model_block"], h["model_endblk"]),
                                     ("*Paper_Space", h["paper_record"], h["paper_block"], h["paper_endblk"])):
        for code, value in ((0, "BLOCK"), (5, block), (330, rec), (100, "AcDbEntity"), (8, "0"),
                            (100, "AcDbBlockBegin"), (2, name), (70, "0"), (10, "0.0"), (20, "0.0"), (30, "0.0"),
                            (3, name), (1, ""),
                            (0, "ENDBLK"), (5, endblk), (330, rec), (100, "AcDbEntity"), (8, "0"),
                            (100, "AcDbBlockEnd")):
            add((code, value))
    add((0, "ENDSEC"))

    section("ENTITIES")
    handle = ENTITY_HANDLE
    if len(pts) >= 2:
        knots, control = cad_spline(pts)
        for code, value in ((0, "SPLINE"), (5, f"{handle:X}"), (330, h["model_record"]), (100, "AcDbEntity"),
                            (8, "CURVE"), (100, "AcDbSpline"), (210, "0.0"), (220, "0.0"), (230, "1.0"),
                            (70, "8"), (71, "3"), (72, str(len(knots))), (73, str(len(control))), (74, "0")):
            add((code, value))
        for k in knots:
            add((40, _num(k, 9)))
        for x, y in control:
            add((10, _num(x, 6)))
            add((20, _num(y, 6)))
            add((30, "0.0"))
        handle += 1
    for x, y in pts:
        for code, value in ((0, "POINT"), (5, f"{handle:X}"), (330, h["model_record"]), (100, "AcDbEntity"),
                            (8, "CURVE_POINTS"), (100, "AcDbPoint"), (10, _num(x, 6)), (20, _num(y, 6)), (30, "0.0")):
            add((code, value))
        handle += 1
    add((0, "ENDSEC"))

    section("OBJECTS")
    for code, value in ((0, "DICTIONARY"), (5, h["root_dict"]), (330, "0"), (100, "AcDbDictionary"), (281, "1"),
                        (3, "ACAD_GROUP"), (350, h["group_dict"]),
                        (0, "DICTIONARY"), (5, h["group_dict"]), (330, h["root_dict"]), (100, "AcDbDictionary"),
                        (281, "1")):
        add((code, value))
    add((0, "ENDSEC"))
    add((0, "EOF"))
    return "".join(f"{code:>3}\n{value}\n" for code, value in tags)


def _num(x: float, digits: int) -> str:
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-") and not s.strip("-0.") else s       # no "-0.000000"


def cad_spline(points_mm) -> tuple[np.ndarray, np.ndarray]:
    """Knots and control points of the cubic B-spline through the points:
    chord-length parameters, the interior ones as knots, the second
    derivative zero at both ends."""
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 2)
    n = len(pts) - 1
    p = 3
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(d)[:-1] / d.sum(), 1.0]
    knots = np.r_[[0.0] * 4, t[1:n], [1.0] * 4]
    count = n + 3
    rows = [_basis(knots, count, u) for u in t]
    up1, up2 = knots[p + 1], knots[p + 2]
    f1 = p * (p - 1) / up1
    c1 = np.r_[f1 / up1, -f1 * (up1 + up2) / (up1 * up2), f1 / up2, np.zeros(n)]
    m = len(knots) - 1
    ump1, ump2 = knots[m - p - 1], knots[m - p - 2]
    f2 = p * (p - 1) / (1.0 - ump1)
    c2 = np.r_[np.zeros(n), f2 / (1.0 - ump2), -f2 * (2.0 - ump1 - ump2) / (1.0 - ump1) / (1.0 - ump2), f2 / (1.0 - ump1)]
    rows.insert(1, c1)
    rows.insert(len(rows) - 1, c2)
    rhs = np.vstack([pts[:1], [[0.0, 0.0]], pts[1:-1], [[0.0, 0.0]], pts[-1:]]) if n >= 1 else pts
    return knots, np.linalg.solve(np.array(rows), rhs)


def _basis(knots: np.ndarray, count: int, u: float) -> np.ndarray:
    """The count cubic basis functions at u (Cox-de Boor, Piegl & Tiller A2.2)."""
    p = 3
    span = count - 1 if u >= knots[count] else int(np.searchsorted(knots, u, side="right")) - 1
    span = max(p, min(span, count - 1))
    N = [1.0, 0.0, 0.0, 0.0]
    left, right = [0.0] * 4, [0.0] * 4
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    row = np.zeros(count)
    row[span - p:span + 1] = N
    return row
