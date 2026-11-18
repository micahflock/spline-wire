"""The DXF of our own (splinewire/dxf.py): a file CAD programs read without
complaint, in mm, with one spline through every curve point."""
import io

import ezdxf
import numpy as np
import pytest
from ezdxf import recover
from ezdxf.math import BSpline, fit_points_to_cad_cv

from splinewire.dxf import cad_spline, dxf_text

CURVES = {
    "two": np.array([[0.0, 0.0], [10.0, 0.0]]),
    "three": np.array([[0.0, 0.0], [10.0, 2.0], [20.0, 0.0]]),
    "walk": np.cumsum(np.random.default_rng(1).normal(0, 5, (30, 2)), axis=0),
}


@pytest.mark.parametrize("name", CURVES)
def test_reads_back_clean(name):
    curve = CURVES[name]
    text = dxf_text(curve)
    doc = ezdxf.read(io.StringIO(text))
    auditor = doc.audit()
    assert not auditor.errors and not auditor.fixes
    _, recovered = recover.read(io.BytesIO(text.encode("ascii")))
    assert not recovered.errors
    assert doc.dxfversion == "AC1015" and doc.header["$INSUNITS"] == 4
    msp = doc.modelspace()
    assert len(msp.query("SPLINE")) == 1
    points = msp.query("POINT")
    assert {e.dxf.layer for e in points} == {"CURVE_POINTS"}
    np.testing.assert_allclose([(p.dxf.location.x, p.dxf.location.y) for p in points], curve, atol=1e-6)


@pytest.mark.parametrize("name", CURVES)
def test_spline_passes_through_every_point_as_cad_builds_it(name):
    curve = CURVES[name]
    knots, control = cad_spline(curve)
    ref = fit_points_to_cad_cv([(x, y, 0.0) for x, y in curve])      # how CAD programs build it
    np.testing.assert_allclose(knots, ref.knots(), atol=1e-12)
    np.testing.assert_allclose(control, np.array(ref.control_points)[:, :2], atol=1e-9)
    # As written (control points to 1 nm, knots to 1e-9): still through every point.
    sp = ezdxf.read(io.StringIO(dxf_text(curve))).modelspace().query("SPLINE")[0]
    tool = BSpline(sp.control_points, order=4, knots=sp.knots)
    d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(d)] / d.sum()
    assert max(np.linalg.norm(np.array(tool.point(u))[:2] - p) for u, p in zip(t, curve)) < 1e-5


def test_same_curve_same_file():
    """No dates, GUIDs or anything else that changes: the browser writes it byte for byte."""
    assert dxf_text(CURVES["walk"]) == dxf_text(CURVES["walk"].copy())
    assert "-0.000000" not in dxf_text(np.array([[-1e-12, 0.0], [10.0, -1e-9], [20.0, 0.0]]))
