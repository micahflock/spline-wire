"""The Fusion add-in, run against a minimal stand-in for Fusion's adsk API."""
import importlib
import sys
import types
from pathlib import Path

import pytest

ADDIN_DIR = Path(__file__).parent.parent / "fusion"


# --- a tiny fake of the parts of adsk.core / adsk.fusion the add-in uses ----

class Point3D:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    @staticmethod
    def create(x, y, z):
        return Point3D(x, y, z)


class ObjectCollection(list):
    @staticmethod
    def create():
        return ObjectCollection()

    def add(self, item):
        self.append(item)


class SketchPoint:
    def __init__(self, geometry):
        self.geometry = geometry


class FakeSketch:
    def __init__(self, accept_sketch_points=True):
        self.points, self.splines, self.name = [], [], ""
        self.accept_sketch_points = accept_sketch_points
        sketch = self
        self.sketchPoints = types.SimpleNamespace(add=self._add_point)

        class Splines:
            def add(self, fit_points):
                if isinstance(fit_points[0], SketchPoint) and not sketch.accept_sketch_points:
                    raise RuntimeError("fit points must be Point3D")
                sketch.splines.append(list(fit_points))
        self.sketchCurves = types.SimpleNamespace(sketchFittedSplines=Splines())

    def _add_point(self, xyz):
        p = SketchPoint(xyz)
        self.points.append(p)
        return p


class FakeDesign:
    def __init__(self):
        self.created = []
        design = self

        class Sketches:
            def add(self, plane):
                s = FakeSketch()
                s.plane = plane
                design.created.append(s)
                return s
        self.rootComponent = types.SimpleNamespace(sketches=Sketches(), xYConstructionPlane="XY")


def _cast(kind):
    return staticmethod(lambda obj: obj if isinstance(obj, kind) else None)


class Controls:
    def __init__(self):
        self.items = {}

    def itemById(self, cid):
        return self.items.get(cid)

    def addCommand(self, cmd_def):
        control = types.SimpleNamespace(isPromoted=False, deleteMe=lambda: self.items.pop(cmd_def.id))
        self.items[cmd_def.id] = control
        return control


class CommandDefinitions:
    def __init__(self):
        self.items = {}

    def itemById(self, cid):
        return self.items.get(cid)

    def addButtonDefinition(self, cid, name, tooltip):
        defs = self
        cmd = types.SimpleNamespace(id=cid, name=name, handlers=[])
        cmd.commandCreated = types.SimpleNamespace(add=cmd.handlers.append)
        cmd.deleteMe = lambda: defs.items.pop(cid)
        self.items[cid] = cmd
        return cmd


@pytest.fixture
def addin(monkeypatch):
    panels = {"SketchCreatePanel": types.SimpleNamespace(controls=Controls()),
              "InsertPanel": types.SimpleNamespace(controls=Controls())}
    ui = types.SimpleNamespace(
        commandDefinitions=CommandDefinitions(),
        allToolbarPanels=types.SimpleNamespace(itemById=panels.get),
        messages=[],
    )
    ui.messageBox = lambda *a: ui.messages.append(a[0])
    app = types.SimpleNamespace(userInterface=ui, activeProduct=FakeDesign(), activeEditObject=None)

    core = types.ModuleType("adsk.core")
    core.Application = types.SimpleNamespace(get=lambda: app)
    core.Point3D, core.ObjectCollection = Point3D, ObjectCollection
    core.CommandCreatedEventHandler = core.CommandEventHandler = object
    fusion = types.ModuleType("adsk.fusion")
    fusion.Design = types.SimpleNamespace(cast=_cast(FakeDesign).__func__)
    fusion.Sketch = types.SimpleNamespace(cast=_cast(FakeSketch).__func__)
    adsk = types.ModuleType("adsk")
    adsk.core, adsk.fusion = core, fusion
    monkeypatch.setitem(sys.modules, "adsk", adsk)
    monkeypatch.setitem(sys.modules, "adsk.core", core)
    monkeypatch.setitem(sys.modules, "adsk.fusion", fusion)
    monkeypatch.syspath_prepend(str(ADDIN_DIR))
    for name in ("SplineWire.SplineWire", "SplineWire.curvedata", "SplineWire"):
        sys.modules.pop(name, None)
    module = importlib.import_module("SplineWire.SplineWire")
    return module, app, panels


TABLE = "x_mm\ty_mm\n0.0000\t0.0000\n10.0000\t5.0000\n20.0000\t3.0000\n"


def test_paste_into_the_active_sketch_converts_mm_to_cm(addin):
    module, app, _ = addin
    app.activeEditObject = FakeSketch()
    sketch = module.paste_points(app, TABLE)
    assert sketch is app.activeEditObject
    assert [(p.geometry.x, p.geometry.y) for p in sketch.points] == [(0, 0), (1.0, 0.5), (2.0, 0.3)]
    assert len(sketch.splines) == 1 and sketch.splines[0][0] is sketch.points[0]


def test_paste_without_an_open_sketch_makes_one_on_xy(addin):
    module, app, _ = addin
    sketch = module.paste_points(app, TABLE)
    assert app.activeProduct.created == [sketch]
    assert sketch.plane == "XY" and sketch.name == "Spline Wire"


def test_spline_falls_back_to_plain_coordinates(addin):
    module, app, _ = addin
    app.activeEditObject = FakeSketch(accept_sketch_points=False)
    sketch = module.paste_points(app, TABLE)
    assert len(sketch.points) == 3                       # the pasted points are still there
    assert isinstance(sketch.splines[0][0], Point3D)


def test_bad_clipboard_or_no_design_is_a_friendly_error(addin):
    module, app, _ = addin
    with pytest.raises(ValueError, match="Copy points"):
        module.paste_points(app, "some text someone copied")
    app.activeProduct = object()
    with pytest.raises(ValueError, match="Open a design"):
        module.paste_points(app, TABLE)


def test_run_adds_promoted_buttons_and_stop_removes_them(addin):
    module, app, panels = addin
    module.run({})
    assert app.userInterface.messages == []
    for panel in panels.values():
        control = panel.controls.itemById(module.CMD_ID)
        assert control is not None and control.isPromoted
    module.stop({})
    assert all(p.controls.itemById(module.CMD_ID) is None for p in panels.values())
    assert app.userInterface.commandDefinitions.itemById(module.CMD_ID) is None


@pytest.mark.parametrize("text, expected", [
    ("x_mm\ty_mm\n1\t2\n3\t4\n", [(1, 2), (3, 4)]),              # what Spline Wire copies
    ("x_cm,y_cm,z_cm\n1,2,0\n3,4,0\n", [(10, 20), (30, 40)]),    # cm header, xyz
    ("1.5 -2e1\n.5 3\n", [(1.5, -20), (0.5, 3)]),                # spaces, exponents
    ("X (in);Y (in)\n1;0\n0;1\n", [(25.4, 0), (0, 25.4)]),       # inches, semicolons
])
def test_parse_points_formats(addin, text, expected):
    import numpy as np

    from SplineWire import curvedata
    np.testing.assert_allclose(curvedata.parse_points(text), expected)


def test_install_copies_the_addin_into_fusions_folder(tmp_path):
    from splinewire.fusion_addin import install_addin

    addins = tmp_path / "Autodesk" / "Autodesk Fusion 360" / "API" / "AddIns"
    with pytest.raises(FileNotFoundError, match="Fusion"):
        install_addin(addins)                      # Fusion not installed yet
    (tmp_path / "Autodesk" / "Autodesk Fusion 360").mkdir(parents=True)
    dest = install_addin(addins)
    assert (dest / "SplineWire.py").is_file()
    assert (dest / "SplineWire.manifest").is_file()
    assert (dest / "curvedata.py").is_file()
    assert not list(dest.rglob("__pycache__"))
    install_addin(addins)                          # reinstalling over an old copy is fine


def test_copied_table_round_trips_through_the_addin_parser():
    import numpy as np

    from splinewire.output import points_tsv
    sys.path.insert(0, str(ADDIN_DIR / "SplineWire"))
    try:
        import curvedata
    finally:
        sys.path.pop(0)
    pts = np.array([[0.0, 0.0], [12.3456, -7.25], [30.5, 4.125]])
    np.testing.assert_allclose(curvedata.parse_points(points_tsv(pts)), pts, atol=1e-4)
