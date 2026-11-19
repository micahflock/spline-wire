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
    ("Inches\n1 0\n0 1\n", [(25.4, 0), (0, 25.4)]),
    ("points in mm\n1\t2\n3\t4\n", [(1, 2), (3, 4)]),             # "in" the word, not the unit
    ("x in cm, y in cm\n1,2\n3,4\n", [(10, 20), (30, 40)]),
    ("1\t2\n3\t4\nx_cm\ty_cm\n5\t6\n", [(1, 2), (3, 4), (5, 6)]),   # units are set by a header, not mid-table
])
def test_parse_points_formats(addin, text, expected):
    import numpy as np

    from SplineWire import curvedata
    np.testing.assert_allclose(curvedata.parse_points(text), expected)


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


# --- the palette spike (fusion/SplineWirePaletteSpike/, docs/fusion-palette.md) -------------
# The fake grows events (add/remove/fire), palettes that a workspace switch deletes,
# custom events and commands whose execute() runs commandCreated and then execute handlers.

SPIKE_PAGE = Path(__file__).parent.parent / "web" / "spikes" / "fusion-palette" / "main.ts"


class Event:
    def __init__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)

    def remove(self, handler):
        self.handlers.remove(handler)

    def fire(self, args):
        for h in list(self.handlers):
            h.notify(args)


class Palette:
    def __init__(self, palettes, pid, url, visible):
        self.id, self.htmlFileURL, self.isVisible = pid, url, visible
        self.incomingFromHTML, self.sent, self._palettes = Event(), [], palettes

    def sendInfoToHTML(self, action, data):
        self.sent.append((action, data))
        return "OK"

    def deleteMe(self):
        self._palettes.items.pop(self.id)


class Palettes:
    def __init__(self):
        self.items = {}

    def itemById(self, pid):
        return self.items.get(pid)

    def add(self, pid, name, url, visible, close_button, resizable, width, height):
        self.items[pid] = Palette(self, pid, url, visible)
        return self.items[pid]


class SpikeCommandDefinitions(CommandDefinitions):
    def addButtonDefinition(self, cid, name, tooltip):
        cmd = types.SimpleNamespace(id=cid, name=name, commandCreated=Event(), runs=0)
        defs = self

        def execute():                       # as Fusion runs a command with no inputs
            cmd.runs += 1
            command = types.SimpleNamespace(execute=Event())
            cmd.commandCreated.fire(types.SimpleNamespace(command=command))
            command.execute.fire(types.SimpleNamespace())
        cmd.execute = execute
        cmd.deleteMe = lambda: defs.items.pop(cid)
        self.items[cid] = cmd
        return cmd


@pytest.fixture
def spike(monkeypatch, tmp_path):
    panel = types.SimpleNamespace(controls=Controls())
    ui = types.SimpleNamespace(
        commandDefinitions=SpikeCommandDefinitions(), palettes=Palettes(), workspaceActivated=Event(),
        allToolbarPanels=types.SimpleNamespace(itemById={"SolidScriptsAddinsPanel": panel}.get), messages=[],
    )
    ui.messageBox = lambda *a: ui.messages.append(a[0])
    app = types.SimpleNamespace(userInterface=ui, activeProduct=FakeDesign(), activeEditObject=None,
                                version="2.0.21000", custom={}, fired=[])

    def register(eid):
        app.custom[eid] = Event()
        return app.custom[eid]
    app.registerCustomEvent = register
    app.unregisterCustomEvent = lambda eid: app.custom.pop(eid)
    app.fireCustomEvent = lambda eid, info: app.fired.append((eid, info))

    core = types.ModuleType("adsk.core")
    core.Application = types.SimpleNamespace(get=lambda: app)
    core.Point3D, core.ObjectCollection = Point3D, ObjectCollection
    for name in ("CommandCreatedEventHandler", "CommandEventHandler", "CustomEventHandler",
                 "WorkspaceEventHandler", "HTMLEventHandler"):
        setattr(core, name, type(name, (), {}))
    core.HTMLEventArgs = types.SimpleNamespace(cast=lambda args: args)
    fusion = types.ModuleType("adsk.fusion")
    fusion.Design = types.SimpleNamespace(cast=_cast(FakeDesign).__func__)
    fusion.Sketch = types.SimpleNamespace(cast=_cast(FakeSketch).__func__)
    adsk = types.ModuleType("adsk")
    adsk.core, adsk.fusion = core, fusion
    monkeypatch.setitem(sys.modules, "adsk", adsk)
    monkeypatch.setitem(sys.modules, "adsk.core", core)
    monkeypatch.setitem(sys.modules, "adsk.fusion", fusion)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.syspath_prepend(str(ADDIN_DIR))
    for name in ("SplineWirePaletteSpike.SplineWirePaletteSpike", "SplineWirePaletteSpike"):
        sys.modules.pop(name, None)
    module = importlib.import_module("SplineWirePaletteSpike.SplineWirePaletteSpike")
    module.run({})
    yield module, app, panel
    if module._events:
        module.stop({})


def _ask(app, action, data):
    """The page's adsk.fusionSendData(action, data): the add-in's returnData, parsed."""
    import json
    args = types.SimpleNamespace(action=action, data=data, returnData="")
    app.userInterface.palettes.itemById("SplineWirePaletteSpike").incomingFromHTML.fire(args)
    return json.loads(args.returnData)


def _deliver(app):
    """Fusion delivering the custom events fired during an HTML event, afterwards."""
    fired, app.fired = app.fired, []
    for eid, info in fired:
        app.custom[eid].fire(types.SimpleNamespace(additionalInfo=info))


def test_spike_opens_the_hosted_page_in_a_palette_and_stop_undoes_everything(spike):
    module, app, panel = spike
    ui = app.userInterface
    palette = ui.palettes.itemById(module.PALETTE_ID)
    assert ui.messages == []
    assert palette.htmlFileURL.startswith("https://") and palette.isVisible     # not a file: Origin null is refused
    assert len(palette.incomingFromHTML.handlers) == 1
    assert panel.controls.itemById(module.CMD_SHOW) is not None
    module.stop({})
    assert ui.palettes.items == {} and app.custom == {} and ui.commandDefinitions.items == {}
    assert panel.controls.itemById(module.CMD_SHOW) is None and ui.workspaceActivated.handlers == []


def test_spike_echo_answers_with_the_nonce_and_fusion_facts(spike):
    module, app, _ = spike
    app.activeEditObject = FakeSketch()
    app.activeEditObject.name = "Sketch3"
    reply = _ask(app, "spike-echo", '{"v": 1, "nonce": "abc"}')
    assert reply["ok"] and reply["nonce"] == "abc" and reply["fusion"] == "2.0.21000"
    assert reply["sketch"] == "Sketch3" and reply["palettes_made"] == 1 and reply["started"]


def test_spike_insert_runs_as_a_command_and_reports_back(spike):
    module, app, _ = spike
    sketch = app.activeEditObject = FakeSketch()
    sketch.name = "Sketch1"
    assert _ask(app, "spike-insert", '{"v": 1, "points_mm": [[0, 0], [20, 10], [40, 0]]}') == {"ok": True}
    assert sketch.splines == []                    # nothing inside the HTML event itself
    _deliver(app)
    assert app.userInterface.commandDefinitions.itemById(module.CMD_INSERT).runs == 1   # one command: one undo step
    assert [(p.x, p.y, p.z) for p in sketch.splines[0]] == [(0, 0, 0), (2.0, 1.0, 0), (4.0, 0, 0)]   # mm -> cm
    palette = app.userInterface.palettes.itemById(module.PALETTE_ID)
    assert palette.sent == [("spike-inserted", '{"ok": true, "sketch": "Sketch1", "points": 3}')]


def test_spike_insert_without_a_sketch_makes_one_on_xy(spike):
    module, app, _ = spike
    _ask(app, "spike-insert", '{"v": 1, "points_mm": [[0, 0], [1, 1]]}')
    _deliver(app)
    [sketch] = app.activeProduct.created
    assert sketch.plane == "XY" and len(sketch.splines) == 1


@pytest.mark.parametrize("action, data, message", [
    ("spike-delete", '{"v": 1}', "Unknown request"),
    ("spike-echo", None, "not JSON"),
    ("spike-echo", "eval('x')", "not JSON"),
    ("spike-echo", '{"v": 2}', "another version"),
    ("spike-echo", "[1]", "another version"),
    ("spike-echo", '{"v": 1, "pad": "' + "x" * 70000 + '"}', "too much"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0], [NaN, 1]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0], [1e6, 1]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0], [true, 1]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0], [1, 1, 1]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": [[0, 0], ["1", 1]]}', "2 to 2000"),
    ("spike-insert", '{"v": 1, "points_mm": ' + str([[0, 0]] * 2001) + "}", "2 to 2000"),
])
def test_spike_refuses_bad_requests_with_a_message(spike, action, data, message):
    _, app, _ = spike
    reply = _ask(app, action, data)
    assert reply["ok"] is False and message in reply["error"] and "Traceback" not in reply["error"]
    assert app.fired == []


def test_spike_insert_with_no_design_reports_an_error_to_the_page(spike):
    module, app, _ = spike
    app.activeProduct = object()
    _ask(app, "spike-insert", '{"v": 1, "points_mm": [[0, 0], [1, 1]]}')
    _deliver(app)
    [(action, data)] = app.userInterface.palettes.itemById(module.PALETTE_ID).sent
    assert action == "spike-inserted" and "Open a design" in data and '"ok": false' in data


def test_spike_makes_the_palette_again_after_a_workspace_switch(spike):
    module, app, _ = spike
    ui = app.userInterface
    ui.workspaceActivated.fire(types.SimpleNamespace())         # the palette survived this one
    ui.palettes.items.clear()                                    # this one deleted it
    ui.workspaceActivated.fire(types.SimpleNamespace())
    palette = ui.palettes.itemById(module.PALETTE_ID)
    assert palette is not None and len(palette.incomingFromHTML.handlers) == 1
    reply = _ask(app, "spike-echo", '{"v": 1}')
    assert (reply["palettes_made"], reply["switches"], reply["palette_lost"]) == (2, 2, 1)


def test_spike_saves_the_summary_to_a_file(spike, tmp_path):
    _, app, _ = spike
    reply = _ask(app, "spike-report", '{"v": 1, "text": "chromium: 108"}')
    assert reply["ok"] and Path(reply["path"]).read_text(encoding="utf-8") == "chromium: 108"
    assert Path(reply["path"]).parent == tmp_path


def test_spike_page_and_add_in_agree_on_the_actions(spike):
    import re
    module, _, _ = spike
    page = SPIKE_PAGE.read_text(encoding="utf-8")
    source = Path(module.__file__).read_text(encoding="utf-8")
    actions = set(re.findall(r'"(spike-[a-z]+)"', source))
    assert actions == {"spike-echo", "spike-insert", "spike-report", "spike-inserted"}
    assert all(f'"{a}"' in page for a in actions)
    assert module.PAGE_URL.endswith("/spikes/fusion-palette/")
