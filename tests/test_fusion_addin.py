"""The Fusion add-in, run against a minimal stand-in for Fusion's adsk API."""
import importlib
import sys
import types
from pathlib import Path

import pytest

ADDIN_DIR = Path(__file__).parent.parent / "fusion"


# --- a fake of the parts of adsk.core / adsk.fusion the add-ins use --------------------------
# Events can be fired; palettes are kept by id (a test can delete them, as a workspace switch may);
# a command's execute() runs its commandCreated handlers, then (unless the test cancels the
# dialog) its execute handlers, then its destroy handlers, as Fusion runs one.

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
    def __init__(self, accept_sketch_points=True, name="Sketch1"):
        self.points, self.splines, self.name = [], [], name
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
                s = FakeSketch(name="")
                s.plane = plane
                design.created.append(s)
                return s
        self.rootComponent = types.SimpleNamespace(sketches=Sketches(), xYConstructionPlane="XY")


def _cast(kind):
    return staticmethod(lambda obj: obj if isinstance(obj, kind) else None)


class Event:
    def __init__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)

    def remove(self, handler):
        self.handlers.remove(handler)

    def fire(self, args=None):
        args = args if args is not None else types.SimpleNamespace()
        for h in list(self.handlers):
            h.notify(args)
        return args


class Controls:
    def __init__(self):
        self.items = {}

    def itemById(self, cid):
        return self.items.get(cid)

    def addCommand(self, cmd_def):
        control = types.SimpleNamespace(isPromoted=False, deleteMe=lambda: self.items.pop(cmd_def.id))
        self.items[cmd_def.id] = control
        return control


class SelectionInput:
    def __init__(self, chosen):
        self.filters, self.limits, self._chosen = [], None, chosen

    def addSelectionFilter(self, name):
        self.filters.append(name)

    def setSelectionLimits(self, low, high):
        self.limits = (low, high)

    @property
    def selectionCount(self):
        return len(self._chosen)

    def selection(self, i):
        return types.SimpleNamespace(entity=self._chosen[i])


class CommandDefinitions:
    def __init__(self, dialog):
        self.items, self.dialog = {}, dialog       # dialog: what the person does in a command's dialog

    def itemById(self, cid):
        return self.items.get(cid)

    def addButtonDefinition(self, cid, name, tooltip):
        defs = self
        cmd = types.SimpleNamespace(id=cid, name=name, commandCreated=Event(), runs=[])

        def execute():
            inputs = []
            command = types.SimpleNamespace(execute=Event(), destroy=Event())
            command.commandInputs = types.SimpleNamespace(
                addSelectionInput=lambda *a: inputs.append(SelectionInput(defs.dialog.get("pick", []))) or inputs[-1])
            cmd.commandCreated.fire(types.SimpleNamespace(command=command))
            ran = None
            if not (inputs and defs.dialog.get("cancel")):
                ran = command.execute.fire(types.SimpleNamespace(executeFailed=False, executeFailedMessage=""))
            command.destroy.fire()
            cmd.runs.append(types.SimpleNamespace(inputs=inputs, args=ran))
        cmd.execute = execute
        cmd.deleteMe = lambda: defs.items.pop(cid)
        self.items[cid] = cmd
        return cmd


class Palette:
    def __init__(self, palettes, pid, name, url, visible):
        self.id, self.name, self.htmlFileURL, self.isVisible = pid, name, url, visible
        self.incomingFromHTML, self.closed, self.sent, self._palettes = Event(), Event(), [], palettes
        self.dockingState = None

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
        self.items[pid] = Palette(self, pid, name, url, visible)
        return self.items[pid]


def fake_fusion(monkeypatch, panel_ids):
    """Install a fake `adsk` and return (app, panels)."""
    panels = {pid: types.SimpleNamespace(controls=Controls()) for pid in panel_ids}
    dialog = {}
    ui = types.SimpleNamespace(
        commandDefinitions=CommandDefinitions(dialog), palettes=Palettes(), workspaceActivated=Event(),
        allToolbarPanels=types.SimpleNamespace(itemById=panels.get), messages=[],
    )
    ui.messageBox = lambda *a: ui.messages.append(a[0])
    app = types.SimpleNamespace(userInterface=ui, activeProduct=FakeDesign(), activeEditObject=None,
                                version="2705.1.30", custom={}, fired=[], dialog=dialog)

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
                 "WorkspaceEventHandler", "HTMLEventHandler", "UserInterfaceGeneralEventHandler"):
        setattr(core, name, type(name, (), {}))
    core.HTMLEventArgs = types.SimpleNamespace(cast=lambda args: args)
    core.PaletteDockingStates = types.SimpleNamespace(PaletteDockStateRight="right")
    fusion = types.ModuleType("adsk.fusion")
    fusion.Design = types.SimpleNamespace(cast=_cast(FakeDesign).__func__)
    fusion.Sketch = types.SimpleNamespace(cast=_cast(FakeSketch).__func__)
    adsk = types.ModuleType("adsk")
    adsk.core, adsk.fusion = core, fusion
    monkeypatch.setitem(sys.modules, "adsk", adsk)
    monkeypatch.setitem(sys.modules, "adsk.core", core)
    monkeypatch.setitem(sys.modules, "adsk.fusion", fusion)
    monkeypatch.syspath_prepend(str(ADDIN_DIR))
    return app, panels


def _import(package, module):
    for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
        sys.modules.pop(name, None)
    return importlib.import_module(f"{package}.{module}")


@pytest.fixture
def addin(monkeypatch):
    app, panels = fake_fusion(monkeypatch, ["SketchCreatePanel", "InsertPanel"])
    module = _import("SplineWire", "SplineWire")
    yield module, app, panels
    module.palette.stop(app, list(panels))            # the palette module's state lives across tests otherwise


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


# --- Add from phone (fusion/SplineWire/palette.py, docs/fusion-palette.md) -----------------------

FUSION_PAGE = Path(__file__).parent.parent / "web" / "src" / "desktop" / "fusion.ts"
CURVE = '{"v": 1, "id": "0mi4x2-ab12", "photo": "IMG_0042.jpg", "points_mm": [[0, 0], [10, 5], [20, 3]], "how": "%s"}'


@pytest.fixture
def phone(addin):
    """The add-in started, with its palette shown by Add from phone."""
    module, app, panels = addin
    module.run({})
    app.userInterface.commandDefinitions.itemById(module.palette.SHOW_ID).execute()
    return module.palette, app, panels


def _request(app, action, data, palette_id="SplineWirePhone"):
    """The page's adsk.fusionSendData(action, data): the add-in's returnData, parsed."""
    import json
    args = types.SimpleNamespace(action=action, data=data, returnData="")
    app.userInterface.palettes.itemById(palette_id).incomingFromHTML.fire(args)
    return json.loads(args.returnData)


def _pushed(app, palette_id="SplineWirePhone"):
    import json
    return [(a, json.loads(d)) for a, d in app.userInterface.palettes.itemById(palette_id).sent]


def test_run_adds_add_from_phone_beside_paste_points_and_stop_removes_everything(addin):
    module, app, panels = addin
    module.run({})
    ui = app.userInterface
    assert ui.messages == []
    for panel in panels.values():
        assert panel.controls.itemById(module.palette.SHOW_ID).isPromoted
    assert ui.palettes.items == {}                       # the palette waits for the button
    ui.commandDefinitions.itemById(module.palette.SHOW_ID).execute()
    module.stop({})
    assert ui.palettes.items == {} and app.custom == {} and ui.commandDefinitions.items == {}
    assert all(p.controls.items == {} for p in panels.values()) and ui.workspaceActivated.handlers == []


def test_add_from_phone_shows_the_hosted_page_and_toggles_it(phone):
    palette, app, _ = phone
    ui = app.userInterface
    shown = ui.palettes.itemById(palette.PALETTE_ID)
    assert shown.htmlFileURL == "https://micahflock.github.io/spline-wire/?fusion"   # https: Origin null is refused
    assert shown.isVisible and shown.dockingState == "right" and len(shown.incomingFromHTML.handlers) == 1
    button = ui.commandDefinitions.itemById(palette.SHOW_ID)
    button.execute()
    assert ui.palettes.itemById(palette.PALETTE_ID) is shown and not shown.isVisible   # hidden, still running
    button.execute()
    assert ui.palettes.itemById(palette.PALETTE_ID) is shown and shown.isVisible


def test_status_names_the_sketch_being_edited_and_the_add_in(phone):
    palette, app, _ = phone
    assert _request(app, "status", '{"v": 1}') == {"ok": True, "v": 1, "addin": palette.VERSION, "sketch": None}
    app.activeEditObject = FakeSketch(name="Sketch3")
    assert _request(app, "status", '{"v": 1}')["sketch"] == "Sketch3"


def test_the_manifest_carries_the_add_in_version(phone):
    import json
    palette, _, _ = phone
    manifest = json.loads((ADDIN_DIR / "SplineWire" / "SplineWire.manifest").read_text(encoding="utf-8"))
    assert manifest["version"] == palette.VERSION


def test_a_curve_goes_into_the_sketch_being_edited_as_one_command(phone):
    palette, app, _ = phone
    sketch = app.activeEditObject = FakeSketch(name="Sketch3")
    assert _request(app, "insert", CURVE % "auto") == {"ok": True, "queued": True}
    assert sketch.splines == []                          # nothing inside the palette's HTML event
    _deliver(app)
    assert len(app.userInterface.commandDefinitions.itemById(palette.INTO_ID).runs) == 1   # one undo step
    assert [(p.geometry.x, p.geometry.y) for p in sketch.points] == [(0, 0), (1.0, 0.5), (2.0, 0.3)]   # mm -> cm
    assert len(sketch.splines) == 1 and app.activeProduct.created == []
    assert _pushed(app) == [("inserted", {"v": 1, "id": "0mi4x2-ab12", "ok": True, "sketch": "Sketch3", "points": 3})]


def test_an_arriving_curve_waits_when_no_sketch_is_open(phone):
    _, app, _ = phone
    assert _request(app, "insert", CURVE % "auto") == {"ok": True, "queued": False, "sketch": None}
    assert app.fired == [] and app.activeProduct.created == []


def test_the_sketch_closed_before_the_command_ran(phone):
    _, app, _ = phone
    app.activeEditObject = FakeSketch()
    _request(app, "insert", CURVE % "auto")
    app.activeEditObject = None
    _deliver(app)
    [(action, reply)] = _pushed(app)
    assert reply["ok"] is False and "No sketch" in reply["error"] and app.activeProduct.created == []


def test_insert_with_no_sketch_asks_for_a_plane_xy_if_none(phone):
    palette, app, _ = phone
    assert _request(app, "insert", CURVE % "ask") == {"ok": True, "queued": True}
    _deliver(app)
    [run] = app.userInterface.commandDefinitions.itemById(palette.NEW_ID).runs
    [pick] = run.inputs
    assert pick.filters == ["ConstructionPlanes", "PlanarFaces"] and pick.limits == (0, 1)
    [made] = app.activeProduct.created
    assert made.plane == "XY" and made.name == "Spline Wire · IMG_0042" and len(made.splines) == 1
    assert _pushed(app)[0][1]["sketch"] == "Spline Wire · IMG_0042"


def test_insert_on_a_picked_face(phone):
    _, app, _ = phone
    app.dialog["pick"] = ["face 7"]
    _request(app, "insert", CURVE % "ask")
    _deliver(app)
    assert app.activeProduct.created[0].plane == "face 7"


def test_insert_with_a_sketch_open_goes_into_it_without_a_dialog(phone):
    palette, app, _ = phone
    sketch = app.activeEditObject = FakeSketch(name="Sketch3")
    _request(app, "insert", CURVE % "ask")
    _deliver(app)
    assert len(sketch.splines) == 1 and app.userInterface.commandDefinitions.itemById(palette.NEW_ID).runs == []


def test_a_cancelled_dialog_inserts_nothing_and_says_so(phone):
    _, app, _ = phone
    app.dialog["cancel"] = True
    _request(app, "insert", CURVE % "ask")
    _deliver(app)
    [(action, reply)] = _pushed(app)
    assert reply == {"v": 1, "id": "0mi4x2-ab12", "ok": False, "error": "Not inserted: the dialog was cancelled.",
                     "cancelled": True}
    assert app.activeProduct.created == []


def test_a_failed_insert_is_undone_and_reported(phone):
    palette, app, _ = phone
    app.activeEditObject = FakeSketch()
    _request(app, "insert", CURVE % "auto")
    app.activeProduct = object()                         # not a design any more
    _deliver(app)
    [run] = app.userInterface.commandDefinitions.itemById(palette.INTO_ID).runs
    assert run.args.executeFailed and "Open a design" in run.args.executeFailedMessage
    assert "Open a design" in _pushed(app)[0][1]["error"]


@pytest.mark.parametrize("action, data, message", [
    ("delete", '{"v": 1}', "doesn't know"),
    ("status", None, "too much data, or none"),
    ("status", "alert(1)", "isn't JSON"),
    ("status", '{"v": 2}', "don't match"),
    ("status", "[1]", "don't match"),
    ("insert", '{"v": 1, "pad": "' + "x" * 300000 + '"}', "too much data"),
    ("insert", (CURVE % "auto").replace('"0mi4x2-ab12"', '"../etc"'), "no usable id"),
    ("insert", (CURVE % "auto").replace('"0mi4x2-ab12"', "7"), "no usable id"),
    ("insert", CURVE % "now", "doesn't know"),
    ("insert", (CURVE % "auto").replace("[[0, 0], [10, 5], [20, 3]]", "[[0, 0]]"), "2 to 2000"),
    ("insert", (CURVE % "auto").replace("[10, 5]", "[NaN, 5]"), "2 to 2000"),
    ("insert", (CURVE % "auto").replace("[10, 5]", "[1e6, 5]"), "2 to 2000"),
    ("insert", (CURVE % "auto").replace("[10, 5]", "[true, 5]"), "2 to 2000"),
    ("insert", (CURVE % "auto").replace("[10, 5]", '["10", 5]'), "2 to 2000"),
])
def test_requests_are_checked_before_anything_happens(phone, action, data, message):
    _, app, _ = phone
    app.activeEditObject = FakeSketch()
    reply = _request(app, action, data)
    assert reply["ok"] is False and message in reply["error"] and "Traceback" not in reply["error"]
    assert app.fired == []


def test_a_photo_name_is_cleaned_for_the_sketch(phone):
    palette, app, _ = phone
    _request(app, "insert", (CURVE % "ask").replace("IMG_0042.jpg", r"a\u0000b\nc.jpeg"))
    _deliver(app)
    assert app.activeProduct.created[0].name == "Spline Wire · a b c"
    assert palette.sketch_name("") == "Spline Wire"


def test_too_many_curves_waiting_is_refused(phone):
    palette, app, _ = phone
    app.activeEditObject = FakeSketch()
    for i in range(palette.MAX_PENDING):
        assert _request(app, "insert", (CURVE % "auto").replace("ab12", f"n{i}"))["queued"]
    assert "busy" in _request(app, "insert", CURVE % "auto")["error"]
    _deliver(app)
    assert len(app.activeEditObject.splines) == palette.MAX_PENDING


def test_the_palette_comes_back_after_a_workspace_switch_deletes_it(phone):
    palette, app, _ = phone
    ui = app.userInterface
    ui.workspaceActivated.fire()                         # kept (as in S6)
    assert len(ui.palettes.itemById(palette.PALETTE_ID).incomingFromHTML.handlers) == 1
    ui.palettes.items.clear()                            # deleted
    ui.workspaceActivated.fire()
    again = ui.palettes.itemById(palette.PALETTE_ID)
    assert again.isVisible and len(again.incomingFromHTML.handlers) == 1
    again.closed.fire()                                  # the person closed it: it comes back hidden
    ui.palettes.items.clear()
    ui.workspaceActivated.fire()
    assert not ui.palettes.itemById(palette.PALETTE_ID).isVisible


def test_no_palette_before_add_from_phone_is_pressed(addin):
    module, app, _ = addin
    module.run({})
    app.userInterface.workspaceActivated.fire()
    assert app.userInterface.palettes.items == {}


def test_the_page_and_the_add_in_agree_on_the_contract(phone):
    import re
    palette, _, _ = phone
    page = FUSION_PAGE.read_text(encoding="utf-8")
    from SplineWire import curvedata
    assert re.search(rf"FUSION_PROTOCOL = {curvedata.PROTOCOL};", page)
    for name in (*curvedata.ACTIONS, "inserted"):
        assert f'"{name}"' in page
    assert "?fusion" in palette.PAGE_URL


# --- the palette spike (fusion/SplineWirePaletteSpike/, docs/fusion-palette.md) -------------

SPIKE_PAGE = Path(__file__).parent.parent / "web" / "spikes" / "fusion-palette" / "main.ts"


@pytest.fixture
def spike(monkeypatch, tmp_path):
    app, panels = fake_fusion(monkeypatch, ["SolidScriptsAddinsPanel"])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    module = _import("SplineWirePaletteSpike", "SplineWirePaletteSpike")
    module.run({})
    yield module, app, panels["SolidScriptsAddinsPanel"]
    if module._events:
        module.stop({})


def _ask(app, action, data):
    return _request(app, action, data, "SplineWirePaletteSpike")


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
    assert reply["ok"] and reply["nonce"] == "abc" and reply["fusion"] == "2705.1.30"
    assert reply["sketch"] == "Sketch3" and reply["palettes_made"] == 1 and reply["started"]


def test_spike_insert_runs_as_a_command_and_reports_back(spike):
    module, app, _ = spike
    sketch = app.activeEditObject = FakeSketch()
    sketch.name = "Sketch1"
    assert _ask(app, "spike-insert", '{"v": 1, "points_mm": [[0, 0], [20, 10], [40, 0]]}') == {"ok": True}
    assert sketch.splines == []                    # nothing inside the HTML event itself
    _deliver(app)
    assert app.userInterface.commandDefinitions.itemById(module.CMD_INSERT).runs.__len__() == 1      # one command: one undo step
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
