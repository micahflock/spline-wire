"""Spline Wire palette spike (docs/fusion-palette.md, "The spike"): a throwaway, not the add-in.

Opens the hosted spike page (web/spikes/fusion-palette/) in a palette, from its https URL: a
local file would send Origin null, which the relay refuses. Answers the page's
adsk.fusionSendData(action, json) with returnData (json), for a fixed set of actions (listed in
the page's main.ts); anything else gets {"ok": false, "error": <a message for a person>}.
"spike-insert" adds a spline through a custom event and a command's execute(), so it is one
undo step, then pushes "spike-inserted" back with palette.sendInfoToHTML. The palette is made
again when a workspace switch deletes it.
"""
import datetime, json, math, os, sys, traceback

import adsk.core
import adsk.fusion

PAGE_URL = "https://micahflock.github.io/spline-wire/spikes/fusion-palette/"
PALETTE_ID, EVENT_ID = "SplineWirePaletteSpike", "SplineWirePaletteSpikeInsert"
CMD_SHOW, CMD_INSERT, PANEL = "SplineWireSpikeShow", "SplineWireSpikeInsert", "SolidScriptsAddinsPanel"
MAX_DATA, MAX_POINTS, MAX_MM = 64 * 1024, 2000, 1e5

_events = []    # (event, handler): Fusion holds handlers weakly, and stop() removes them
_state = {"made": 0, "switches": 0, "lost": 0, "points": None, "started": None}


def _on(event, base, fn):
    handler = type("Handler", (base,), {"notify": lambda self, args: fn(args)})()
    event.add(handler)
    _events.append((event, handler))


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        _state.update(made=0, switches=0, lost=0, started=datetime.datetime.now().isoformat())
        defs = ui.commandDefinitions
        show = defs.itemById(CMD_SHOW) or defs.addButtonDefinition(CMD_SHOW, "Spline Wire spike", "Open the palette spike.")
        insert = defs.itemById(CMD_INSERT) or defs.addButtonDefinition(CMD_INSERT, "Spline Wire spike: insert", "")
        for cmd, fn in ((show, lambda a: _palette(ui)), (insert, _insert)):
            _on(cmd.commandCreated, adsk.core.CommandCreatedEventHandler,
                lambda a, fn=fn: _on(a.command.execute, adsk.core.CommandEventHandler, fn))
        panel = ui.allToolbarPanels.itemById(PANEL)
        if panel and not panel.controls.itemById(CMD_SHOW):
            panel.controls.addCommand(show)
        _on(app.registerCustomEvent(EVENT_ID), adsk.core.CustomEventHandler, lambda a: insert.execute())
        _on(ui.workspaceActivated, adsk.core.WorkspaceEventHandler, lambda a: _switched(ui))
        _palette(ui)
    except Exception:
        ui.messageBox("Spline Wire spike failed to start:\n" + traceback.format_exc())


def stop(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    for event, handler in _events:
        try:
            event.remove(handler)
        except Exception:
            pass                                # a finished command's event
    _events.clear()
    app.unregisterCustomEvent(EVENT_ID)
    panel = ui.allToolbarPanels.itemById(PANEL)
    for item in (ui.palettes.itemById(PALETTE_ID), panel and panel.controls.itemById(CMD_SHOW),
                 ui.commandDefinitions.itemById(CMD_SHOW), ui.commandDefinitions.itemById(CMD_INSERT)):
        if item:
            item.deleteMe()


def _palette(ui):
    palette = ui.palettes.itemById(PALETTE_ID)
    if not palette:
        palette = ui.palettes.add(PALETTE_ID, "Spline Wire spike", PAGE_URL, True, True, True, 380, 720)
        _on(palette.incomingFromHTML, adsk.core.HTMLEventHandler, _incoming)
        _state["made"] += 1
    palette.isVisible = True


def _switched(ui):
    _state["switches"] += 1
    if _state["made"] and not ui.palettes.itemById(PALETTE_ID):
        _state["lost"] += 1
        _palette(ui)


def _incoming(args):
    html = adsk.core.HTMLEventArgs.cast(args)
    try:
        reply = answer(html.action, html.data)
    except ValueError as err:
        reply = {"ok": False, "error": str(err)}
    except Exception:
        reply = {"ok": False, "error": "The add-in failed:\n" + traceback.format_exc()}
    html.returnData = json.dumps(reply)


def answer(action, data):
    """The reply to one request from the page. Raises ValueError for a person."""
    if action not in ("spike-echo", "spike-insert", "spike-report"):
        raise ValueError(f"Unknown request {str(action)[:40]!r}: update the spike add-in.")
    if len(data or "") > MAX_DATA:
        raise ValueError("The page sent too much data.")
    try:
        msg = json.loads(data or "")
    except ValueError:
        raise ValueError("The page sent something that is not JSON.") from None
    if not isinstance(msg, dict) or msg.get("v") != 1:
        raise ValueError("The page speaks another version: update the spike add-in.")
    app = adsk.core.Application.get()
    if action == "spike-echo":
        sketch = adsk.fusion.Sketch.cast(app.activeEditObject)
        return {"ok": True, "nonce": str(msg.get("nonce"))[:64], "fusion": app.version, "os": sys.platform,
                "started": _state["started"], "sketch": sketch.name if sketch else None,
                "palettes_made": _state["made"], "switches": _state["switches"], "palette_lost": _state["lost"]}
    if action == "spike-insert":
        _state["points"] = points_mm(msg.get("points_mm"))
        app.fireCustomEvent(EVENT_ID, "")        # API calls wait for the command, not this HTML event
        return {"ok": True}
    path = os.path.join(os.path.expanduser("~"), "spline-wire-palette-spike.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(msg.get("text"))[:20000])
    return {"ok": True, "path": path}


def points_mm(value):
    """2-2000 [x, y] pairs of finite numbers within 1e5 mm, or ValueError."""
    ok = isinstance(value, list) and 2 <= len(value) <= MAX_POINTS and all(
        isinstance(p, list) and len(p) == 2 and all(
            type(c) in (int, float) and math.isfinite(c) and abs(c) <= MAX_MM for c in p) for p in value)
    if not ok:
        raise ValueError(f"The points must be 2 to {MAX_POINTS} [x, y] pairs in mm, each within {MAX_MM:g} mm.")
    return [(float(x), float(y)) for x, y in value]


def _insert(args):
    app = adsk.core.Application.get()
    points, _state["points"] = _state["points"], None
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design or not points:
            raise ValueError("Open a design, then press Insert in the palette.")
        sketch = adsk.fusion.Sketch.cast(app.activeEditObject)
        if not sketch:
            root = design.rootComponent
            sketch = root.sketches.add(root.xYConstructionPlane)
            sketch.name = "Spline Wire spike"
        fit = adsk.core.ObjectCollection.create()
        for x, y in points:
            fit.add(adsk.core.Point3D.create(x / 10.0, y / 10.0, 0.0))   # mm -> the API's cm
        sketch.sketchCurves.sketchFittedSplines.add(fit)
        reply = {"ok": True, "sketch": sketch.name, "points": len(points)}
    except ValueError as err:
        reply = {"ok": False, "error": str(err)}
    except Exception:
        reply = {"ok": False, "error": traceback.format_exc()}
    palette = app.userInterface.palettes.itemById(PALETTE_ID)
    if palette:
        palette.sendInfoToHTML("spike-inserted", json.dumps(reply))
