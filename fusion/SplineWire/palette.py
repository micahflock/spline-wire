"""Add from phone: curves measured on a paired phone go straight into a sketch (docs/fusion-palette.md).

A palette loads the website's desktop page in its ?fusion mode from PAGE_URL. It must be the
https page: a local file sends Origin null, which the relay refuses. The page pairs with the phone,
receives and keeps the curves, and asks this module to insert them; this module only answers and
inserts. The contract, version curvedata.PROTOCOL, is written out in web/src/desktop/fusion.ts:

  page -> add-in   adsk.fusionSendData(action, json) -> returnData (json)
    "status" {v}                                -> {ok, v, addin, sketch: name | null}
    "insert" {v, id, photo, points_mm, how}     -> {ok, queued: true}; for how "auto" with no
                                                   sketch being edited, {ok, queued: false, sketch: null}
    anything refused                            -> {ok: false, error: <a message for a person>}
  add-in -> page   palette.sendInfoToHTML("inserted", json)
    {v, id, ok: true, sketch, points} | {v, id, ok: false, error, cancelled}

Inserts run as commands (a custom event, then a command's execute()), never inside the palette's
HTML event: each is one undo step (checked in Fusion, S6), and no API call waits on the palette.
With a sketch being edited the curve goes into it with no dialog; otherwise, when the person
pressed Insert, a dialog asks for a plane or planar face (XY if none) for a new sketch.
"""
import json
import os
import traceback

import adsk.core
import adsk.fusion

from . import curvedata, sketch

VERSION = "0.2.0"          # the manifest's; the page shows it
PAGE_URL = "https://micahflock.github.io/spline-wire/?fusion"
PALETTE_ID, PALETTE_NAME = "SplineWirePhone", "Spline Wire"
SHOW_ID, SHOW_NAME = "SplineWireAddFromPhone", "Add from phone"
SHOW_TOOLTIP = ("Show or hide the Spline Wire palette. Pair a phone there once; curves measured on it "
                "then go into the sketch being edited.")
INTO_ID, NEW_ID = "SplineWireInsertIntoSketch", "SplineWireInsertNewSketch"
EVENT_ID = "SplineWireInsertCurve"
MAX_PENDING = 20

_handlers = []             # (event, handler) for the add-in's life: Fusion holds handlers weakly
_command_handlers = []     # the same for the command running now (one at a time)
_state = {"pending": {}, "current": None, "made": False, "visible": False}


def _on(event, base, fn, keep=_handlers):
    handler = type("Handler", (base,), {"notify": lambda self, args: fn(args)})()
    event.add(handler)
    keep.append((event, handler))


def _app():
    return adsk.core.Application.get()


# --- starting and stopping ----------------------------------------------------------------

def start(app, panels):
    """Add the Add from phone button to `panels` and the insert commands; the palette opens on the button."""
    ui = app.userInterface
    defs = ui.commandDefinitions
    show = defs.itemById(SHOW_ID) or defs.addButtonDefinition(SHOW_ID, SHOW_NAME, SHOW_TOOLTIP)
    _on(show.commandCreated, adsk.core.CommandCreatedEventHandler, lambda a: _on(
        a.command.execute, adsk.core.CommandEventHandler, lambda _: toggle(_app()), _fresh()))
    for cid, name, created in ((INTO_ID, "Insert the curve from the phone", _into_created),
                               (NEW_ID, "Insert the curve from the phone", _new_created)):
        cmd = defs.itemById(cid) or defs.addButtonDefinition(cid, name, "")
        _on(cmd.commandCreated, adsk.core.CommandCreatedEventHandler, created)
    for panel_id in panels:
        panel = ui.allToolbarPanels.itemById(panel_id)
        if panel and not panel.controls.itemById(SHOW_ID):
            panel.controls.addCommand(show).isPromoted = True
    try:
        app.unregisterCustomEvent(EVENT_ID)         # left over from a run that did not stop cleanly
    except Exception:
        pass
    _on(app.registerCustomEvent(EVENT_ID), adsk.core.CustomEventHandler, _on_event)
    _on(ui.workspaceActivated, adsk.core.WorkspaceEventHandler, lambda a: _restore(_app()))


def stop(app, panels):
    ui = app.userInterface
    for event, handler in _handlers + _command_handlers:
        try:
            event.remove(handler)
        except Exception:
            pass                                    # a finished command's event
    _handlers.clear()
    _command_handlers.clear()
    try:
        app.unregisterCustomEvent(EVENT_ID)
    except Exception:
        pass
    palette = ui.palettes.itemById(PALETTE_ID)
    if palette:
        palette.deleteMe()
    for panel_id in panels:
        panel = ui.allToolbarPanels.itemById(panel_id)
        control = panel.controls.itemById(SHOW_ID) if panel else None
        if control:
            control.deleteMe()
    for cid in (SHOW_ID, INTO_ID, NEW_ID):
        cmd = ui.commandDefinitions.itemById(cid)
        if cmd:
            cmd.deleteMe()
    _state.update(pending={}, current=None, made=False, visible=False)


# --- the palette ------------------------------------------------------------------------

def toggle(app):
    """Add from phone: show the palette (made on first use), or hide it if it is showing.
    Hidden, the page keeps running, so curves still arrive and go into the open sketch."""
    palette = app.userInterface.palettes.itemById(PALETTE_ID)
    if palette and palette.isVisible:
        palette.isVisible = False
        _state["visible"] = False
        return palette
    palette = palette or _make_palette(app)
    palette.isVisible = True
    _state["visible"] = True
    return palette


def _make_palette(app):
    palette = app.userInterface.palettes.add(PALETTE_ID, PALETTE_NAME, PAGE_URL, True, True, True, 380, 720)
    try:
        palette.dockingState = adsk.core.PaletteDockingStates.PaletteDockStateRight
    except Exception:
        pass                                        # floating is fine too
    _on(palette.incomingFromHTML, adsk.core.HTMLEventHandler, _incoming)
    _on(palette.closed, adsk.core.UserInterfaceGeneralEventHandler, lambda a: _state.update(visible=False))
    _state["made"] = True
    return palette


def _restore(app):
    """Palettes may be deleted by a workspace switch (it kept it in S6, on Windows): make it again."""
    if _state["made"] and not app.userInterface.palettes.itemById(PALETTE_ID):
        palette = _make_palette(app)
        palette.isVisible = _state["visible"]


def _incoming(args):
    html = adsk.core.HTMLEventArgs.cast(args)
    try:
        reply = answer(_app(), html.action, html.data)
    except ValueError as err:
        reply = {"ok": False, "error": str(err)}
    except Exception:
        reply = {"ok": False, "error": "The Spline Wire add-in failed:\n" + traceback.format_exc()}
    html.returnData = json.dumps(reply)


def answer(app, action, data):
    """The reply to one request from the page. Raises ValueError with a message for a person."""
    msg = curvedata.parse_request(action, data)
    if action == "status":
        current = sketch.active_sketch(app)
        return {"ok": True, "v": curvedata.PROTOCOL, "addin": VERSION, "sketch": current.name if current else None}
    req = curvedata.parse_insert(msg)
    if req["how"] == "auto" and not sketch.active_sketch(app):
        return {"ok": True, "queued": False, "sketch": None}
    pending = _state["pending"]
    if len(pending) >= MAX_PENDING and req["id"] not in pending:
        raise ValueError("Fusion is still busy with earlier curves: try again in a moment.")
    pending[req["id"]] = req
    app.fireCustomEvent(EVENT_ID, req["id"])        # inserted by a command, after this event
    return {"ok": True, "queued": True}


# --- inserting --------------------------------------------------------------------------

def _on_event(args):
    app = _app()
    req = _state["pending"].pop(args.additionalInfo, None)
    if req is None:
        return
    if sketch.active_sketch(app):
        cid = INTO_ID
    elif req["how"] == "ask":
        cid = NEW_ID
    else:
        _report(app, req, error="No sketch is being edited any more: open one, or press Insert.")
        return
    _state["current"] = req
    app.userInterface.commandDefinitions.itemById(cid).execute()


def _fresh():
    """The handler list for a command just created: the last command's handlers are done with."""
    _command_handlers.clear()
    return _command_handlers


def _into_created(args):
    """Into the sketch being edited: no inputs, so Fusion runs it at once."""
    req, _state["current"] = _state["current"], None
    _on(args.command.execute, adsk.core.CommandEventHandler,
        lambda a: _insert(a, req, lambda: sketch.active_sketch(_app()) or sketch.new_sketch(_app())), _fresh())


def _new_created(args):
    """A new sketch: a dialog asks for a plane or planar face, XY if none is picked."""
    req, _state["current"] = _state["current"], None
    cmd = args.command
    pick = cmd.commandInputs.addSelectionInput("plane", "Plane", "A plane or planar face for the new sketch (XY if none)")
    pick.addSelectionFilter("ConstructionPlanes")
    pick.addSelectionFilter("PlanarFaces")
    pick.setSelectionLimits(0, 1)
    ran = []

    def execute(a):
        ran.append(True)
        plane = pick.selection(0).entity if pick.selectionCount else None
        _insert(a, req, lambda: sketch.new_sketch(_app(), plane, sketch_name(req["photo"] if req else "")))

    keep = _fresh()
    _on(cmd.execute, adsk.core.CommandEventHandler, execute, keep)
    _on(cmd.destroy, adsk.core.CommandEventHandler,
        lambda a: ran or _report(_app(), req, error="Not inserted: the dialog was cancelled.", cancelled=True), keep)


def sketch_name(photo):
    """'Spline Wire · IMG_0042' for a new sketch."""
    stem = os.path.splitext(photo or "")[0].strip()[:60]
    return f"{sketch.SKETCH_NAME} · {stem}" if stem else sketch.SKETCH_NAME


def _insert(args, req, target):
    app = _app()
    if req is None:
        return
    try:
        sketch.design_of(app)
        made = target()
        sketch.add_curve(made, req["points"])
        _report(app, req, into=made.name)
    except Exception as err:
        message = str(err) if isinstance(err, ValueError) else "Inserting failed:\n" + traceback.format_exc()
        args.executeFailed = True                  # Fusion undoes what the command did
        args.executeFailedMessage = message
        _report(app, req, error=message)


def _report(app, req, into=None, error=None, cancelled=False):
    """Tell the page how an insert ended."""
    if req is None:
        return
    reply = {"v": curvedata.PROTOCOL, "id": req["id"]}
    reply.update({"ok": True, "sketch": into, "points": len(req["points"])} if error is None
                 else {"ok": False, "error": error, "cancelled": cancelled})
    palette = app.userInterface.palettes.itemById(PALETTE_ID)
    if palette:
        palette.sendInfoToHTML("inserted", json.dumps(reply))
