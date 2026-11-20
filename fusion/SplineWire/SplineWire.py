"""Spline Wire add-in for Autodesk Fusion: measured curves into a sketch.

Adds two buttons to the sketch Create panel and to the Insert panel:

- Add from phone (palette.py) shows a palette with the Spline Wire page: pair a phone once, and
  each curve measured on it goes into the sketch being edited.
- Paste points reads the table the desktop page's Copy points puts on the clipboard.

Paste points reads that table (or any pasted x/y table in mm) and adds, in the sketch
being edited (or a new sketch on the XY plane), as Add from phone does (sketch.py):

- a sketch point at every curve point, and
- a fitted spline through those points (delete it to keep only points).

Everything lands as one undo step. Fusion's API works in centimeters, so
the millimeter values are divided by 10.
"""
import traceback

import adsk.core
import adsk.fusion

from . import curvedata, palette, sketch

CMD_ID = "SplineWirePastePoints"
CMD_NAME = "Paste points"
CMD_TOOLTIP = ("Paste the points copied from Spline Wire into the sketch being edited "
               "(or a new sketch on the XY plane), with a spline through them.")
PANELS = ["SketchCreatePanel", "InsertPanel"]

_handlers = []   # Fusion only holds weak references to event handlers


def run(context):
    ui = adsk.core.Application.get().userInterface
    try:
        cmd_def = ui.commandDefinitions.itemById(CMD_ID) or \
            ui.commandDefinitions.addButtonDefinition(CMD_ID, CMD_NAME, CMD_TOOLTIP)
        handler = _CommandCreated()
        cmd_def.commandCreated.add(handler)
        _handlers.append(handler)
        for panel_id in PANELS:
            panel = ui.allToolbarPanels.itemById(panel_id)
            if panel and not panel.controls.itemById(CMD_ID):
                control = panel.controls.addCommand(cmd_def)
                control.isPromoted = True   # show on the panel, not only in its dropdown
        palette.start(adsk.core.Application.get(), PANELS)
    except Exception:
        ui.messageBox("Spline Wire add-in failed to start:\n" + traceback.format_exc())


def stop(context):
    app = adsk.core.Application.get()
    palette.stop(app, PANELS)
    ui = app.userInterface
    for panel_id in PANELS:
        panel = ui.allToolbarPanels.itemById(panel_id)
        control = panel.controls.itemById(CMD_ID) if panel else None
        if control:
            control.deleteMe()
    cmd_def = ui.commandDefinitions.itemById(CMD_ID)
    if cmd_def:
        cmd_def.deleteMe()
    _handlers.clear()


class _CommandCreated(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        handler = _Execute()
        args.command.execute.add(handler)
        _handlers.append(handler)


class _Execute(adsk.core.CommandEventHandler):
    def notify(self, args):
        app = adsk.core.Application.get()
        try:
            paste_points(app, curvedata.read_clipboard())
        except ValueError as err:
            app.userInterface.messageBox(str(err), CMD_NAME)
        except Exception:
            app.userInterface.messageBox("Paste points failed:\n" + traceback.format_exc(), CMD_NAME)


def paste_points(app, clipboard_text):
    """Add the clipboard's points (mm) and a spline through them to a sketch."""
    points_mm, corners = curvedata.parse_table(clipboard_text)
    sketch.design_of(app)
    target = sketch.active_sketch(app) or sketch.new_sketch(app)
    return sketch.add_curve(target, points_mm, corners)
