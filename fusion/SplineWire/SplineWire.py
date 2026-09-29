"""Spline Wire add-in for Autodesk Fusion: paste measured points into a sketch.

Adds a "Paste points" button to the sketch Create panel and to the Insert
panel. It reads the point table that SplineWire.exe's "Copy points" puts on
the clipboard (or any pasted x/y table in mm) and adds, in the sketch being
edited (or a new sketch on the XY plane):

- a sketch point at every curve point, and
- a fitted spline through those points (delete it to keep only points).

Everything lands as one undo step. Fusion's API works in centimeters, so
the millimeter values are divided by 10.
"""
import traceback

import adsk.core
import adsk.fusion

from . import curvedata

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
    except Exception:
        ui.messageBox("Spline Wire add-in failed to start:\n" + traceback.format_exc())


def stop(context):
    ui = adsk.core.Application.get().userInterface
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
    points_mm = curvedata.parse_points(clipboard_text)
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise ValueError("Open a design (not a drawing or CAM setup) before pasting points.")

    sketch = adsk.fusion.Sketch.cast(app.activeEditObject)
    if not sketch:
        root = design.rootComponent
        sketch = root.sketches.add(root.xYConstructionPlane)
        sketch.name = "Spline Wire"

    sketch_points = adsk.core.ObjectCollection.create()
    coordinates = adsk.core.ObjectCollection.create()
    for x_mm, y_mm in points_mm:
        xyz = adsk.core.Point3D.create(x_mm / 10.0, y_mm / 10.0, 0.0)
        sketch_points.add(sketch.sketchPoints.add(xyz))
        coordinates.add(xyz)
    # The pasted points stay even if the spline is deleted. Preferably they
    # are also the spline's own fit points; the documented form takes plain
    # coordinates, which leaves a second, coincident point at each spot.
    splines = sketch.sketchCurves.sketchFittedSplines
    try:
        splines.add(sketch_points)
    except Exception:
        splines.add(coordinates)
    return sketch
