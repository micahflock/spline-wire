"""Putting a curve into a sketch: shared by Paste points (SplineWire.py) and Add from phone (palette.py)."""
import adsk.core
import adsk.fusion

SKETCH_NAME = "Spline Wire"


def design_of(app):
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise ValueError("Open a design (not a drawing or CAM setup), then try again.")
    return design


def active_sketch(app):
    """The sketch being edited, or None."""
    return adsk.fusion.Sketch.cast(app.activeEditObject)


def new_sketch(app, plane=None, name=SKETCH_NAME):
    """A new sketch in the root component, on `plane` (a construction plane or planar face) or XY."""
    root = design_of(app).rootComponent
    sketch = root.sketches.add(plane or root.xYConstructionPlane)
    sketch.name = name
    return sketch


def add_curve(sketch, points_mm):
    """A sketch point at each point (mm) and a fitted spline through them. Fusion's API works in cm."""
    sketch_points = adsk.core.ObjectCollection.create()
    coordinates = adsk.core.ObjectCollection.create()
    for x_mm, y_mm in points_mm:
        xyz = adsk.core.Point3D.create(x_mm / 10.0, y_mm / 10.0, 0.0)
        sketch_points.add(sketch.sketchPoints.add(xyz))
        coordinates.add(xyz)
    # The points stay even if the spline is deleted. Preferably they are also the
    # spline's own fit points; the documented form takes plain coordinates, which
    # leaves a second, coincident point at each spot.
    splines = sketch.sketchCurves.sketchFittedSplines
    try:
        splines.add(sketch_points)
    except Exception:
        splines.add(coordinates)
    return sketch
