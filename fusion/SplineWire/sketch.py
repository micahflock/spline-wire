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


def add_curve(sketch, points_mm, corners=()):
    """A sketch point at each point (mm) and a fitted spline through them. Fusion's API works in cm.

    corners: indices of points where the curve has a sharp corner. The curve is then one
    spline per piece between corners (a line for a piece of two points), and neighbouring
    pieces end on the same sketch point, so they stay joined."""
    sketch_points = []
    coordinates = []
    for x_mm, y_mm in points_mm:
        xyz = adsk.core.Point3D.create(x_mm / 10.0, y_mm / 10.0, 0.0)
        sketch_points.append(sketch.sketchPoints.add(xyz))
        coordinates.append(xyz)
    bounds = [0, *sorted(set(i for i in corners if 0 < i < len(points_mm) - 1)), len(points_mm) - 1]
    for a, b in zip(bounds, bounds[1:]):
        if b - a == 1:
            sketch.sketchCurves.sketchLines.addByTwoPoints(sketch_points[a], sketch_points[b])
        else:
            _spline(sketch, sketch_points[a:b + 1], coordinates[a:b + 1])
    return sketch


def _spline(sketch, sketch_points, coordinates):
    # The points stay even if the spline is deleted. Preferably they are also the
    # spline's own fit points; the documented form takes plain coordinates, which
    # leaves a second, coincident point at each spot.
    splines = sketch.sketchCurves.sketchFittedSplines
    fit = adsk.core.ObjectCollection.create()
    for p in sketch_points:
        fit.add(p)
    try:
        splines.add(fit)
    except Exception:
        plain = adsk.core.ObjectCollection.create()
        for p in coordinates:
            plain.add(p)
        splines.add(plain)
