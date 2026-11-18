"""The snap-fit link's dimensions, read from its STEP file.

The link is designed by hand in Fusion; data/chain-snap.yaml and the snap
renderer (synthetic.render_snap_photo) take their numbers from it. This
reads the solid and prints what they need: the pitch, the rim, the heights
of the head and the sleeve below the rim, the radius that touches the
target at the pin (the head) and at the sleeve, and the joint's range, found
by turning a second link about the pin until the two solids overlap.

    uv run --with cadquery-ocp python experiments/snap_link_geometry.py [file.step]

The default file is experiments/snap_link/3DP_Link_Ver_MF.step (2026-11-17).
The link lies with its pin axis and sleeve axis along x, the pin's at the
larger x, both vertical, the part standing on z = its lowest point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh

STEP = Path(__file__).parent / "snap_link" / "3DP_Link_Ver_MF.step"


def load(path: Path) -> trimesh.Trimesh:
    """The STEP solid as a fine triangle mesh (OpenCascade, via cadquery-ocp)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.005, False, 0.05, True)
    verts, faces = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr, base = loc.Transformation(), len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tr)
                verts.append((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    a, b = b, a
                faces.append((base + a - 1, base + b - 1, base + c - 1))
        explorer.Next()
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.merge_vertices()
    return mesh


def axes(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """(pin axis, sleeve axis) in plan: the rim's centre, and the centre of
    the circle through the sleeve's outer edge at its foot, on the side away
    from the arm."""
    z0, z1 = mesh.bounds[:, 2]
    rim = _section(mesh, z1 - 0.05)
    pin = np.array(rim.centroid.coords[0])
    foot = _section(mesh, z0 + 0.6)
    pts = np.concatenate([np.array(g.exterior.coords) for g in getattr(foot, "geoms", [foot])])
    far = pts[pts[:, 0] < pts[:, 0].min() + 1.5]              # the outer arc only, away from the arm
    A = np.c_[2 * far, np.ones(len(far))]
    cx, cy, _ = np.linalg.lstsq(A, (far ** 2).sum(1), rcond=None)[0]
    return pin, np.array([cx, cy])


def _section(mesh: trimesh.Trimesh, z: float):
    """The solid's cross-section at height z, as shapely polygons (each
    closed outline flips inside and outside, so holes stay holes)."""
    segments = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z])[:, :, :2]
    merged = shapely.line_merge(shapely.union_all(shapely.multilinestrings(np.round(segments, 9))))
    region = shapely.Polygon()
    for ring in shapely.get_parts(merged):
        if ring.is_ring:
            region = region.symmetric_difference(shapely.Polygon(ring.coords))
    return region


def radius(mesh: trimesh.Trimesh, axis: np.ndarray, z: float, away_from: np.ndarray | None = None) -> float:
    """The largest distance from the axis to the solid's outline at height z
    (only on the side facing away from `away_from`, to leave out the arm)."""
    sec = _section(mesh, z)
    pts = np.concatenate([np.array(g.exterior.coords) for g in getattr(sec, "geoms", [sec])])
    if away_from is not None:
        d = axis - away_from
        pts = pts[(pts - axis) @ d > 0]
    near = np.linalg.norm(pts - axis, axis=1)
    near = near[near < 6.0]
    return float(near.max()) if len(near) else 0.0


def joint_range(mesh: trimesh.Trimesh, pin: np.ndarray, sleeve: np.ndarray) -> tuple[float, float]:
    """How far the joint turns each way before two links overlap (degrees):
    a second link with its sleeve on this link's pin, turned about it."""
    shift = trimesh.transformations.translation_matrix([*(pin - sleeve), 0.0])

    def overlap(deg: float) -> float:
        turn = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1], point=[*pin, 0.0])
        other = mesh.copy()
        other.apply_transform(turn @ shift)
        both = trimesh.boolean.intersection([mesh, other], engine="manifold")
        return both.volume if len(both.vertices) else 0.0

    def limit(sign: int) -> float:
        lo = 0.0
        hi = next(d for d in range(10, 181, 10) if overlap(sign * d) > 1e-3)
        lo = hi - 10.0
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if overlap(sign * mid) > 1e-3 else (mid, hi)
        return lo

    return limit(+1), limit(-1)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else STEP
    mesh = load(path)
    z0, z1 = mesh.bounds[:, 2]
    pin, sleeve = axes(mesh)
    print(f"{path.name}: {mesh.volume:.0f} mm3, plan {np.ptp(mesh.bounds[:, 0]):.2f} x {np.ptp(mesh.bounds[:, 1]):.2f} mm, "
          f"height {z1 - z0:.2f} mm")
    print(f"pitch (pin axis to sleeve axis): {np.linalg.norm(pin - sleeve):.3f} mm")

    # The rim: the top 0.4 mm, a ring about the pin.
    top = _section(mesh, z1 - 0.05)
    outer = 2 * float(np.max(np.linalg.norm(np.array(top.exterior.coords) - pin, axis=1)))
    inner = 2 * float(np.min([np.linalg.norm(np.array(i.coords) - pin, axis=1).min() for i in top.interiors]))
    zs = np.round(np.arange(z0 + 0.01, z1, 0.01), 3)
    solid_top = next(z for z in zs[::-1] if _section(mesh, z).area > 1.5 * top.area)
    print(f"rim: {outer:.3f} mm outside, {inner:.3f} inside ({inner / outer:.3f}), band {(outer - inner) / 2:.3f} mm, "
          f"{z1 - solid_top:.2f} mm tall")

    # Radii about each axis, by depth below the rim's top (every 0.05 mm, printed every 0.5).
    print("depth below the rim top: head radius (about the pin), sleeve radius (about the sleeve, away from the arm)")
    rows = []
    for depth in np.arange(z1 - solid_top + 0.01, z1 - z0, 0.05):
        z = z1 - depth
        rows.append((depth, radius(mesh, pin, z, away_from=sleeve), radius(mesh, sleeve, z, away_from=pin)))
    for depth, h, s in rows[::10]:
        print(f"  {depth:5.2f}  {h:5.3f}  {s:5.3f}")
    sleeve_top = min(d for d, _, s in rows if s > 0)
    print(f"widest: head {max(h for _, h, _ in rows):.3f} mm (at the rim), sleeve {max(s for _, _, s in rows):.3f} mm; "
          f"sleeve top {sleeve_top:.2f} mm below the rim top, the part's foot {z1 - z0:.2f} mm")

    plus, minus = joint_range(mesh, pin, sleeve)
    print(f"joint range: +{plus:.1f} / -{minus:.1f} degrees before the links overlap")


if __name__ == "__main__":
    main()
