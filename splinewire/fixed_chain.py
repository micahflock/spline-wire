"""Fixed-chain test plaque: the snap-fit chain posed and fused into one print
(docs/real-world-tests.md, build step 3).

The real link (data/snap-link.stl.gz, from its STEP file by
experiments/snap_link_geometry.py --mesh) is placed once per link in one of
the plaque poses of splinewire.targets, and the joints are fused, so the
print looks to the camera exactly like the chain (rims, heads, sleeves and
their relief) while its pins sit exactly where the truth file says.

Link j has its pin, head and rim at pin j and its sleeve at pin j + 1,
round the next link's pin; the first pin has no sleeve, and the last
link's sleeve, one pitch past the last rim, has no rim. A short cylinder
fills each pin's 0.01 mm gap to the sleeve round it, so the joints fuse.

One filament change at the rims' foot (12.65 mm) gives the chain's colours:
black then white, or white then orange.

Needs trimesh and manifold3d (the dev dependency group).
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec, _data_dir
from splinewire.targets import PLAQUE_POSES, plaque_json, plaque_pins

LINK_MESH = "snap-link.stl.gz"
RIM_MM = 0.4                 # the rims' height above the head tops (the colour change)
FUSE_RADIUS_MM = 3.0         # the pin's widest; the sleeve's bore is 0.01 mm larger
FUSE_Z_MM = (0.2, 7.9)       # within the sleeve (its top is 7.98 mm above the table)
BAR_MM = (50.0, 6.0, 2.0)    # the scale bar: length, width, height


def load_link():
    """The link as a trimesh: pin axis on the origin, sleeve axis at (-pitch, 0), foot on z = 0."""
    import trimesh

    data = gzip.decompress((_data_dir() / LINK_MESH).read_bytes())
    return trimesh.load(io.BytesIO(data), file_type="stl")


def plaque_mesh(pose: str, spec: ChainSpec, n_links: int | None = None):
    """The fused chain in this pose, as one trimesh, pins where plaque_pins puts them."""
    import trimesh

    n = n_links or spec.n_pins
    pins = plaque_pins(pose, spec, n)
    ends = np.vstack([pins, pins[-1] + (pins[-1] - pins[-2])])     # the last link's sleeve, straight on
    link = load_link()
    parts = []
    for j in range(n):
        u = ends[j + 1] - ends[j]
        angle = np.arctan2(u[1], u[0]) - np.pi                     # the link's sleeve lies along -x
        m = link.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
        m.apply_translation([ends[j][0], ends[j][1], 0.0])
        parts.append(m)
    lo, hi = FUSE_Z_MM
    for p in pins[1:]:
        c = trimesh.creation.cylinder(radius=FUSE_RADIUS_MM, height=hi - lo, sections=64)
        c.apply_translation([p[0], p[1], (lo + hi) / 2])
        parts.append(c)
    return trimesh.boolean.union(parts, engine="manifold")


def split_colours(mesh):
    """(below the rims' foot, the rims): for a printer with two filaments."""
    import trimesh

    (x0, y0, z0), (x1, y1, z1) = mesh.bounds
    z = z1 - RIM_MM

    def box(lo: float, hi: float):
        b = trimesh.creation.box(extents=[x1 - x0 + 2, y1 - y0 + 2, hi - lo])
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (lo + hi) / 2])
        return b

    body = trimesh.boolean.intersection([mesh, box(z0 - 1, z)], engine="manifold")
    rims = trimesh.boolean.intersection([mesh, box(z, z1 + 1)], engine="manifold")
    return body, rims


def scale_bar():
    import trimesh

    bar = trimesh.creation.box(extents=BAR_MM)
    bar.apply_translation([0, 0, BAR_MM[2] / 2])
    return bar


def write_fixed_chain(out: Path, pose: str, spec: ChainSpec, n_links: int | None = None) -> dict[str, Path]:
    """STLs (whole, body, rims), the scale bar, the truth file and printing notes."""
    if pose not in PLAQUE_POSES:
        raise ValueError(f"no plaque pose {pose!r}; there are {', '.join(PLAQUE_POSES)}")
    n = n_links or spec.n_pins
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{pose}-{n}"
    mesh = plaque_mesh(pose, spec, n)
    body, rims = split_colours(mesh)
    paths = {
        "plaque": out / f"{name}.stl",
        "body": out / f"{name}-body.stl",
        "rims": out / f"{name}-rims.stl",
        "bar": out / "scale-bar-50mm.stl",
        "truth": out / f"{name}-truth.json",
        "instructions": out / f"{name}-PRINTING.txt",
    }
    mesh.export(paths["plaque"])
    body.export(paths["body"])
    rims.export(paths["rims"])
    scale_bar().export(paths["bar"])
    truth = plaque_json(pose, spec, n)
    truth["schema_note"] = "pin_points: the rims' centres, in the STL's x/y"
    paths["truth"].write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    paths["instructions"].write_text(printing_notes(pose, n, mesh), encoding="utf-8")
    return paths


def printing_notes(pose: str, n: int, mesh) -> str:
    top = mesh.bounds[1, 2]
    size = mesh.extents
    return "\n".join([
        f"Fixed-chain test plaque: {PLAQUE_POSES[pose]}",
        f"{n} links fused in one piece, {size[0]:.0f} x {size[1]:.0f} x {size[2]:.2f} mm. "
        "Print as it lies, rims up, as the links themselves print.",
        "",
        "Colours, as your chain: one filament change at the rims' foot,",
        f"z = {top - RIM_MM:.2f} mm (the rims are the top {RIM_MM} mm):",
        "  white rims on black links: black, then white;",
        "  orange rims on white links: white, then orange.",
        f"With two filaments instead: the -body.stl and -rims.stl parts.",
        "",
        "Print scale-bar-50mm.stl in the same job, and measure it with calipers:",
        "more than 0.1 mm off 50.00 means the printer's scale is off, and every score",
        "from this print is read after the scale fit.",
        "",
        "In the test page, pick this plaque as the target and set the link count to "
        f"{n}. The truth file holds the rims' centres.",
    ]) + "\n"
