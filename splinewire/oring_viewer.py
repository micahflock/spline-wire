"""Assembly views of the O-ring chain: an interactive 3D page and a glTF model.

The page (oring_viewer.html filled in here) shows the two-joint test piece
built from the same solids as the STLs: explode it along the screw axis,
bend each joint to its stop, cut it open through the pins, and read the
fits. It loads three.js from a CDN, so it needs a network connection; the
geometry is embedded. The .glb is the written chain, gently bent, one
coloured mesh per part, for any glTF viewer or to import into CAD.
"""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np

from splinewire.chain import ChainSpec
from splinewire.oring_chain import DEFAULT_PARAMS, OringParams, chain_parts, joint_geometry, posed

TEMPLATE = Path(__file__).with_name("oring_viewer.html")
COLOURS = {"outer": (70, 110, 160), "button": (70, 110, 160), "inner": (214, 140, 52),
           "screw": (150, 158, 168), "washer": (196, 202, 209), "oring": (192, 57, 43)}


def _mesh_arrays(solid) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh()
    return (np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3],
            np.asarray(mesh.tri_verts, dtype=np.uint32))


def viewer_data(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS) -> dict:
    """The two-joint piece's geometry (each distinct solid once, in its own
    frame, base64) and which part uses which."""
    j = joint_geometry(spec, p)
    parts = chain_parts(spec, p, n_pins=4)
    geoms, names, items = {}, {}, []
    for pt in parts:
        if id(pt.solid) not in names:
            v, f = _mesh_arrays(pt.solid)
            small = len(v) < 65536
            name = f"g{len(geoms)}"
            names[id(pt.solid)] = name
            geoms[name] = {"pos": base64.b64encode(v.tobytes()).decode(),
                           "idx": base64.b64encode((f.astype(np.uint16) if small else f).tobytes()).decode(),
                           "u32": not small}
        items.append({"kind": pt.kind, "pins": list(pt.pins), "geom": names[id(pt.solid)]})
    return {"geoms": geoms, "items": items, "meta": {"pitch": j.pitch, "stop": j.stop_deg, "top": j.top, "n": 4}}


def fit_rows(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS) -> list[tuple[str, str, str]]:
    """(where, mating sizes, result) for every fit that decides whether the
    joint goes together and works."""
    j = joint_geometry(spec, p)
    sleeve, hole = p.sleeve_mm, 2 * j.r_hole
    stretch = (sleeve - p.oring_id_mm) / p.oring_id_mm
    th = math.radians(j.stop_deg)
    gap = j.pitch - 2 * j.R
    return [
        ("O-ring on the sleeve", f"ID {p.oring_id_mm:.1f} on Ø{sleeve:.1f}",
         "slides on, no stretch" if stretch <= 0 else f"stretches {100 * stretch:.0f}%"),
        ("Washer on the sleeve", f"Ø{p.washer_id_mm:.1f} on Ø{sleeve:.1f}",
         f"{(p.washer_id_mm - sleeve) / 2:.2f} mm play, free"),
        ("Inner link hole on the sleeve", f"Ø{hole:.1f} on Ø{sleeve:.1f}",
         f"{j.r_hole - sleeve / 2:.2f} mm gap, never touches"),
        ("Screw head in the countersink", f"Ø{p.head_d_mm:.1f} head, 90°, in Ø{2 * j.r_cs:.1f} at the face",
         f"seats on the cone, {p.head_recess_mm:g} mm recessed"),
        ("Screw in the sleeve's hole", f"M{p.screw_d_mm:g} in Ø{p.pilot_mm:g}",
         f"cuts its own thread, {(sleeve - p.pilot_mm) / 2:.2f} mm wall"),
        ("Screw length in the sleeve", f"tip at {j.z_tip:.1f}, hole to {j.z_bore:.1f} mm",
         f"{j.z_bore - j.z_tip:.1f} mm spare"),
        ("O-ring squeeze", f"{p.squeeze_mm:g} mm into its seat", "set by the head stop"),
        ("Washer on the inner link", f"ring Ø{hole:g}–{2 * j.r_ring:g}, {p.ring_mm:g} high", "slides on the ring only"),
        ("Inner link under the outer link", f"top {p.inner_mm:g} / outer from {j.z_clear:g}",
         f"{j.z_clear - p.inner_mm:.2f} mm clear"),
        ("Stop post past the inner link ends", f"post {2 * j.post_r:g} wide in a {gap:g} gap",
         f"{(gap - 2 * j.post_r) / 2:.2f} mm each side"),
        ("Outer plates at the stop", f"pins either side {2 * j.pitch * math.sin(th / 2):.1f} apart",
         f"plates {j.pitch * math.sin(th) - 2 * j.R:.1f} mm apart"),
    ]


def viewer_html(spec: ChainSpec, p: OringParams = DEFAULT_PARAMS) -> str:
    """The interactive assembly page, complete (open it in a browser)."""
    from html import escape

    j = joint_geometry(spec, p)
    rows = "\n".join(f'            <tr><td>{escape(a)}</td><td class="num">{escape(b)}</td>'
                     f'<td class="ok">{escape(c)}</td></tr>' for a, b, c in fit_rows(spec, p))
    fill = {"ORING": f"{p.oring_id_mm:g} × {p.oring_cs_mm:g}", "TURN": f"{180 - j.stop_deg:.1f}",
            "BETWEEN": f"{j.stop_deg:.1f}", "BETWEEN_ROUND": f"{j.stop_deg:.0f}", "PILOT": f"{p.pilot_mm:g}",
            "SLEEVE": f"{p.sleeve_mm:.1f}", "WASHER_ID": f"{p.washer_id_mm:g}", "HOLE": f"{2 * j.r_hole:g}"}
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fill.items():
        page = page.replace("{{" + key + "}}", value)
    page = page.replace("<!--FIT_ROWS-->", rows.lstrip())
    page = page.replace("/*CHAIN_DATA*/", "window.CHAIN=" + json.dumps(viewer_data(spec, p)) + ";")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + page + '\n</body>\n</html>\n')


def write_assembly_glb(path: Path, spec: ChainSpec, p: OringParams = DEFAULT_PARAMS,
                       n_pins: int | None = None, bend_deg: float = 20.0) -> Path:
    """The assembled chain as a glTF binary, one coloured mesh per part,
    every joint bent bend_deg the same way (a wrap around a pipe)."""
    import trimesh

    j = joint_geometry(spec, p)
    parts = chain_parts(spec, p, n_pins)
    n = max(max(pt.pins) for pt in parts) + 1
    scene = trimesh.Scene()
    for k, (pt, solid) in enumerate(zip(parts, posed(parts, j, np.full(n - 2, bend_deg)))):
        v, f = _mesh_arrays(solid)
        mesh = trimesh.Trimesh(v, f, process=False)
        mesh.visual.face_colors = (*COLOURS[pt.kind], 255)
        scene.add_geometry(mesh, node_name=f"{pt.kind}-{'-'.join(map(str, pt.pins))}-{k}")
    path = Path(path)
    scene.export(path)
    return path
