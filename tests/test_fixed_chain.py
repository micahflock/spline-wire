"""The fixed-chain test plaque: the real link, posed and fused."""
import json

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("manifold3d")

import trimesh  # noqa: E402

from splinewire.chain import chain_presets, load_chain_spec  # noqa: E402
from splinewire.cli import main  # noqa: E402
from splinewire.fixed_chain import RIM_MM, load_link, plaque_mesh, split_colours  # noqa: E402
from splinewire.targets import PLAQUE_POSES, plaque_pins  # noqa: E402


@pytest.fixture(scope="module")
def snap():
    return load_chain_spec(chain_presets()["snap"])


def test_the_link_mesh_is_the_link(snap):
    """data/snap-link.stl.gz: watertight, 13.05 mm tall, its rim on the
    origin and its sleeve one pitch along -x (snap_link_geometry --mesh)."""
    link = load_link()
    assert link.is_watertight
    assert link.bounds[0, 2] == pytest.approx(0.0, abs=1e-6) and link.extents[2] == pytest.approx(13.05, abs=0.01)
    top = link.vertices[link.vertices[:, 2] > link.bounds[1, 2] - 1e-6]
    assert np.linalg.norm(top[:, :2], axis=1).max() == pytest.approx(snap.fiducial_mm / 2, abs=0.01)
    assert np.linalg.norm(top[:, :2].mean(axis=0)) < 0.05
    assert link.bounds[0, 0] == pytest.approx(-snap.pitch_mm - snap.half_width_mm, abs=0.01)   # the sleeve's widest


@pytest.mark.parametrize("pose", list(PLAQUE_POSES))
def test_one_solid_with_its_rims_on_the_truth_pins(snap, pose):
    mesh = plaque_mesh(pose, snap)
    assert mesh.is_watertight and len(mesh.split()) == 1
    body, rims = split_colours(mesh)
    assert body.volume + rims.volume == pytest.approx(mesh.volume, rel=1e-6)
    assert rims.extents[2] == pytest.approx(RIM_MM, abs=1e-6)
    centres = np.array([p.centroid[:2] for p in rims.split(only_watertight=False)])
    pins = plaque_pins(pose, snap)
    assert len(centres) == len(pins)
    assert max(np.linalg.norm(centres - p, axis=1).min() for p in pins) < 0.005


def test_no_two_links_pass_through_each_other(snap):
    """In the tight pose (bends of 95-100 degrees), placed links touch only
    where a pin sits in its sleeve, 0.01 mm apart: no overlap at all."""
    pins = plaque_pins("plaque-tight", snap)
    ends = np.vstack([pins, pins[-1] + (pins[-1] - pins[-2])])
    link = load_link()
    parts = []
    for j in range(len(pins)):
        u = ends[j + 1] - ends[j]
        m = link.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(np.arctan2(u[1], u[0]) - np.pi, [0, 0, 1]))
        m.apply_translation([*ends[j], 0.0])
        parts.append(m)
    for i in range(len(parts) - 1):
        for k in range(i + 1, min(i + 4, len(parts))):
            both = trimesh.boolean.intersection([parts[i], parts[k]], engine="manifold")
            assert len(both.vertices) == 0 or abs(both.volume) < 1e-3, (i, k)


def test_fixed_chain_command(tmp_path, capsys):
    assert main(["fixed-chain", "--pose", "plaque-tight", "--links", "9", "--out", str(tmp_path)]) == 0
    truth = json.loads((tmp_path / "plaque-tight-9-truth.json").read_text())
    assert truth["name"] == "plaque-tight" and len(truth["pin_points"]) == 9
    for part in ("", "-body", "-rims"):
        assert (tmp_path / f"plaque-tight-9{part}.stl").stat().st_size > 0
    assert "black, then white" in capsys.readouterr().out
    assert main(["fixed-chain", "--links", "3", "--out", str(tmp_path)]) == 1
