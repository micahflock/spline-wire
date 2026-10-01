from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from splinewire.chain import ChainSpec, load_chain_spec

REPO = Path(__file__).parent.parent
RING = dict(fiducial="ring", fiducial_mm=5.0, ring_inner_mm=2.0)


@pytest.fixture
def dot_spec() -> ChainSpec:
    spec = load_chain_spec(REPO / "data" / "chain.yaml")
    assert spec.fiducial == "dot"
    return spec


@pytest.fixture
def ring_spec(dot_spec) -> ChainSpec:
    return replace(dot_spec, **RING)


@pytest.fixture(params=["dot", "ring"])
def spec(request, dot_spec) -> ChainSpec:
    """The chain as specified (dot fiducials), and the same chain with rings."""
    return dot_spec if request.param == "dot" else replace(dot_spec, **RING)


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through points: (center, radius)."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = np.sum(pts ** 2, axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(c + cx ** 2 + cy ** 2))
