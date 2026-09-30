from pathlib import Path

import numpy as np
import pytest

from splinewire.chain import ChainSpec, load_chain_spec

REPO = Path(__file__).parent.parent


@pytest.fixture
def spec() -> ChainSpec:
    return load_chain_spec(REPO / "data" / "chain.yaml")


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through points: (center, radius)."""
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = np.sum(pts ** 2, axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(c + cx ** 2 + cy ** 2))
