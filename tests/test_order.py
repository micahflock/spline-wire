import numpy as np
import pytest

from splinewire.chain import pins_from_turns
from splinewire.order import order_chain


def _shuffled(pins, seed=0):
    perm = np.random.default_rng(seed).permutation(len(pins))
    return pins[perm], perm


def _assert_chain_order(order, perm, expected):
    got = list(perm[order.indices])
    assert got == expected or got == expected[::-1]


@pytest.mark.parametrize("turns_deg", [
    np.r_[np.full(5, 14.0), np.full(6, -14.0)],   # S-curve
    np.full(11, 23.0),                             # wrapped around a pipe
    np.r_[np.zeros(4), np.full(3, 60.0), np.zeros(4)],   # U-bend
])
def test_orders_shuffled_detections(turns_deg):
    pins = pins_from_turns(40.0, np.radians(turns_deg))   # 40 px pitch
    pts, perm = _shuffled(pins)
    order = order_chain(pts)
    _assert_chain_order(order, perm, list(range(len(pins))))
    assert order.gaps == [] and order.rejected == []
    assert order.links == [(k, k + 1) for k in range(len(pins) - 1)]


def test_stray_detection_is_rejected():
    pins = pins_from_turns(40.0, np.radians(np.full(11, 10.0)))
    pts = np.vstack([pins, [[pins[:, 0].mean(), pins[:, 1].max() + 300.0]]])
    order = order_chain(pts)
    assert order.rejected == [len(pins)]
    assert len(order.indices) == len(pins)


def _axes(n, d=20.0):
    return np.tile([d, d], (n, 1))                        # 20 px rings, pitch 40 px = 2 diameters


def test_ring_sizes_keep_a_bigger_look_alike_off_the_chain_end():
    """A washer one pitch past the end, in line with the chain: by position
    alone it extends the chain; its size gives it away."""
    pins = pins_from_turns(40.0, np.radians(np.full(11, 10.0)))
    end_dir = (pins[-1] - pins[-2]) / np.linalg.norm(pins[-1] - pins[-2])
    pts = np.vstack([pins, pins[-1] + 40.0 * end_dir])
    assert len(order_chain(pts).indices) == len(pins) + 1          # fooled without sizes
    axes = np.vstack([_axes(len(pins)), [[30.0, 30.0]]])            # 1.5x the ring size
    order = order_chain(pts, axes_px=axes, pitch_per_diameter=2.0)
    assert len(order.indices) == len(pins) and order.rejected == [len(pins)]


def test_ring_sizes_set_the_pitch_among_dense_clutter():
    """Many small look-alikes (printed letters) make the median neighbour
    distance useless as a pitch estimate; each ring's size still predicts it."""
    pins = pins_from_turns(40.0, np.radians(np.full(11, 12.0)))
    rng = np.random.default_rng(1)
    letters = rng.uniform(pins.min(axis=0) - 100, pins.max(axis=0) + 100, (80, 2))
    letters = letters[np.min(np.linalg.norm(letters[:, None] - pins[None], axis=2), axis=1) > 25]
    pts = np.vstack([pins, letters])
    axes = np.vstack([_axes(len(pins)), rng.uniform(6.0, 12.0, (len(letters), 2))])
    order = order_chain(pts, axes_px=axes, pitch_per_diameter=2.0)
    assert sorted(order.indices) == list(range(len(pins)))


def test_missing_pin_becomes_a_gap():
    pins = pins_from_turns(40.0, np.radians(np.full(11, 10.0)))
    kept = [i for i in range(len(pins)) if i != 5]
    order = order_chain(pins[kept])
    got = [kept[i] for i in order.indices]
    assert got == kept or got == kept[::-1]
    assert len(order.gaps) == 1
    assert len(order.links) == len(kept) - 2
