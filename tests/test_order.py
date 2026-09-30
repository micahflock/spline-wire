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


def test_missing_pin_becomes_a_gap():
    pins = pins_from_turns(40.0, np.radians(np.full(11, 10.0)))
    kept = [i for i in range(len(pins)) if i != 5]
    order = order_chain(pins[kept])
    got = [kept[i] for i in order.indices]
    assert got == kept or got == kept[::-1]
    assert len(order.gaps) == 1
    assert len(order.links) == len(kept) - 2
