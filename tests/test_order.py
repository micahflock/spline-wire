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


SHARP = 106.0   # the O-ring chain's stop: links 74° apart, pins either side 1.2 pitches apart


@pytest.mark.parametrize("turns_deg", [
    np.r_[np.zeros(5), SHARP, np.zeros(5)],                   # one sharp corner
    np.r_[np.zeros(3), SHARP, 0.0, 0.0, -SHARP, np.zeros(4)],  # sharp S
    np.r_[np.zeros(4), SHARP, 60.0, np.zeros(5)],             # hook
    np.r_[np.zeros(4), 90.0, 90.0, np.zeros(5)],              # U with legs one pitch apart
    SHARP * np.array([(-1) ** k for k in range(11)]),         # zigzag, every joint at its stop
])
@pytest.mark.parametrize("tilt_deg", [0.0, 25.0])
def test_orders_sharp_bends(turns_deg, tilt_deg):
    """With max_turn_deg raised for a chain that bends to 106°, foreshortened
    by a tilted camera: the walk neither skips the pin at a sharp bend nor
    starts mid-chain and doubles back."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        pins = pins_from_turns(40.0, np.radians(turns_deg), heading_rad=rng.uniform(0, 2 * np.pi))
        pins = pins * [1.0, np.cos(np.radians(tilt_deg))] + rng.normal(0, 0.3, pins.shape)
        pts, perm = _shuffled(pins, seed=int(rng.integers(1000)))
        order = order_chain(pts, max_turn_deg=125.0)
        _assert_chain_order(order, perm, list(range(len(pins))))
        assert order.gaps == []


@pytest.mark.parametrize("seed", [142, 291, 539, 545, 572])
def test_walks_over_the_same_pins_are_not_told_apart_by_rounding(seed):
    """A walk that doubles back over a sharp bend visits the same pins as the
    chain. The two tie on how much their detections stand out, and the walk
    with fewer steps off the pitch wins; summed in walk order, the contrasts
    differed in their last bit and that decided instead (8 zigzags in 1000)."""
    rng = np.random.default_rng(seed)
    turns = SHARP * np.array([(-1) ** k for k in range(11)])
    pins = pins_from_turns(40.0, np.radians(turns), heading_rad=rng.uniform(0, 2 * np.pi))
    pins = pins * [1.0, np.cos(np.radians(25.0))] + rng.normal(0, 0.3, pins.shape)
    contrast = rng.uniform(80, 180, len(pins))
    order = order_chain(pins, max_turn_deg=125.0, strength=contrast)
    assert order.indices in (list(range(13)), list(range(12, -1, -1)))


def test_a_longer_string_of_look_alikes_does_not_win_on_contrast_alone():
    """No walk counts as longer than the chain, and neither may its extra
    detections count towards how much it stands out: 14 printed letters in a
    row, each a little fainter than a dot, beat the chain's 13 dots when
    their contrasts were simply added up."""
    chain = pins_from_turns(40.0, np.radians(np.full(11, 10.0)))
    letters = pins_from_turns(40.0, np.zeros(12), start_mm=(0.0, 600.0))        # 14 in a row, well away
    pts = np.vstack([chain, letters])
    contrast = np.r_[np.full(len(chain), 100.0), np.full(len(letters), 95.0)]   # 1300 against 1330 in all
    order = order_chain(pts, n_pins=13, strength=contrast)
    assert sorted(order.indices) == list(range(13))
    fainter = np.r_[np.full(len(chain), 90.0), np.full(len(letters), 95.0)]     # pin for pin, the letters win
    assert min(order_chain(pts, n_pins=13, strength=fainter).indices) >= 13


def test_sharp_bends_need_the_raised_turn_limit():
    pins = pins_from_turns(40.0, np.radians(np.r_[np.zeros(5), SHARP, np.zeros(5)]))
    assert len(order_chain(pins).indices) < len(pins)                  # default 80°: stops at the corner


# ---------------------------------------------------------------------------
# Busy backgrounds: the neighbour search and the bounded walks

def _cluttered(seed: int):
    """A 13-pin chain with sharp and gentle bends, slightly foreshortened,
    among look-alikes: some the pins' size (stray dots), some smaller (a
    texture), a few a pitch or two past the ends, and a near-duplicate."""
    rng = np.random.default_rng(seed)
    turns = rng.choice([0.0, 15.0, -30.0, 60.0, SHARP, -SHARP], 11)
    pins = pins_from_turns(40.0, np.radians(turns), heading_rad=rng.uniform(0, 2 * np.pi))
    pins = pins * [1.0, np.cos(np.radians(rng.uniform(0, 30)))] + rng.normal(0, 0.3, pins.shape)
    n = int(rng.integers(20, 120))
    lo, hi = pins.min(axis=0) - 80, pins.max(axis=0) + 80
    clutter = rng.uniform(lo, hi, (n, 2))
    end_dir = pins[-1] - pins[-2]
    extra = [pins[-1] + end_dir * rng.uniform(0.9, 2.1), pins[0] + rng.normal(0, 0.2, 2)]
    pts = np.vstack([pins, clutter, extra])
    size = np.r_[rng.uniform(18, 22, len(pins)), rng.choice([8.0, 12.0, 20.0, 26.0], n),
                 rng.uniform(18, 22, 2)]
    axes = np.c_[size, size * rng.uniform(0.85, 1.0, len(size))]
    strength = rng.uniform(40, 160, len(pts))
    perm = rng.permutation(len(pts))
    return pts[perm], axes[perm], strength[perm]


@pytest.mark.parametrize("seed", range(16))
def test_same_order_as_the_all_pairs_version(seed):
    """The neighbour search gives the order the all-pairs tables gave, link
    for link and gap for gap, with every combination of what is known."""
    from tests.dense_order import order_chain as dense_order_chain
    pts, axes, strength = _cluttered(seed)
    for kw in (dict(max_turn_deg=80.0), dict(max_turn_deg=125.0, n_pins=13),
               dict(axes_px=axes, pitch_per_diameter=2.0, max_turn_deg=80.0, n_pins=13, strength=strength),
               dict(axes_px=axes, pitch_per_diameter=2.0, max_turn_deg=125.0, n_pins=13, strength=strength),
               dict(axes_px=axes, pitch_per_diameter=2.0, max_turn_deg=125.0)):
        new, old = order_chain(pts, **kw), dense_order_chain(pts, **kw)
        assert (new.indices, new.links, new.gaps, new.rejected) == (old.indices, old.links, old.gaps, old.rejected)


def _woven(n_spots: int):
    """A chain among n_spots fainter spots on a lattice (woven fabric seen
    close up), smaller than a pin and two of their diameters apart, so
    every walk over them is long."""
    rng = np.random.default_rng(7)
    pins = pins_from_turns(40.0, np.radians(np.full(11, 12.0)), start_mm=(600.0, 600.0))
    g = np.arange(0.0, 1700.0, 16.0)
    lattice = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2) + rng.normal(0, 0.5, (len(g) ** 2, 2))
    lattice = lattice[np.min(np.linalg.norm(lattice[:, None] - pins[None], axis=2), axis=1) > 14]
    lattice = lattice[np.argsort(np.linalg.norm(lattice - pins.mean(axis=0), axis=1))[:n_spots]]
    pts = np.vstack([pins, lattice])
    axes = np.vstack([_axes(len(pins)), np.full((len(lattice), 2), 8.0)])
    strength = np.r_[np.full(len(pins), 100.0), rng.uniform(30, 70, len(lattice))]
    return pts, dict(axes_px=axes, pitch_per_diameter=2.0, n_pins=13, max_turn_deg=125.0, strength=strength)


def test_chain_on_a_woven_texture_is_ordered_in_bounded_time():
    """10,000 spots: the all-pairs tables took 0.8 GB each and ordering ran
    for minutes; now about 2 s."""
    import time
    pts, kw = _woven(10000)
    t = time.perf_counter()
    order = order_chain(pts, **kw)
    seconds = time.perf_counter() - t
    assert sorted(order.indices) == list(range(13)) and order.gaps == []
    assert seconds < 15.0, f"{seconds:.1f} s"


def test_ordering_memory_grows_with_the_detections_not_their_square():
    """5,000 scattered specks smaller than a pin: the all-pairs tables
    took 200 MB each (the differences twice that); the neighbour search
    takes a few MB."""
    import tracemalloc
    rng = np.random.default_rng(8)
    pins = pins_from_turns(40.0, np.radians(np.full(11, 12.0)), start_mm=(1000.0, 1000.0))
    specks = rng.uniform(0, 2500, (5000, 2))
    specks = specks[np.min(np.linalg.norm(specks[:, None] - pins[None], axis=2), axis=1) > 30]
    pts = np.vstack([pins, specks])
    axes = np.vstack([_axes(len(pins)), np.repeat(rng.uniform(4, 13, (len(specks), 1)), 2, axis=1)])
    tracemalloc.start()
    try:
        order = order_chain(pts, axes_px=axes, pitch_per_diameter=2.0, n_pins=13, max_turn_deg=125.0,
                            strength=np.r_[np.full(len(pins), 100.0), rng.uniform(30, 70, len(specks))])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sorted(order.indices) == list(range(13))
    assert peak < 20 * 2 ** 20, f"{peak / 2 ** 20:.0f} MiB"


@pytest.mark.parametrize("seed", range(6))
def test_direction_is_the_earlier_end_on_a_near_tie(seed, monkeypatch):
    """Both directions along a chain tie on everything but total turning,
    which then differs only by rounding, and by platform through atan2. A
    difference under 1e-9 rad is no reason to prefer a walk: the one from
    the end detected first wins, so every platform agrees. Here later starts
    are made to turn 1e-12 rad less, as another platform's rounding might."""
    import splinewire.order as order_mod
    walk = order_mod._walk

    def rounded_otherwise(start, *args, **kwargs):
        path, turn = walk(start, *args, **kwargs)
        return path, turn - 1e-12 * start

    monkeypatch.setattr(order_mod, "_walk", rounded_otherwise)
    pins = pins_from_turns(10.0, np.radians(np.r_[np.full(5, 14.0), np.full(6, -14.0)])) * 9.0
    pts, perm = _shuffled(pins, seed)
    order = order_chain(pts)
    ends = sorted([int(np.flatnonzero(perm == 0)[0]), int(np.flatnonzero(perm == len(pins) - 1)[0])])
    assert order.indices[0] == ends[0]
