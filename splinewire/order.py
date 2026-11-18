"""Put unlabeled pin detections in chain order.

The fiducials carry no IDs, so order comes from geometry: walk from pin to
pin, stepping about one pitch each time and turning as little as possible.
A turn limit stops the walk from jumping across to a neighbouring part of
the chain (e.g. the other leg of a tight U-bend), and a step of about two
pitches is accepted as one missing detection. A step never passes over an
unvisited pin that lies about one pitch from both its ends: at a sharp
bend the pin after next can be nearer than the next pin's turn suggests
(1.2 pitches at the O-ring chain's stop), and skipping it would lose it.
Nor is a step over a missing pin taken where that pin is there, looking
like the others: past the turn limit (a bend sharper than the chain
allows) the walk would otherwise step round the bend's pin as if it were
undetected, and the chain read with a pin in the wrong place instead of
as cut at the bend. (A pin that fails the size tests, say half washed out
by glare, is still bridged as missing.)
Every detection is tried as the starting point (on a busy background,
only the most end-like: see _starts); the walk that visits the most pins
wins.

Distances are only ever needed between detections within a step of each
other, so they come from a k-d tree, not an all-pairs table: woven fabric
seen close up gives over 10,000 look-alikes, and the table for those took
2 GB per array (13 GB at the peak, more than a browser tab can have).

When ring sizes are given, each ring also predicts how long a step from it
should be (the pitch is a known multiple of the ring diameter, foreshortened
by at most the ring's own minor/major ratio), and neighbours must be about
the same size. That keeps look-alikes out of the chain: washers, printed
letters, a speckled table.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ChainOrder:
    indices: list[int]              # detection indices in chain order
    links: list[tuple[int, int]]    # positions in `indices` exactly one pitch apart
    gaps: list[int]                 # k where indices[k] -> indices[k+1] skips one missing pin
    rejected: list[int]             # detections not on the chain


def order_chain(
    points_px: np.ndarray,
    max_turn_deg: float = 80.0,
    axes_px: np.ndarray | None = None,
    pitch_per_diameter: float | None = None,
    n_pins: int | None = None,
    strength: np.ndarray | None = None,
) -> ChainOrder:
    """Order detections along the chain.

    axes_px: optional (N, 2) outer ellipse (major, minor) diameter of each
    fiducial; pitch_per_diameter: pitch / fiducial outer diameter from the
    spec. n_pins: how many pins the chain has; strength: optional (N,) how
    clearly each detection stands out (its contrast).

    The walk that visits the most pins wins, but with n_pins given no walk
    counts as longer than the chain (a regular texture can string together
    more look-alikes than the chain has pins); among those, the one whose
    detections stand out most, pin for pin; among walks over the same detections, the one
    with the fewest steps off a whole pitch, then gaps, then the least turning.
    """
    pts = np.asarray(points_px, dtype=float)
    n = len(pts)
    if n < 3:
        raise ValueError(f"need at least 3 detections to find a chain, got {n}")
    near = _Near(pts)
    nn_pitch = float(np.median(near.nearest()))
    max_turn = np.radians(max_turn_deg)

    sizes = None
    if axes_px is not None and pitch_per_diameter:
        ax = np.asarray(axes_px, dtype=float).reshape(n, 2)
        sizes = _Sizes(major=ax[:, 0], minor=ax[:, 1], k=float(pitch_per_diameter))

    starts = _starts(pts, near, nn_pitch, sizes, None if strength is None else np.asarray(strength, dtype=float))
    # On a busy background walks also stop at a few times the chain's length:
    # one that long is following a texture, and on fabric most walks do.
    max_len = _BUSY_WALK * n_pins if n_pins and len(starts) < n else None
    best_path, best_key, best_turn = None, None, 0.0
    for start in starts:
        path, turn_total = _walk(start, pts, near, nn_pitch, max_turn, sizes, max_len)
        length = min(len(path), n_pins) if n_pins else len(path)
        off, gaps = _off_pitch(path, pts)
        # How much its detections stand out, for as many as count: summed over
        # all of a walk longer than the chain, a string of 14 printed letters
        # beat the 13 dots on contrast alone. fsum: walks over the same
        # detections in another order must tie here exactly. A plain sum
        # differs in its last bit with the order, which then decided between
        # them instead of the terms after it.
        stands_out = math.fsum(strength[path]) * length / len(path) if strength is not None else 0.0
        # Least turning last, and only by more than rounding: the two
        # directions along one chain turn alike, and which of them adds up a
        # hair less depends on the platform's atan2 in its last bit (glibc,
        # musl under Pyodide, MSVC and V8 differ there). On a near tie the
        # earlier start keeps it, so every platform picks the same direction.
        key = (length, stands_out, -off, -gaps)
        if (best_key is None or key > best_key
                or (key == best_key and turn_total < best_turn - _TURN_TIE)):
            best_path, best_key, best_turn = path, key, turn_total

    steps = list(_steps(best_path, pts))
    links, gaps = [], []
    for k, s in enumerate(steps):
        if sizes is not None:
            a, b = best_path[k], best_path[k + 1]
            expected = sizes.k * (sizes.major[a] + sizes.major[b]) / 2
            local = min(np.median(steps[max(0, k - 3):k + 4]), expected)
        else:
            local = min(np.median(steps[max(0, k - 3):k + 4]), nn_pitch * 1.3)
        if s > 1.5 * local:
            gaps.append(k)
        else:
            links.append((k, k + 1))
    rejected = sorted(set(range(n)) - set(best_path))
    return ChainOrder(indices=best_path, links=links, gaps=gaps, rejected=rejected)


# Walks are started from at most this many detections: up to it from every
# one, so the order is the same as when every detection started a walk.
# More detections than this come only from busy backgrounds (woven fabric,
# terrazzo, print seen close up: none of the 167 benchmark photos has more
# than 510). With as few as 300 starts the chain was still found wherever it
# was found before, in the benchmark and the video frame study's sharp
# frames and stills (815 photos); with 150 it was lost among printed text,
# whose letters' counters look like chain ends.
_MAX_STARTS = 600
_TURN_TIE = 1e-9         # radians of total turning that count as equal (see order_chain)
_BUSY_WALK = 2          # walks then stop at this many times the chain's pins


def _starts(pts: np.ndarray, near: _Near, nn_pitch: float, sizes: _Sizes | None,
            strength: np.ndarray | None, limit: int | None = None) -> range | np.ndarray:
    """The detections to start walks from, in increasing order.

    A walk is greedy and only goes forward, so the one that wins starts at
    an end of the chain (or a look-alike just past it). An end has one
    neighbour about a pitch away that looks like it; a pin inside the
    chain has two, and a spot on a woven texture four or more. Every walk
    costs up to its length in steps, and on a texture almost every walk is
    long: ordering the 1,336 spots a 12 MP photo of fabric leaves after
    detection took 5.6 s at an 80 degree turn limit and 23 s at 125. So
    beyond the limit walks start from the detections with the fewest such
    neighbours, and among as many, those that stand out most (as the walks
    themselves are judged).
    """
    limit = _MAX_STARTS if limit is None else limit
    n = len(pts)
    if n <= limit:
        return range(n)
    pitch = (np.full(n, nn_pitch) if sizes is None
             else sizes.k * 0.5 * (sizes.major + sizes.minor))
    count = np.zeros(n, dtype=int)
    for i, js in enumerate(near.tree.query_ball_point(pts, 1.5 * pitch)):
        js = np.asarray(js, dtype=np.intp)
        d = np.linalg.norm(pts[js] - pts[i], axis=1)
        ok = (d > 0.5 * pitch[i]) & (d < 1.5 * pitch[i])
        if sizes is not None:
            r = sizes.major[i] / sizes.major[js]
            ok &= (1 / 1.35 < r) & (r < 1.35)
        count[i] = int(ok.sum())
    # Detections with no neighbour come last: their walks can only skip
    # over a missing pin, as from an end whose neighbour went undetected.
    rank = np.where(count == 0, n, count)
    stands_out = -strength if strength is not None else np.zeros(n)
    return np.sort(np.lexsort((np.arange(n), stands_out, rank))[:limit])


class _Near:
    """Distances from a detection to those around it, from a k-d tree.

    Every distance is computed as the all-pairs table computed it (the norm
    of the difference, reduced along its last axis), so comparisons and
    sums over them come out the same to the last bit, and with them the
    choice of walk. A detection's distance to itself is infinite, as the
    table's diagonal was.
    """

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.tree = cKDTree(pts)
        self._balls: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}

    def nearest(self) -> np.ndarray:
        """Each detection's distance to its nearest other one."""
        k = min(len(self.pts), 4)       # a few, so a duplicate point cannot hide the nearest
        _, nb = self.tree.query(self.pts, k=k)
        d = np.linalg.norm(self.pts[nb] - self.pts[:, None], axis=2)
        d[nb == np.arange(len(self.pts))[:, None]] = np.inf
        return d.min(axis=1)

    def around(self, i: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices, in increasing order, and distances of at least every
        detection nearer to i than radius (and maybe some farther)."""
        ball = self._balls.get(i)
        if ball is None or ball[0] < radius:
            # A hair wider than asked: the tree's own distance may differ
            # from ours in the last bit.
            idx = np.asarray(self.tree.query_ball_point(self.pts[i], radius * (1 + 1e-9) + 1e-9,
                                                        return_sorted=True), dtype=np.intp)
            d = np.linalg.norm(self.pts[idx] - self.pts[i], axis=1)
            d[idx == i] = np.inf
            ball = (radius, idx, d)
            self._balls[i] = ball
        return ball[1], ball[2]


def _steps(path: list[int], pts: np.ndarray) -> np.ndarray:
    p = np.asarray(path, dtype=np.intp)
    return np.linalg.norm(pts[p[:-1]] - pts[p[1:]], axis=1)


@dataclass(frozen=True)
class _Sizes:
    major: np.ndarray
    minor: np.ndarray
    k: float

    def step_range(self, i: int, pitches: int) -> tuple[float, float]:
        """Plausible image length of a step of `pitches` pitches from ring i."""
        return (0.7 * pitches * self.k * self.minor[i] - 0.5 * self.minor[i],
                1.3 * pitches * self.k * self.major[i] + 0.5 * self.minor[i])

    def fits(self, i: int, js: np.ndarray, d: np.ndarray, gap: np.ndarray, band: float) -> np.ndarray:
        """Which steps from ring i to rings js, of image lengths d (gap: over
        a missing pin), have a plausible length, join rings that look alike,
        and stay within 1.6x of the walk's median size `band`.

        Neighbouring pins look alike: about the same size, and about the
        same ellipse shape, since all lie on one plane seen from nearly the
        same direction (the counter of a printed "O" is a narrower oval)."""
        (lo1, hi1), (lo2, hi2) = self.step_range(i, 1), self.step_range(i, 2)
        lo, hi = np.where(gap, lo2, lo1), np.where(gap, hi2, hi1)
        major, minor = self.major[js], self.minor[js]
        r = self.major[i] / major
        aspect = np.abs(self.minor[i] / self.major[i] - minor / major)
        return ((lo <= d) & (d <= hi) & (1 / 1.35 < r) & (r < 1.35)
                & (aspect < 0.15 + 2.0 / np.minimum(self.minor[i], minor))
                & (band / 1.6 < major) & (major < band * 1.6))


def _walk(start, pts, near: _Near, nn_pitch, max_turn, sizes: _Sizes | None, max_len: int | None = None):
    """Greedy walk from `start`, of at most max_len detections. With sizes,
    every pin also stays within 1.6x of the walk's median size: perspective
    changes size along a real chain by ~1.3x at most, but a walk judging
    only neighbours could drift from pin to ever smaller specks."""
    path = [start]
    visited = {start}
    pitch = nn_pitch if sizes is None else sizes.k * 0.5 * (sizes.major[start] + sizes.minor[start])
    heading = None
    single_steps: list[float] = []
    path_sizes = [] if sizes is None else [sizes.major[start]]      # sorted, for the median
    turn_total = 0.0
    cos_max = math.cos(max_turn) if max_turn < math.pi else -2.0
    while max_len is None or len(path) < max_len:
        cur = path[-1]
        best, best_score, best_turn, best_len = None, np.inf, 0.0, 0.0
        idx, dist = near.around(cur, 2.4 * pitch)
        ratio = dist / pitch
        gap = ratio > 1.5
        ok = (dist > 0.5 * pitch) & (dist < 2.4 * pitch)
        if sizes is not None:
            # Size and length tests for all neighbours at once; only the
            # turn, which needs an arctangent, is left to the loop.
            ok &= sizes.fits(cur, idx, dist, gap, float(_median(path_sizes)))
        sel = np.flatnonzero(ok)
        if heading is not None and len(sel):
            # Leave out steps turning clearly more than max_turn, by their
            # cosine; the loop decides those near the limit by the angle.
            cos = ((pts[idx[sel]] - pts[cur]) @ heading) / (dist[sel] * math.hypot(*heading))
            sel = sel[cos >= cos_max - 1e-9]
        over = _passes_over(idx, dist, sel[~gap[sel]], pts, pitch, visited)
        # A step over a missing pin is not taken where a detection that would
        # pass for the next pin (all but the turn) lies where that pin would be.
        skips = _passes_over(idx, dist, sel[gap[sel]], pts, pitch, visited, like=ok)
        for k in sel:
            j, d, is_gap = idx[k], dist[k], gap[k]
            if j in visited:
                continue
            if is_gap and skips[k]:
                continue
            step = pts[j] - pts[cur]
            turn = 0.0 if heading is None else abs(_angle(heading, step))
            if turn > max_turn:
                continue
            score = turn + 2.0 * abs(ratio[k] - (2.0 if is_gap else 1.0)) + (1.0 if is_gap else 0.0)
            if not is_gap and over[k]:
                score += 1.0
            if score < best_score:
                best, best_score, best_turn, best_len = j, score, turn, d
        if best is None:
            return path, turn_total
        step_len = best_len
        if step_len <= 1.5 * pitch:
            single_steps.append(step_len)
            pitch = float(_median(sorted(single_steps[-5:])))
        heading = pts[best] - pts[cur]
        turn_total += best_turn
        path.append(int(best))
        visited.add(int(best))
        if sizes is not None:
            bisect.insort(path_sizes, sizes.major[best])
    return path, turn_total


def _median(v: list[float]) -> float:
    """The median of sorted values, as np.median gives it to the last bit."""
    h = len(v) // 2
    return v[h] if len(v) % 2 else (v[h - 1] + v[h]) / 2


def _off_pitch(path: list[int], pts: np.ndarray) -> tuple[int, int]:
    """Steps of the walk more than 10% off a whole pitch, and steps of two
    pitches (over a missing pin). A walk that doubles back over a sharp bend
    takes a step of ~1.2 pitches where the chain itself takes two of one."""
    steps = _steps(path, pts)
    if len(steps) == 0:
        return 0, 0
    ratio = steps / np.median(steps)
    whole = np.clip(np.round(ratio), 1, 2)
    return int(np.sum(np.abs(ratio - whole) > 0.1)), int(np.sum(whole == 2))


def _passes_over(idx: np.ndarray, dist: np.ndarray, steps: np.ndarray, pts: np.ndarray,
                 pitch: float, visited: set, like: np.ndarray | None = None) -> dict:
    """For each of the steps from cur to idx[k] (k in steps; idx and dist are
    what lies around cur, out to more than 1.25 pitches): whether some
    unvisited pin is about one pitch from both ends, and nearer to each
    than they are to each other. For a step of two pitches (over a missing
    pin) that is the pin it would skip. like: only detections idx[m] with
    like[m] count (those that pass for a step from cur)."""
    out = dict.fromkeys(steps.tolist(), False)
    mid = np.flatnonzero((np.abs(dist / pitch - 1.0) < 0.25) & (True if like is None else like))
    mid = mid[[idx[m] not in visited for m in mid]] if len(mid) else mid
    if len(mid) == 0 or len(steps) == 0:
        return out
    js, d = idx[steps], dist[steps]
    to_j = np.linalg.norm(pts[idx[mid]][None] - pts[js][:, None], axis=2)
    near = ((dist[mid][None] < d[:, None]) & (idx[mid][None] != js[:, None])
            & (np.abs(to_j / pitch - 1.0) < 0.25) & (to_j < d[:, None]))
    return dict(zip(steps.tolist(), near.any(axis=1).tolist()))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
