"""The ordering as it was before it used a neighbour search: every distance
from an all-pairs table, a walk started from every detection. Kept only to
check that splinewire.order gives the same order to the last bit wherever
this one runs (its tables need n x n memory, 2 GB per array at 11,000
detections). Not collected by pytest: the tests import it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn_pitch = float(np.median(dist.min(axis=1)))
    max_turn = np.radians(max_turn_deg)

    sizes = None
    if axes_px is not None and pitch_per_diameter:
        ax = np.asarray(axes_px, dtype=float).reshape(n, 2)
        sizes = _Sizes(major=ax[:, 0], minor=ax[:, 1], k=float(pitch_per_diameter))

    best_path, best_key = None, None
    for start in range(n):
        path, turn_total = _walk(start, pts, dist, nn_pitch, max_turn, sizes)
        length = min(len(path), n_pins) if n_pins else len(path)
        off, gaps = _off_pitch(path, dist)
        # How much its detections stand out, for as many as count: summed over
        # all of a walk longer than the chain, a string of 14 printed letters
        # beat the 13 dots on contrast alone. fsum: walks over the same
        # detections in another order must tie here exactly. A plain sum
        # differs in its last bit with the order, which then decided between
        # them instead of the terms after it.
        stands_out = math.fsum(strength[path]) * length / len(path) if strength is not None else 0.0
        key = (length, stands_out, -off, -gaps, -turn_total)
        if best_key is None or key > best_key:
            best_path, best_key = path, key

    steps = [dist[a, b] for a, b in zip(best_path[:-1], best_path[1:])]
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


@dataclass(frozen=True)
class _Sizes:
    major: np.ndarray
    minor: np.ndarray
    k: float

    def step_range(self, i: int, pitches: int) -> tuple[float, float]:
        """Plausible image length of a step of `pitches` pitches from ring i."""
        return (0.7 * pitches * self.k * self.minor[i] - 0.5 * self.minor[i],
                1.3 * pitches * self.k * self.major[i] + 0.5 * self.minor[i])

    def similar(self, i: int, j: int) -> bool:
        """Neighbouring pins look alike: about the same size, and about the
        same ellipse shape, since all lie on one plane seen from nearly the
        same direction (the counter of a printed "O" is a narrower oval)."""
        r = self.major[i] / self.major[j]
        aspect = abs(self.minor[i] / self.major[i] - self.minor[j] / self.major[j])
        return 1 / 1.35 < r < 1.35 and aspect < 0.15 + 2.0 / min(self.minor[i], self.minor[j])


def _walk(start, pts, dist, nn_pitch, max_turn, sizes: _Sizes | None):
    """Greedy walk from `start`. With sizes, every pin also stays within 1.6x
    of the walk's median size: perspective changes size along a real chain
    by ~1.3x at most, but a walk judging only neighbours could drift from
    pin to ever smaller specks."""
    def _fits(cur: int, j: int, path: list[int]) -> bool:
        """A step from cur to j passes the size and length tests."""
        if sizes is None:
            return True
        lo, hi = sizes.step_range(cur, 2 if dist[cur, j] / pitch > 1.5 else 1)
        if not (lo <= dist[cur, j] <= hi and sizes.similar(cur, j)):
            return False
        band = float(np.median(sizes.major[path]))
        return band / 1.6 < sizes.major[j] < band * 1.6

    path = [start]
    visited = {start}
    pitch = nn_pitch if sizes is None else sizes.k * 0.5 * (sizes.major[start] + sizes.minor[start])
    heading = None
    single_steps: list[float] = []
    turn_total = 0.0
    while True:
        cur = path[-1]
        best, best_score, best_turn = None, np.inf, 0.0
        for j in np.flatnonzero((dist[cur] > 0.5 * pitch) & (dist[cur] < 2.4 * pitch)):
            if j in visited:
                continue
            step = pts[j] - pts[cur]
            turn = 0.0 if heading is None else abs(_angle(heading, step))
            if turn > max_turn:
                continue
            ratio = dist[cur, j] / pitch
            is_gap = ratio > 1.5
            if not _fits(cur, j, path):
                continue
            if is_gap and _passes_over(cur, j, dist, pitch, visited, like=lambda m: _fits(cur, m, path)):
                continue                    # the pin it would step over is there, not missing
            score = turn + 2.0 * abs(ratio - (2.0 if is_gap else 1.0)) + (1.0 if is_gap else 0.0)
            if not is_gap and _passes_over(cur, j, dist, pitch, visited):
                score += 1.0
            if score < best_score:
                best, best_score, best_turn = j, score, turn
        if best is None:
            return path, turn_total
        step_len = dist[cur, best]
        if step_len <= 1.5 * pitch:
            single_steps.append(step_len)
            pitch = float(np.median(single_steps[-5:]))
        heading = pts[best] - pts[cur]
        turn_total += best_turn
        path.append(int(best))
        visited.add(int(best))


def _off_pitch(path: list[int], dist: np.ndarray) -> tuple[int, int]:
    """Steps of the walk more than 10% off a whole pitch, and steps of two
    pitches (over a missing pin). A walk that doubles back over a sharp bend
    takes a step of ~1.2 pitches where the chain itself takes two of one."""
    steps = np.array([dist[a, b] for a, b in zip(path[:-1], path[1:])])
    if len(steps) == 0:
        return 0, 0
    ratio = steps / np.median(steps)
    whole = np.clip(np.round(ratio), 1, 2)
    return int(np.sum(np.abs(ratio - whole) > 0.1)), int(np.sum(whole == 2))


def _passes_over(cur: int, j: int, dist: np.ndarray, pitch: float, visited: set, like=None) -> bool:
    """Some unvisited pin (for which like(m), if given) is about one pitch
    from both cur and j, and nearer to each than they are to each other."""
    d = dist[cur, j]
    near = ((np.abs(dist[cur] / pitch - 1.0) < 0.25) & (np.abs(dist[j] / pitch - 1.0) < 0.25)
            & (dist[cur] < d) & (dist[j] < d))
    near[[cur, j]] = False
    return any(m not in visited and (like is None or like(m)) for m in np.flatnonzero(near))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
