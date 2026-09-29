"""Put unlabeled pin detections in chain order.

The fiducials carry no IDs, so order comes from geometry: walk from pin to
pin, stepping about one pitch each time and turning as little as possible.
A turn limit stops the walk from jumping across to a neighbouring part of
the chain (e.g. the other leg of a tight U-bend), and a step of about two
pitches is accepted as one missing detection. Every detection is tried as
the starting point; the walk that visits the most pins wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChainOrder:
    indices: list[int]              # detection indices in chain order
    links: list[tuple[int, int]]    # positions in `indices` exactly one pitch apart
    gaps: list[int]                 # k where indices[k] -> indices[k+1] skips one missing pin
    rejected: list[int]             # detections not on the chain


def order_chain(points_px: np.ndarray, max_turn_deg: float = 80.0) -> ChainOrder:
    pts = np.asarray(points_px, dtype=float)
    n = len(pts)
    if n < 3:
        raise ValueError(f"need at least 3 detections to find a chain, got {n}")
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn_pitch = float(np.median(dist.min(axis=1)))
    max_turn = np.radians(max_turn_deg)

    best_path, best_key = None, None
    for start in range(n):
        path, turn_total = _walk(start, pts, dist, nn_pitch, max_turn)
        key = (len(path), -turn_total)
        if best_key is None or key > best_key:
            best_path, best_key = path, key

    steps = [dist[a, b] for a, b in zip(best_path[:-1], best_path[1:])]
    links, gaps = [], []
    for k, s in enumerate(steps):
        local = np.median(steps[max(0, k - 3):k + 4])
        if s > 1.5 * min(local, nn_pitch * 1.3):
            gaps.append(k)
        else:
            links.append((k, k + 1))
    rejected = sorted(set(range(n)) - set(best_path))
    return ChainOrder(indices=best_path, links=links, gaps=gaps, rejected=rejected)


def _walk(start, pts, dist, nn_pitch, max_turn):
    path = [start]
    visited = {start}
    pitch = nn_pitch
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
            score = turn + 2.0 * abs(ratio - (2.0 if is_gap else 1.0)) + (1.0 if is_gap else 0.0)
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


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
