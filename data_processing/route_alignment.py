"""Assign every GPS point to one arc of the reference route.

The .route file lists the arcs the vehicle drove, in order, but not which GPS
point belongs to which arc. This module finds that assignment:

    position[i] = index into the route list for GPS point i

subject to time order (position never decreases from one point to the next),
minimising the total point-to-arc distance. Solved exactly by dynamic
programming:

    D[i, j] = cost[i, j] + min_{k <= j} D[i-1, k]

Used for (a) per-point training labels (P1) and (b) Top-1 accuracy in the API's
server-side scoring (P4). Both must use this same function.

Limitations: the reference route itself may contain errors, and where two route
arcs are equally close (e.g. a U-turn on the same road) the choice between them
follows time order only.
"""
from __future__ import annotations

import numpy as np

_CHUNK_CELLS = 4_000_000  # points x arcs computed at once, bounds memory


def point_to_segment_distances(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment. Shapes (n, 2), (m, 2), (m, 2) -> (n, m), float32."""
    ab = seg_b - seg_a
    ab_len2 = np.einsum("ij,ij->i", ab, ab)
    ab_len2 = np.where(ab_len2 == 0, 1.0, ab_len2)  # zero-length arcs: distance to the endpoint
    out = np.empty((len(points), len(seg_a)), dtype=np.float32)
    rows = max(1, _CHUNK_CELLS // max(1, len(seg_a)))
    for start in range(0, len(points), rows):
        p = points[start:start + rows, None, :]                       # (r, 1, 2)
        t = np.einsum("rmk,mk->rm", p - seg_a[None], ab) / ab_len2    # (r, m)
        t = np.clip(t, 0.0, 1.0)
        closest = seg_a[None] + t[..., None] * ab[None]               # (r, m, 2)
        out[start:start + rows] = np.linalg.norm(p - closest, axis=2)
    return out


def align_points_to_route(gps_xy: np.ndarray, nodes_xy: np.ndarray, arcs: np.ndarray,
                          route: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (aligned_arc_id per GPS point, distance in metres to that arc)."""
    route = np.asarray(route, dtype=np.int64)
    seg_a = nodes_xy[arcs[route, 0]]
    seg_b = nodes_xy[arcs[route, 1]]
    cost = point_to_segment_distances(gps_xy, seg_a, seg_b)          # (n, m)

    n, m = cost.shape
    dp = np.empty_like(cost)
    dp[0] = cost[0]
    for i in range(1, n):
        dp[i] = cost[i] + np.minimum.accumulate(dp[i - 1])

    position = np.empty(n, dtype=np.int64)
    j = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        position[i] = j
        j = int(np.argmin(dp[i - 1, :j + 1]))
    position[0] = j

    return route[position], cost[np.arange(n), position]
