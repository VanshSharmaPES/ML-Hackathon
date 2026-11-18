"""End-to-end prediction for one trip: predict_trip(trip_dir) -> dict.

CURRENT METHOD: nearest-arc baseline. Each GPS point is snapped to the
geometrically closest road segment. Most roads are stored as two arcs with
identical geometry (A->B and B->A), so the baseline then keeps whichever of
the pair points the same way the vehicle is moving. It is a real computation
that lets the API and UI be built and tested now. It will be replaced by
candidates (P1) -> classifier (P2) -> HMM/Viterbi (P3).

The .route file is never read here. See api/scoring.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import shapely

from .geo import TripProjection
from .trip_io import load_trip

METHOD = "nearest-arc baseline"


def _arc_lines(nodes_xy: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    """One straight LineString per arc, from its from-node to its to-node."""
    coords = np.stack([nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]], axis=1)
    return shapely.linestrings(coords)


def _travel_direction(gps_xy: np.ndarray) -> np.ndarray:
    """Movement vector at each point: next point minus previous point."""
    nxt = np.vstack([gps_xy[1:], gps_xy[-1:]])
    prv = np.vstack([gps_xy[:1], gps_xy[:-1]])
    return nxt - prv


def _pick_direction(arc_per_obs: np.ndarray, arcs: np.ndarray, nodes_xy: np.ndarray,
                    gps_xy: np.ndarray) -> np.ndarray:
    """Swap an arc for its reverse twin when the twin points the way the vehicle moves."""
    reverse_of = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
    arc_vec = nodes_xy[arcs[arc_per_obs, 1]] - nodes_xy[arcs[arc_per_obs, 0]]
    moving_backwards = np.einsum("ij,ij->i", arc_vec, _travel_direction(gps_xy)) < 0
    out = arc_per_obs.copy()
    for k in np.flatnonzero(moving_backwards):
        a, b = arcs[arc_per_obs[k]]
        twin = reverse_of.get((int(b), int(a)))
        if twin is not None:
            out[k] = twin
    return out


def _collapse_repeats(arc_per_obs: list[int]) -> list[int]:
    """[5, 5, 5, 9, 9, 4] -> [5, 9, 4]: the route as an ordered arc sequence."""
    route: list[int] = []
    for arc in arc_per_obs:
        if not route or route[-1] != arc:
            route.append(arc)
    return route


def _route_geojson(route: list[int], nodes: np.ndarray, arcs: np.ndarray) -> dict:
    features = []
    for order, arc_id in enumerate(route):
        a, b = arcs[arc_id]
        features.append({
            "type": "Feature",
            "properties": {"arc_id": int(arc_id), "order": order},
            "geometry": {
                "type": "LineString",
                "coordinates": [nodes[a].tolist(), nodes[b].tolist()],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def predict_trip(trip_dir: Path) -> dict:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    trip = load_trip(Path(trip_dir))
    timings["load"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    gps_lonlat = trip.track[["lon", "lat"]].to_numpy()
    project = TripProjection(gps_lonlat)
    gps_xy = project(gps_lonlat)
    nodes_xy = project(trip.nodes)
    lines = _arc_lines(nodes_xy, trip.arcs)
    point_idx, arc_idx = shapely.STRtree(lines).query_nearest(shapely.points(gps_xy), all_matches=False)
    arc_per_obs = np.empty(len(gps_xy), dtype=np.int64)
    arc_per_obs[point_idx] = arc_idx
    arc_per_obs = _pick_direction(arc_per_obs, trip.arcs, nodes_xy, gps_xy)
    timings["matching"] = (time.perf_counter() - t) * 1000

    route = _collapse_repeats(arc_per_obs.tolist())
    track = trip.track
    return {
        "trip_id": trip.trip_id,
        "method": METHOD,
        "obs": [
            {"obs_idx": int(i), "lon": float(lon), "lat": float(lat), "t": float(ts)}
            for i, lon, lat, ts in zip(track.obs_idx, track.lon, track.lat, track.t)
        ],
        "predicted_arc_per_obs": arc_per_obs.tolist(),
        "predicted_arcs": route,
        "geojson": _route_geojson(route, trip.nodes, trip.arcs),
        "timings": timings,
    }
