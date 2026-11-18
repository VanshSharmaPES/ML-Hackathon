"""End-to-end prediction for one trip.

    predictor = Predictor(load_artifacts(artifacts_dir))   # once, at startup
    result = predictor.predict(trip_dir)                   # per request

Two methods, chosen at startup:

* Full pipeline (when artifacts/ holds the exported model):
  candidates (P1, features.candidates.build_candidates)
  -> classifier probabilities (P2, artifacts/model.joblib)
  -> HMM/Viterbi decoding (P3, hmm.viterbi.decode)

* Nearest-arc baseline (until then): each GPS point is snapped to the closest
  road segment, then the reverse twin is kept if it points the way the vehicle
  is moving (most roads are stored as A->B and B->A with identical geometry).

The .route file is never read here. See api/scoring.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import shapely

from .artifacts import Artifacts
from .trip_io import Trip, load_trip

BASELINE_METHOD = "nearest-arc baseline"
NO_ARC = -1  # placeholder for a GPS point the decoder could not assign


# ---------- shared helpers ----------

def _collapse_repeats(arc_per_obs: list[int]) -> list[int]:
    """[5, 5, -1, 9, 9, 4] -> [5, 9, 4]: the route as an ordered arc sequence."""
    route: list[int] = []
    for arc in arc_per_obs:
        if arc != NO_ARC and (not route or route[-1] != arc):
            route.append(arc)
    return route


def _route_geojson(route: list[int], nodes: np.ndarray, arcs: np.ndarray) -> dict:
    features = []
    for order, arc_id in enumerate(route):
        a, b = arcs[arc_id]
        features.append({
            "type": "Feature",
            "properties": {"arc_id": int(arc_id), "order": order},
            "geometry": {"type": "LineString", "coordinates": [nodes[a].tolist(), nodes[b].tolist()]},
        })
    return {"type": "FeatureCollection", "features": features}


# ---------- nearest-arc baseline ----------

def _travel_direction(gps_xy: np.ndarray) -> np.ndarray:
    """Movement vector at each point: next point minus previous point."""
    nxt = np.vstack([gps_xy[1:], gps_xy[-1:]])
    prv = np.vstack([gps_xy[:1], gps_xy[:-1]])
    return nxt - prv


def _baseline_arcs(trip: Trip) -> np.ndarray:
    arcs, nodes_xy, gps_xy = trip.arcs, trip.nodes_xy, trip.gps_xy
    lines = shapely.linestrings(np.stack([nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]], axis=1))
    point_idx, arc_idx = shapely.STRtree(lines).query_nearest(shapely.points(gps_xy), all_matches=False)
    arc_per_obs = np.empty(len(gps_xy), dtype=np.int64)
    arc_per_obs[point_idx] = arc_idx

    # Swap to the reverse twin where the chosen arc points against the movement.
    reverse_of = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
    arc_vec = nodes_xy[arcs[arc_per_obs, 1]] - nodes_xy[arcs[arc_per_obs, 0]]
    backwards = np.einsum("ij,ij->i", arc_vec, _travel_direction(gps_xy)) < 0
    for k in np.flatnonzero(backwards):
        a, b = arcs[arc_per_obs[k]]
        twin = reverse_of.get((int(b), int(a)))
        if twin is not None:
            arc_per_obs[k] = twin
    return arc_per_obs


# ---------- predictor ----------

class Predictor:
    def __init__(self, artifacts: Artifacts):
        self.artifacts = artifacts
        if not artifacts.loaded:
            self.method = BASELINE_METHOD
            return
        # Imported only when a model exists, so the baseline runs before P1/P3 code lands.
        try:
            from features.candidates import build_candidates
            from hmm.viterbi import decode
        except ImportError as exc:
            raise RuntimeError(
                "Model artifacts are present but the pipeline code is missing: "
                f"{exc}. Expected features.candidates.build_candidates and hmm.viterbi.decode."
            ) from exc
        self._build_candidates = build_candidates
        self._decode = decode
        self._positive = list(artifacts.model.classes_).index(1)
        self.method = f"{artifacts.model_name} + HMM/Viterbi"

    @property
    def model_loaded(self) -> bool:
        return self.artifacts.loaded

    def _full_pipeline_arcs(self, trip: Trip, timings: dict[str, float]) -> np.ndarray:
        cfg = self.artifacts.hmm_config

        t = time.perf_counter()
        cands = self._build_candidates(trip, cfg)
        timings["candidates"] = (time.perf_counter() - t) * 1000

        missing = [f for f in self.artifacts.features if f not in cands.columns]
        if missing:
            raise RuntimeError(f"Candidate table is missing model features: {missing}")

        t = time.perf_counter()
        probs = self.artifacts.model.predict_proba(cands[self.artifacts.features])[:, self._positive]
        timings["model"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        decoded = self._decode(cands, probs, trip, cfg)  # {obs_idx: arc_id}
        timings["viterbi"] = (time.perf_counter() - t) * 1000

        return np.array([decoded.get(int(i), NO_ARC) for i in trip.track.obs_idx], dtype=np.int64)

    def predict(self, trip_dir: Path) -> dict:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        trip = load_trip(Path(trip_dir))
        timings["load"] = (time.perf_counter() - t) * 1000

        if self.model_loaded:
            arc_per_obs = self._full_pipeline_arcs(trip, timings)
        else:
            t = time.perf_counter()
            arc_per_obs = _baseline_arcs(trip)
            timings["matching"] = (time.perf_counter() - t) * 1000

        route = _collapse_repeats(arc_per_obs.tolist())
        track = trip.track
        return {
            "trip_id": trip.trip_id,
            "method": self.method,
            "obs": [
                {"obs_idx": int(i), "lon": float(lon), "lat": float(lat), "t": float(ts)}
                for i, lon, lat, ts in zip(track.obs_idx, track.lon, track.lat, track.t)
            ],
            "predicted_arc_per_obs": arc_per_obs.tolist(),
            "predicted_arcs": route,
            "geojson": _route_geojson(route, trip.nodes, trip.arcs),
            "timings": timings,
        }
