"""End-to-end prediction for one trip.

    predictor = Predictor(load_artifacts(artifacts_dir), pipeline_config)   # once, at startup
    prediction, trip = predictor.predict(trip_dir)                          # per request

Trips are loaded with P1's loader (pathsnap_p1.data.load_trip) with
with_reference=False, so the .route file is never opened here.

Three methods, chosen at startup from what is in artifacts/:

* Classifier + HMM (model.joblib + hmm_config.json):
  candidates + features (P1, pathsnap_p1.features.generate_features, using the
  configuration stored in the model bundle, i.e. exactly the training settings)
  -> classifier probabilities (P2, artifacts/model.joblib)
  -> HMM/Viterbi decoding (P3, hmm.viterbi.decode)

* Classifier only (model.joblib alone): same candidates and probabilities, then
  the highest-scoring candidate per GPS point (ties: lower distance rank, then
  lower arc ID), exactly as observation-level Top-1 is computed in training.

* Nearest-arc baseline (no model): each GPS point is snapped to the closest
  valid road segment, then the reverse twin is kept if it points the way the
  vehicle is moving (most roads are stored as A->B and B->A with identical geometry).

Ground truth is only read by api/scoring.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import shapely

from pathsnap_p1.data import Trip, load_trip
from pathsnap_p1.features import generate_features

from .artifacts import Artifacts

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


def _route_geojson(route: list[int], trip: Trip) -> dict:
    lonlat = trip.nodes[["longitude", "latitude"]].to_numpy()
    ends = trip.arcs[["from_node", "to_node"]].to_numpy()
    features = []
    for order, arc_id in enumerate(route):
        a, b = ends[arc_id]
        features.append({
            "type": "Feature",
            "properties": {"arc_id": int(arc_id), "order": order},
            "geometry": {"type": "LineString", "coordinates": [lonlat[a].tolist(), lonlat[b].tolist()]},
        })
    return {"type": "FeatureCollection", "features": features}


# ---------- nearest-arc baseline ----------

def _travel_direction(gps_xy: np.ndarray) -> np.ndarray:
    """Movement vector at each point: next point minus previous point."""
    nxt = np.vstack([gps_xy[1:], gps_xy[-1:]])
    prv = np.vstack([gps_xy[:1], gps_xy[:-1]])
    return nxt - prv


def _baseline_arcs(trip: Trip) -> np.ndarray:
    arcs = trip.arcs.loc[trip.arcs.valid]
    arc_ids = arcs.arc_id.to_numpy()
    a, b = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    nodes_xy = trip.nodes[["x_m", "y_m"]].to_numpy()
    gps_xy = trip.observations[["x_m", "y_m"]].to_numpy()

    lines = shapely.linestrings(np.stack([nodes_xy[a], nodes_xy[b]], axis=1))
    point_idx, line_idx = shapely.STRtree(lines).query_nearest(shapely.points(gps_xy), all_matches=False)
    choice = np.empty(len(gps_xy), dtype=np.int64)  # index into the valid-arc arrays
    choice[point_idx] = line_idx

    # Swap to the reverse twin where the chosen arc points against the movement.
    reverse_of = {(int(f), int(t)): k for k, (f, t) in enumerate(zip(a, b))}
    arc_vec = nodes_xy[b[choice]] - nodes_xy[a[choice]]
    backwards = np.einsum("ij,ij->i", arc_vec, _travel_direction(gps_xy)) < 0
    for k in np.flatnonzero(backwards):
        twin = reverse_of.get((int(b[choice[k]]), int(a[choice[k]])))
        if twin is not None:
            choice[k] = twin
    return arc_ids[choice]


# ---------- classifier only ----------

def _classifier_only(candidates, probs, trip: Trip, cfg) -> dict[int, int]:
    """Highest-scoring candidate per GPS point. Ties: lower distance rank, then lower arc ID
    (the same order pathsnap_p1.models.evaluate_predictions uses for Top-1)."""
    scored = candidates[["observation_id", "candidate_arc_id", "distance_rank"]].assign(score=probs)
    scored = scored.sort_values(["observation_id", "score", "distance_rank", "candidate_arc_id"],
                                ascending=[True, False, True, True], kind="stable")
    best = scored.drop_duplicates("observation_id")
    return dict(zip(best.observation_id.astype(int), best.candidate_arc_id.astype(int)))


# ---------- predictor ----------

class Predictor:
    def __init__(self, artifacts: Artifacts, pipeline_config: dict):
        self.artifacts = artifacts
        if not artifacts.loaded:
            self.config = pipeline_config
            self.method = BASELINE_METHOD
            return
        # Serve with the exact settings the model was trained with.
        self.config = artifacts.feature_config
        self._positive = list(artifacts.pipeline.classes_).index(1)
        if not artifacts.uses_hmm:
            self._decode = _classifier_only
            self.method = f"{artifacts.model_name}, classifier only"
            return
        try:
            from hmm.viterbi import decode
        except ImportError as exc:
            raise RuntimeError(
                f"hmm_config.json is present but the decoder is missing: {exc}. Expected hmm.viterbi.decode."
            ) from exc
        self._decode = decode
        self.method = f"{artifacts.model_name} + HMM/Viterbi"

    @property
    def model_loaded(self) -> bool:
        return self.artifacts.loaded

    def _full_pipeline_arcs(self, trip: Trip, timings: dict[str, float]) -> np.ndarray:
        t = time.perf_counter()
        cands, _, _, _ = generate_features(trip, self.config)
        timings["candidates"] = (time.perf_counter() - t) * 1000

        features = self.artifacts.features
        missing = [f for f in features if f not in cands.columns]
        if missing:
            raise RuntimeError(f"Candidate table is missing model features: {missing}")

        t = time.perf_counter()
        probs = self.artifacts.pipeline.predict_proba(cands[features])[:, self._positive]
        timings["model"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        decoded = self._decode(cands, probs, trip, self.artifacts.hmm_config)  # {observation_id: arc_id}
        timings["viterbi" if self.artifacts.uses_hmm else "select"] = (time.perf_counter() - t) * 1000

        return np.array([decoded.get(int(o), NO_ARC) for o in trip.observations.observation_id], dtype=np.int64)

    def predict(self, trip_dir: Path) -> tuple[dict, Trip]:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        trip = load_trip(Path(trip_dir), "inference", self.config, with_reference=False)
        timings["load"] = (time.perf_counter() - t) * 1000

        if self.model_loaded:
            arc_per_obs = self._full_pipeline_arcs(trip, timings)
        else:
            t = time.perf_counter()
            arc_per_obs = _baseline_arcs(trip)
            timings["matching"] = (time.perf_counter() - t) * 1000

        route = _collapse_repeats(arc_per_obs.tolist())
        obs = trip.observations
        prediction = {
            "trip_id": trip.trip_id,
            "method": self.method,
            "obs": [
                {"obs_idx": int(i), "lon": float(lon), "lat": float(lat), "t": float(ts)}
                for i, lon, lat, ts in zip(obs.observation_id, obs.longitude, obs.latitude, obs.timestamp)
            ],
            "predicted_arc_per_obs": arc_per_obs.tolist(),
            "predicted_arcs": route,
            "geojson": _route_geojson(route, trip),
            "timings": timings,
        }
        return prediction, trip
