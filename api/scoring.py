"""Server-side scoring against the reference route.

This is the ONLY API module that reads a .route file. Reference arcs never
leave this module: only summary metrics are returned to the caller.

Per-point reference arcs come from P1's alignment (pathsnap_p1.alignment),
the same function that produced the training labels, so the API's Top-1
accuracy means exactly what the report's Top-1 accuracy means.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from pathsnap_p1.alignment import align_reference
from pathsnap_p1.data import Trip


def load_reference_route(trip_dir: Path, trip: Trip) -> np.ndarray:
    path = trip_dir / f"{trip_dir.name}.route"
    route = np.array([int(line) for line in path.read_text().split()], dtype=np.int64)
    valid = trip.arcs.valid.to_numpy()
    if len(route) == 0 or route.min() < 0 or route.max() >= len(valid) or not valid[route].all():
        raise ValueError(f"Reference route for trip {trip.trip_id} refers to missing or invalid arcs")
    return route


def route_overlap(predicted: list[int], reference: list[int]) -> dict[str, float]:
    """Precision / recall / F1 of the predicted arc set against the reference arc set."""
    pred, ref = set(predicted), set(reference)
    hits = len(pred & ref)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"route_precision": precision, "route_recall": recall, "route_f1": f1}


def continuity(route: list[int], trip: Trip) -> float | None:
    """Share of consecutive predicted arcs where one arc's to-node is the next arc's from-node."""
    if len(route) < 2:
        return None
    from_node = trip.arcs.from_node.to_numpy()
    to_node = trip.arcs.to_node.to_numpy()
    joined = sum(to_node[a] == from_node[b] for a, b in zip(route, route[1:]))
    return float(joined) / (len(route) - 1)


def top1_accuracy(predicted_arc_per_obs: list[int], reference_arc_per_obs: np.ndarray) -> float | None:
    """Share of labelled GPS points whose predicted arc is the aligned reference arc."""
    predicted = np.asarray(predicted_arc_per_obs, dtype=np.int64)
    labelled = reference_arc_per_obs >= 0
    if not labelled.any():
        return None
    return float(np.mean(predicted[labelled] == reference_arc_per_obs[labelled]))


def score_trip(trip_dir: Path, trip: Trip, prediction: dict, config: dict) -> dict:
    reference = load_reference_route(trip_dir, trip)
    # The trip was loaded for prediction without its reference. Attach the route to a
    # copy, here on the server only, so P1's alignment can label each point.
    labels = align_reference(dataclasses.replace(trip, route=reference), config)
    route = prediction["predicted_arcs"]
    return {
        "top1_accuracy": top1_accuracy(prediction["predicted_arc_per_obs"], labels.reference_arc_id.to_numpy()),
        **route_overlap(route, reference.tolist()),
        "continuity": continuity(route, trip),
    }
