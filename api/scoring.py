"""Server-side scoring against the reference route.

This is the ONLY API module that reads a .route file. Reference arcs never
leave this module: only summary metrics are returned to the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data_processing.route_alignment import align_points_to_route
from inference.trip_io import load_trip, trip_file


def load_reference_route(trip_dir: Path) -> list[int]:
    with open(trip_file(trip_dir, "route")) as f:
        return [int(line) for line in f if line.strip()]


def route_overlap(predicted: list[int], reference: list[int]) -> dict[str, float]:
    """Precision / recall / F1 of the predicted arc set against the reference arc set."""
    pred, ref = set(predicted), set(reference)
    hits = len(pred & ref)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"route_precision": precision, "route_recall": recall, "route_f1": f1}


def continuity(route: list[int], arcs: np.ndarray) -> float | None:
    """Share of consecutive predicted arcs where one arc's to-node is the next arc's from-node."""
    if len(route) < 2:
        return None
    joined = sum(arcs[a, 1] == arcs[b, 0] for a, b in zip(route, route[1:]))
    return float(joined) / (len(route) - 1)


def top1_accuracy(predicted_arc_per_obs: list[int], aligned_arc_per_obs: np.ndarray) -> float:
    """Share of GPS points whose predicted arc is the reference arc aligned to that point."""
    predicted = np.asarray(predicted_arc_per_obs, dtype=np.int64)
    return float(np.mean(predicted == aligned_arc_per_obs))


def score_trip(trip_dir: Path, prediction: dict) -> dict:
    trip = load_trip(trip_dir)
    reference = load_reference_route(trip_dir)
    aligned, _ = align_points_to_route(trip.gps_xy, trip.nodes_xy, trip.arcs, reference)
    route = prediction["predicted_arcs"]
    return {
        "top1_accuracy": top1_accuracy(prediction["predicted_arc_per_obs"], aligned),
        **route_overlap(route, reference),
        "continuity": continuity(route, trip.arcs),
    }
