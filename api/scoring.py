"""Server-side scoring against the reference route.

This is the ONLY API module that reads a .route file. Reference arcs never
leave this module: only summary metrics are returned to the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.trip_io import load_arcs, trip_file


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


def score_trip(trip_dir: Path, prediction: dict) -> dict:
    reference = load_reference_route(trip_dir)
    route = prediction["predicted_arcs"]
    return {
        "top1_accuracy": None,  # needs per-point labels from P1's alignment
        **route_overlap(route, reference),
        "continuity": continuity(route, load_arcs(trip_dir)),
    }
