"""Read a trip folder's GPS track and road network, already projected to metres.

This module never reads the .route file. Ground truth is read only by
api/scoring.py, on the server, after the prediction has been made.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geo import TripProjection


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame   # columns: obs_idx, lon, lat, t
    nodes: np.ndarray     # shape (n_nodes, 2): lon, lat. Row index = node_id
    arcs: np.ndarray      # shape (n_arcs, 2): from_node, to_node. Row index = arc_id
    epsg: int             # UTM zone used for this trip
    gps_xy: np.ndarray    # shape (n_points, 2): metres, same row order as track
    nodes_xy: np.ndarray  # shape (n_nodes, 2): metres


def trip_file(trip_dir: Path, ext: str) -> Path:
    """Path of <trip_dir>/<trip_id>.<ext>."""
    return trip_dir / f"{trip_dir.name}.{ext}"


def _read_table(path: Path, names: list[str], dtype) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_track(trip_dir: Path) -> pd.DataFrame:
    """GPS points in file order. obs_idx is the row number in the .track file."""
    df = _read_table(trip_file(trip_dir, "track"), ["lon", "lat", "t"], float)
    df.insert(0, "obs_idx", np.arange(len(df)))
    return df.dropna(subset=["lon", "lat", "t"]).reset_index(drop=True)


def load_nodes(trip_dir: Path) -> np.ndarray:
    return _read_table(trip_file(trip_dir, "nodes"), ["lon", "lat"], float).to_numpy()


def load_arcs(trip_dir: Path) -> np.ndarray:
    return _read_table(trip_file(trip_dir, "arcs"), ["from_node", "to_node"], np.int64).to_numpy()


def load_trip(trip_dir: Path) -> Trip:
    trip_dir = Path(trip_dir)
    track = load_track(trip_dir)
    nodes = load_nodes(trip_dir)
    gps_lonlat = track[["lon", "lat"]].to_numpy()
    project = TripProjection(gps_lonlat)
    return Trip(
        trip_id=trip_dir.name,
        track=track,
        nodes=nodes,
        arcs=load_arcs(trip_dir),
        epsg=project.epsg,
        gps_xy=project(gps_lonlat),
        nodes_xy=project(nodes),
    )


def count_points(trip_dir: Path) -> int:
    """Number of rows in the .track file, without parsing it."""
    with open(trip_file(trip_dir, "track"), "rb") as f:
        return sum(1 for line in f if line.strip())
