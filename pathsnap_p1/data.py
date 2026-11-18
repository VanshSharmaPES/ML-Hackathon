"""Raw loaders preserve row IDs and never read evaluator partitions."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geometry import choose_utm, project_utm

ALLOWED_SPLITS = ("train", "student_test")


def read_numeric(path, columns):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=columns, dtype=float)
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(columns):
        raise ValueError(f"{path}: expected {len(columns)} columns, got {frame.shape[1]}")
    frame.columns = columns
    return frame.apply(pd.to_numeric, errors="coerce")


@dataclass
class Trip:
    split: str
    trip_id: str
    observations: pd.DataFrame
    rejected: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray | None
    audit: dict


def discover(root, split):
    if split not in ALLOWED_SPLITS:
        raise ValueError("Only train/student_test are permitted for development.")
    directory = Path(root) / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(p for p in directory.iterdir() if p.is_dir())


def load_trip(directory, split, config, with_reference=True):
    """with_reference=False guarantees .route is not opened (backend entry point)."""
    if with_reference and split not in ALLOWED_SPLITS:
        raise ValueError('Reference loading is restricted to development partitions.')
    directory = Path(directory)
    tid = directory.name
    paths = {ext: directory / f"{tid}.{ext}" for ext in ("track", "nodes", "arcs", "route")}
    raw = read_numeric(paths['track'], ['longitude', 'latitude', 'timestamp'])
    raw.insert(0, 'observation_id', np.arange(len(raw)))
    invalid = (~np.isfinite(raw[['longitude','latitude','timestamp']]).all(axis=1)
               | ~raw.longitude.between(-180,180) | ~raw.latitude.between(-80,84))
    duplicated = raw.duplicated(['longitude','latitude','timestamp']) & ~invalid
    rejected = raw.loc[invalid | duplicated].copy()
    rejected['reason'] = np.where(invalid[invalid | duplicated], 'invalid_observation', 'exact_duplicate')
    observations = raw.loc[~invalid & ~duplicated].sort_values(['timestamp','observation_id'], kind='stable').copy()
    observations.reset_index(drop=True, inplace=True)
    if observations.empty:
        raise ValueError(f"{tid}: no usable observations")
    epsg = choose_utm(observations.longitude, observations.latitude)
    if config.get('projection') == 'utm32n':
        if epsg != 32632:
            raise ValueError(f"{tid}: UTM32N requested but trip's suitable CRS is EPSG:{epsg}")
        epsg = 32632
    projection_engine=config.get('projection_engine','series')
    xy, engine = project_utm(observations.longitude, observations.latitude, epsg,projection_engine)
    observations[['x_m','y_m']] = xy
    observations['sequence_index'] = np.arange(len(observations))
    observations['time_gap_s'] = observations.timestamp.diff()
    observations['large_gap'] = observations.time_gap_s > config['gap_seconds']
    observations['nonpositive_dt'] = observations.time_gap_s <= 0
    movement = np.vstack([[np.nan,np.nan], np.diff(xy, axis=0)])
    step = np.linalg.norm(movement, axis=1)
    observations['step_distance_m'] = step
    valid_motion = (observations.time_gap_s > 0) & ~observations.large_gap
    observations['speed_mps'] = np.where(valid_motion, step / observations.time_gap_s.replace(0,np.nan), np.nan)
    valid_heading = valid_motion & (step >= config['stationary_threshold_m'])
    observations['heading_deg'] = np.where(valid_heading, np.degrees(np.arctan2(movement[:,1],movement[:,0])) % 360, np.nan)
    nodes = read_numeric(paths['nodes'], ['longitude','latitude'])
    nodes.insert(0,'node_id', np.arange(len(nodes)))
    node_valid = np.isfinite(nodes[['longitude','latitude']]).all(axis=1) & nodes.longitude.between(-180,180) & nodes.latitude.between(-80,84)
    nodes[['x_m','y_m']] = np.nan
    projected, _ = project_utm(nodes.loc[node_valid,'longitude'],nodes.loc[node_valid,'latitude'],epsg,projection_engine)
    nodes.loc[node_valid,['x_m','y_m']] = projected
    arcs = read_numeric(paths['arcs'], ['from_node','to_node'])
    arcs.insert(0,'arc_id',np.arange(len(arcs)))
    endpoints = arcs[['from_node','to_node']].to_numpy(float)
    arc_valid = np.isfinite(endpoints).all(axis=1) & (endpoints == np.floor(endpoints)).all(axis=1) & (endpoints >= 0).all(axis=1) & (endpoints < len(nodes)).all(axis=1)
    valid_ids = np.flatnonzero(arc_valid)
    ep = endpoints[valid_ids].astype(int)
    arc_valid[valid_ids] &= node_valid.to_numpy()[ep].all(axis=1)
    arcs['valid'] = arc_valid
    arcs[['from_node','to_node']] = arcs[['from_node','to_node']].replace([np.inf,-np.inf],np.nan).fillna(-1).astype('int64')
    valid_arcs = arcs.loc[arc_valid]
    ep = valid_arcs[['from_node','to_node']].to_numpy(int)
    graph = coo_matrix((np.ones(len(ep)), (ep[:,0],ep[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    weak_n, components = connected_components(graph, directed=True, connection='weak')
    strong_n = connected_components(graph, directed=True, connection='strong',return_labels=False)
    degree = np.bincount(ep.ravel(), minlength=len(nodes))
    nodes['weak_component'] = components
    nodes['degree'] = degree
    route = None
    invalid_route = breaks = route_length = repeated_route_arcs = 0
    if with_reference and paths['route'].exists():
        values = read_numeric(paths['route'], ['arc_id']).arc_id.to_numpy(float)
        route_length = len(values)
        usable = np.isfinite(values) & (values == np.floor(values)) & (values >= 0) & (values < len(arcs))
        indices = np.flatnonzero(usable)
        usable[indices] &= arc_valid[values[indices].astype(int)]
        invalid_route = int((~usable).sum())
        if not invalid_route and len(values):
            route = values.astype(int)
            route_arcs = arcs.iloc[route]
            breaks = int(np.sum(route_arcs.to_node.to_numpy()[:-1] != route_arcs.from_node.to_numpy()[1:]))
            repeated_route_arcs = len(route) - len(np.unique(route))
    audit = dict(split=split,trip_id=tid,raw_observations=len(raw),clean_observations=len(observations),
                 invalid_observations=int(invalid.sum()),exact_duplicates=int(duplicated.sum()),
                 missing_track_values=int(raw.isna().sum().sum()),
                 backwards_timestamps=int((raw.timestamp.diff()<0).sum()),
                 duplicate_timestamps=int(observations.timestamp.duplicated().sum()),
                 large_gaps=int(observations.large_gap.sum()),max_gap_s=float(observations.time_gap_s.max()),
                 median_gap_s=float(observations.time_gap_s.median()),
                 fraction_1s=float((observations.time_gap_s.iloc[1:] == 1).mean()),
                 duration_s=float(observations.timestamp.iloc[-1]-observations.timestamp.iloc[0]),
                 observed_length_m=float(np.nansum(step)),nodes=len(nodes),arcs=len(arcs),
                 invalid_nodes=int((~node_valid).sum()),invalid_arcs=int((~arc_valid).sum()),
                 isolated_nodes=int((degree==0).sum()),weak_components=int(weak_n),strong_components=int(strong_n),
                 largest_weak_component_nodes=int(np.bincount(components).max()),
                 route_entries=route_length,invalid_route_entries=invalid_route,route_breaks=breaks,
                 repeated_route_arcs=repeated_route_arcs,reference_usable=route is not None,
                 lon_min=float(observations.longitude.min()),lon_max=float(observations.longitude.max()),
                 lat_min=float(observations.latitude.min()),lat_max=float(observations.latitude.max()),
                 epsg=epsg,projection_engine=engine)
    for ext,p in paths.items():
        if ext != 'route' or with_reference:
            audit[f'{ext}_present']=p.exists()
            audit[f'{ext}_bytes']=p.stat().st_size if p.exists() else 0
    return Trip(split,tid,observations,rejected,nodes,arcs,route,audit)
