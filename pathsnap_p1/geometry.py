"""Metre-based geometry and an exact segment search backed by a KD-tree.

No road-midpoint approximation: the spatial index samples each segment at
bounded spacing, expands the search by half that spacing, then computes exact
distances to the original finite segments. IDs always remain original arc IDs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def choose_utm(lon, lat):
    longitude, latitude = float(np.median(lon)), float(np.median(lat))
    if not -80 <= latitude <= 84:
        raise ValueError("UTM is unsuitable outside 80 S to 84 N.")
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        zone = 32
    if 72 <= latitude < 84:
        for lo, hi, special in [(0, 9, 31), (9, 21, 33), (21, 33, 35), (33, 42, 37)]:
            if lo <= longitude < hi:
                zone = special
    return (32600 if latitude >= 0 else 32700) + zone


def project_utm(lon, lat, epsg, engine='series'):
    """Prefer PROJ; provide a WGS84 sixth-order TM series for offline use.

    Fallback is limited to <= 6 degrees from the central meridian. It is
    checked against published PROJ examples in tests; not for surveying.
    """
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    zone, south = epsg % 100, epsg // 100 == 327
    if epsg // 100 not in (326, 327) or not 1 <= zone <= 60:
        raise ValueError("Expected a WGS84 UTM EPSG code.")
    delta = (lon - (zone * 6 - 183) + 180) % 360 - 180
    if np.any(np.abs(delta) > 6) or np.any((lat < -80) | (lat > 84)):
        raise ValueError("Trip/network extends outside supported local UTM extent.")
    if engine not in ('series','pyproj','auto'):
        raise ValueError('Projection engine must be series, pyproj or auto.')
    Transformer = None
    if engine != 'series':
        try:
            from pyproj import Transformer
        except ImportError:
            if engine == 'pyproj':raise
    if Transformer is not None:
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        return np.column_stack([x, y]), "pyproj"
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = np.radians(lat)
    n = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    t, c = np.tan(phi) ** 2, ep2 * np.cos(phi) ** 2
    A = np.cos(phi) * np.radians(delta)
    m = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * phi)
             + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * phi)
             - 35 * e2**3 / 3072 * np.sin(6 * phi))
    x = 500000 + k0 * n * (A + (1 - t + c) * A**3 / 6
                            + (5 - 18*t + t*t + 72*c - 58*ep2) * A**5 / 120)
    y = k0 * (m + n * np.tan(phi) * (A*A/2 + (5-t+9*c+4*c*c)*A**4/24
                     + (61-58*t+t*t+600*c-330*ep2)*A**6/720))
    if south:
        y += 10000000
    return np.column_stack([x, y]), "wgs84_tm_series_fallback"


def point_segment(point, starts, ends):
    vector = ends - starts
    length2 = np.einsum("ij,ij->i", vector, vector)
    fraction = np.divide(np.einsum("ij,ij->i", point - starts, vector), length2,
                         out=np.zeros(len(starts)), where=length2 > 0)
    fraction = np.clip(fraction, 0, 1)
    projected = starts + fraction[:, None] * vector
    return np.linalg.norm(projected - point, axis=1), fraction, projected


def angle_difference(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


@dataclass
class RoadIndex:
    arc_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    spacing: float = 50.0

    def __post_init__(self):
        if not len(self.arc_ids) or self.spacing <= 0:
            raise ValueError("Road index needs arcs and positive sample spacing.")
        self.lengths = np.linalg.norm(self.ends - self.starts, axis=1)
        self.headings = np.degrees(np.arctan2((self.ends-self.starts)[:, 1],
                                              (self.ends-self.starts)[:, 0])) % 360
        steps = np.maximum(1, np.ceil(self.lengths / self.spacing).astype(int))
        # Midpoints of short subsegments cover every location within spacing/2.
        self.sample_to_arc = np.repeat(np.arange(len(steps)), steps)
        offsets = np.repeat(np.cumsum(steps) - steps, steps)
        within = np.arange(steps.sum()) - offsets
        fractions = (within + .5) / steps[self.sample_to_arc]
        samples = self.starts[self.sample_to_arc] + fractions[:, None] * (
            self.ends-self.starts)[self.sample_to_arc]
        self.tree = cKDTree(samples)

    def radius(self, point, radius):
        sample_ids = self.tree.query_ball_point(point, radius + self.spacing / 2 + 1e-8)
        ids = np.unique(self.sample_to_arc[sample_ids])
        if not len(ids):
            return ids, np.array([], dtype=float)
        distances, _, _ = point_segment(point, self.starts[ids], self.ends[ids])
        keep = distances <= radius + 1e-8
        ids, distances = ids[keep], distances[keep]
        order = np.lexsort((self.arc_ids[ids], distances))
        return ids[order], distances[order]

    def nearest(self, point, k):
        k = min(int(k), len(self.arc_ids))
        if k < 1:
            raise ValueError("k must be positive")
        nearest_sample = self.tree.query(point, k=1)[0]
        radius = max(self.spacing / 2, float(nearest_sample) + self.spacing / 2)
        while True:
            ids, distances = self.radius(point, radius)
            if len(ids) >= k:
                return ids[:k], distances[:k]
            radius *= 2
