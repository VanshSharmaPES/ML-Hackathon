"""Per-trip metric projection.

Trips are spread across the world (24 UTM zones, both hemispheres), so no
single projected CRS works for all of them. Each trip is projected into the
UTM zone containing the median position of its GPS track. Within one trip
(at most about 1 degree across) UTM distance error stays well under 0.1%.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat): 326xx north, 327xx south."""
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class TripProjection:
    """Projects lon/lat arrays to metres in the trip's own UTM zone."""

    def __init__(self, reference_lonlat: np.ndarray):
        lon, lat = np.median(reference_lonlat, axis=0)
        self.epsg = utm_epsg_for(float(lon), float(lat))
        self._to_xy = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)

    def __call__(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._to_xy.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])
