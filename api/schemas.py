"""Request and response models. Shapes follow docs/interfaces.md."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    # allow a field named model_* (Pydantic reserves that prefix by default)
    model_config = ConfigDict(protected_namespaces=())

    status: str
    method: str
    model_loaded: bool
    n_trips: int


class TripInfo(BaseModel):
    trip_id: str
    n_points: int


class TripsResponse(BaseModel):
    trips: list[TripInfo]


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$", examples=["00000040"])


class Observation(BaseModel):
    obs_idx: int
    lon: float
    lat: float
    t: float


class Metrics(BaseModel):
    # None until P1's point-to-route alignment provides per-point labels
    top1_accuracy: Optional[float] = None
    route_precision: float
    route_recall: float
    route_f1: float
    continuity: Optional[float] = None


class PredictResponse(BaseModel):
    trip_id: str
    method: str
    obs: list[Observation]
    predicted_arc_per_obs: list[int]
    predicted_arcs: list[int]
    geojson: dict
    metrics: Metrics
    latency_ms: dict[str, float]
