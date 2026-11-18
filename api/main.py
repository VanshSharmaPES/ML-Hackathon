"""PathSnap API (Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log

Every request is logged to this terminal (evaluators check these logs).
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from inference.pipeline import METHOD, predict_trip
from inference.trip_io import count_points, trip_file

from . import config
from .schemas import HealthResponse, PredictRequest, PredictResponse, TripInfo, TripsResponse
from .scoring import score_trip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pathsnap.api")


def list_trip_ids() -> list[str]:
    """Trip folders currently in EVAL_DIR. Read on every call, never hardcoded."""
    if not config.EVAL_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in config.EVAL_DIR.iterdir()
        if p.is_dir() and config.TRIP_ID_PATTERN.match(p.name) and trip_file(p, "track").is_file()
    )


def resolve_trip(trip_id: str) -> Path:
    if trip_id not in list_trip_ids():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} is not in the evaluation folder.")
    return config.EVAL_DIR / trip_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    trips = list_trip_ids()
    log.info("PathSnap API starting | method=%s | eval_dir=%s | trips=%d", METHOD, config.EVAL_DIR, len(trips))
    if not trips:
        log.warning("No trips found in %s. Set EVAL_DIR to the folder holding the trip folders.", config.EVAL_DIR)
    yield
    log.info("PathSnap API stopped")


app = FastAPI(title="PathSnap API", version="0.1.0", lifespan=lifespan)

# The UI runs on another laptop, so allow cross-origin calls on the LAN.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s | client=%s | unhandled error", request.method, request.url.path, client)
        raise
    elapsed = (time.perf_counter() - start) * 1000
    log.info("%s %s | client=%s | status=%d | %.0f ms",
             request.method, request.url.path, client, response.status_code, elapsed)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", method=METHOD, model_loaded=False, n_trips=len(list_trip_ids()))


@app.get("/trips", response_model=TripsResponse)
def trips() -> TripsResponse:
    return TripsResponse(trips=[
        TripInfo(trip_id=tid, n_points=count_points(config.EVAL_DIR / tid)) for tid in list_trip_ids()
    ])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    trip_dir = resolve_trip(req.trip_id)
    start = time.perf_counter()
    try:
        prediction = predict_trip(trip_dir)
        t = time.perf_counter()
        metrics = score_trip(trip_dir, prediction)
        scoring_ms = (time.perf_counter() - t) * 1000
    except Exception as exc:
        log.exception("Prediction failed | trip=%s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed for trip {req.trip_id}: {exc}") from exc

    latency = {**prediction["timings"], "scoring": scoring_ms, "total": (time.perf_counter() - start) * 1000}
    log.info(
        "PREDICT trip=%s | points=%d | route_arcs=%d | route_f1=%.3f | continuity=%s | %.2f s",
        req.trip_id, len(prediction["obs"]), len(prediction["predicted_arcs"]), metrics["route_f1"],
        "n/a" if metrics["continuity"] is None else f"{metrics['continuity']:.3f}", latency["total"] / 1000,
    )
    return PredictResponse(
        trip_id=req.trip_id,
        method=prediction["method"],
        obs=prediction["obs"],
        predicted_arc_per_obs=prediction["predicted_arc_per_obs"],
        predicted_arcs=prediction["predicted_arcs"],
        geojson=prediction["geojson"],
        metrics=metrics,
        latency_ms=latency,
    )
