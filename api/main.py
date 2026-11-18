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

from inference.artifacts import load_artifacts
from inference.pipeline import Predictor

from . import config
from .schemas import HealthResponse, PredictRequest, PredictResponse, TripInfo, TripsResponse
from .scoring import score_trip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pathsnap.api")


def track_file(trip_dir: Path) -> Path:
    return trip_dir / f"{trip_dir.name}.track"


def count_points(trip_dir: Path) -> int:
    """Number of rows in the .track file, without parsing it."""
    with open(track_file(trip_dir), "rb") as f:
        return sum(1 for line in f if line.strip())


def list_trip_ids() -> list[str]:
    """Trip folders currently in EVAL_DIR. Read on every call, never hardcoded."""
    if not config.EVAL_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in config.EVAL_DIR.iterdir()
        if p.is_dir() and config.TRIP_ID_PATTERN.match(p.name) and track_file(p).is_file()
    )


def resolve_trip(trip_id: str) -> Path:
    if trip_id not in list_trip_ids():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} is not in the evaluation folder.")
    return config.EVAL_DIR / trip_id


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and configs once, not per request.
    app.state.predictor = Predictor(load_artifacts(config.ARTIFACTS_DIR), config.load_pipeline_config())
    trips = list_trip_ids()
    log.info("PathSnap API starting | method=%s | artifacts=%s | eval_dir=%s | trips=%d",
             app.state.predictor.method, config.ARTIFACTS_DIR, config.EVAL_DIR, len(trips))
    if not app.state.predictor.model_loaded:
        log.warning("No model in %s. Serving the nearest-arc baseline.", config.ARTIFACTS_DIR)
    if not trips:
        log.warning("No trips found in %s. Set EVAL_DIR to the folder holding the trip folders.", config.EVAL_DIR)
    yield
    log.info("PathSnap API stopped")


app = FastAPI(title="PathSnap API", version="0.3.0", lifespan=lifespan)

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
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(status="ok", method=predictor.method, model_loaded=predictor.model_loaded,
                          n_trips=len(list_trip_ids()))


@app.get("/trips", response_model=TripsResponse)
def trips() -> TripsResponse:
    return TripsResponse(trips=[
        TripInfo(trip_id=tid, n_points=count_points(config.EVAL_DIR / tid)) for tid in list_trip_ids()
    ])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    trip_dir = resolve_trip(req.trip_id)
    predictor = request.app.state.predictor
    start = time.perf_counter()
    try:
        prediction, trip = predictor.predict(trip_dir)
        t = time.perf_counter()
        metrics = score_trip(trip_dir, trip, prediction, predictor.config)
        scoring_ms = (time.perf_counter() - t) * 1000
    except Exception as exc:
        log.exception("Prediction failed | trip=%s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed for trip {req.trip_id}: {exc}") from exc

    latency = {**prediction["timings"], "scoring": scoring_ms, "total": (time.perf_counter() - start) * 1000}
    log.info(
        "PREDICT trip=%s | method=%s | points=%d | route_arcs=%d | top1=%s | route_f1=%s | continuity=%s | %.2f s",
        req.trip_id, prediction["method"], len(prediction["obs"]), len(prediction["predicted_arcs"]),
        fmt(metrics["top1_accuracy"]), fmt(metrics["route_f1"]), fmt(metrics["continuity"]),
        latency["total"] / 1000,
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
