# Interfaces

## Conventions
- Raw files store **longitude, latitude** (in that order).
- Trips come from all over the world (24 UTM zones, both hemispheres). Each trip is projected into the **UTM zone of its own GPS track** (`inference.geo.TripProjection`); `x`/`y` are metres. Never use one fixed CRS for all trips.
- Node IDs and arc IDs are **0-based row indices, local to each trip**. Never use them as model features. Raw lon/lat must not be features either.
- Most roads appear as **two arcs with identical geometry** (A→B and B→A). Distance alone cannot separate them; direction of travel must.
- `trip_id` is the 8-digit folder name as a string, e.g. `"00000040"`.
- `obs_idx` is the 0-based row index in the trip's `.track` file.
- Paths come from `DATA_DIR` (default `./data`); never hardcode absolute paths.

## 0. Trip object (shared by everyone)
`inference.trip_io.load_trip(trip_dir) -> Trip` reads `.track`, `.nodes`, `.arcs` (never `.route`):

| Field | Shape | Meaning |
|---|---|---|
| `trip_id` | str | Folder name |
| `track` | DataFrame | `obs_idx, lon, lat, t` |
| `nodes`, `nodes_xy` | (n_nodes, 2) | lon/lat and metres |
| `arcs` | (n_arcs, 2) int | `from_node, to_node` |
| `gps_xy` | (n_points, 2) | metres, same row order as `track` |
| `epsg` | int | UTM zone used |

## 1. Labels and candidates (P1 → P2, P3, P4)
- **Per-point labels:** `data_processing.route_alignment.align_points_to_route(gps_xy, nodes_xy, arcs, route) -> (aligned_arc_id per point, distance_m)`. Training labels and the API's Top-1 scoring both use this function.
- **Candidates:** `features/candidates.py` must define
  `build_candidates(trip: Trip, cfg: dict) -> pd.DataFrame`, one row per (GPS point, candidate arc), with at least `obs_idx`, `arc_id` and every column in `features.json`. `cfg` is `artifacts/hmm_config.json`.
- Training table (parquet, local only, not committed) adds `trip_id` and `label` (1 if `arc_id` equals the aligned arc for that point).
- Points with zero candidates are recorded during training (for candidate miss rate).

## 2. Model artifacts (P2, P3 → P4)
All three files go in `artifacts/`. The API loads them once at startup.
- `model.joblib`: fitted scikit-learn `Pipeline` (preprocessing + model) with `predict_proba` and `classes_` containing `1`.
- `features.json`: `{"features": [ordered column names], "model_name": str, "trained_on": "train (80 trips)", "seed": int}`
- `hmm_config.json`: candidate settings (radius, k) and emission/transition/gap parameters.

None present → API serves the nearest-arc baseline. Some but not all present → API refuses to start.

## 3. Decoder (P3 → P4)
`hmm/viterbi.py` must define
```python
def decode(cand_df, probs, trip, cfg) -> dict[int, int]:
    """cand_df: build_candidates output; probs: P(label=1) per row of cand_df;
    trip: Trip; cfg: hmm_config.json. Returns {obs_idx: arc_id}.
    Points left out are reported as -1 (no arc)."""
```

## 4. Inference (P4)
`inference.pipeline.Predictor(artifacts).predict(trip_dir) -> dict`. Never reads `.route`.

## 5. API (P4)
`GET /health` → `{"status": "ok", "method": str, "model_loaded": bool, "n_trips": int}`

`GET /trips` → `{"trips": [{"trip_id": str, "n_points": int}]}`

`POST /predict` with `{"trip_id": "<8 digits>"}` →
```json
{
  "trip_id": "00000040",
  "method": "nearest-arc baseline",
  "obs": [{"obs_idx": 0, "lon": 61.133155, "lat": 55.416221, "t": 1.0}],
  "predicted_arc_per_obs": [12, 12, 13],
  "predicted_arcs": [12, 13, 57],
  "geojson": {"type": "FeatureCollection", "features": []},
  "metrics": {"top1_accuracy": 0.0, "route_precision": 0.0, "route_recall": 0.0,
              "route_f1": 0.0, "continuity": 0.0},
  "latency_ms": {"load": 0, "matching": 0, "scoring": 0, "total": 0}
}
```
- `method` names the pipeline that produced the prediction (`"<model_name> + HMM/Viterbi"` once a model is loaded).
- `predicted_arc_per_obs` uses `-1` for points with no arc; `predicted_arcs` drops those and collapses repeats.
- `latency_ms` keys follow the pipeline stages (`candidates`, `model`, `viterbi` for the full pipeline); `total` is always present.
- Errors: 404 unknown trip, 422 malformed trip ID, 500 pipeline failure.

The response never contains reference-route arcs or geometry. Metrics are computed on the server.

## Metric definitions (shared by the report and the API)
- **Top-1 accuracy:** share of GPS points whose predicted arc equals the reference arc aligned to that point (`align_points_to_route`). A `-1` counts as wrong.
- **Candidate miss rate:** share of GPS points whose aligned reference arc is not in the candidate set.
- **Route precision / recall / F1:** predicted arc set vs reference arc set.
- **Continuity:** share of consecutive distinct predicted arcs where the first arc's `to_node` equals the next arc's `from_node`.
