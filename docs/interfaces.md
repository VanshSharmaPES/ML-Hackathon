# Interfaces

## Conventions
- Raw files store **longitude, latitude** (in that order).
- Trips come from all over the world (24 UTM zones, both hemispheres). Project each trip into the **UTM zone of its own GPS track** with `inference.geo.TripProjection`; `x`/`y` are metres. Never use one fixed CRS for all trips.
- Node IDs and arc IDs are **0-based row indices, local to each trip**. Never use them as model features. Raw lon/lat must not be features either.
- Most roads appear as **two arcs with identical geometry** (A→B and B→A). Distance alone cannot separate them; direction of travel must.
- `trip_id` is the 8-digit folder name as a string, e.g. `"00000040"`.
- `obs_idx` is the 0-based row index in the trip's `.track` file.
- Paths come from `DATA_DIR` (default `./data`); never hardcode absolute paths.

## 1. Candidate table (P1 → P2, P3)
Parquet, one row per (GPS point, candidate arc). Saved locally under `data/`; not committed.

| Column | Type | Meaning |
|---|---|---|
| `trip_id` | str | Trip folder name |
| `obs_idx` | int | GPS point index |
| `t` | float | Timestamp (s) |
| `x`, `y` | float | Projected GPS position (m, trip's UTM zone) |
| `arc_id` | int | Candidate arc (local to trip) |
| `<features…>` | float | Listed in `artifacts/features.json` |
| `label` | int | 1 if `arc_id` is the aligned reference arc for this point, else 0 |

Every `obs_idx` with zero candidates must still be recorded (for candidate miss rate).

## 2. Model artifact (P2 → P3, P4)
- `artifacts/model.joblib`: a fitted scikit-learn `Pipeline` (preprocessing + model) with `predict_proba`.
- `artifacts/features.json`: `{"features": [ordered column names], "model_name": str, "trained_on": "train (80 trips)", "seed": int}`

## 3. Decoder (P3 → P4)
```python
def decode(cand_df, probs, graph, params) -> dict[int, int]:
    """cand_df: candidate rows for one trip; probs: P(label=1) per row;
    graph: the trip's road network; params: loaded from artifacts/hmm_config.json.
    Returns {obs_idx: predicted arc_id} for every decodable point."""
```
- `artifacts/hmm_config.json`: emission/transition parameters, candidate radius, gap thresholds.

## 4. Inference (P4)
```python
def predict_trip(trip_dir: Path) -> dict
```
Reads `.track`, `.nodes`, `.arcs` only. **Never reads `.route`.**

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
  "metrics": {"top1_accuracy": null, "route_precision": 0.0, "route_recall": 0.0,
              "route_f1": 0.0, "continuity": 0.0},
  "latency_ms": {"load": 0, "matching": 0, "scoring": 0, "total": 0}
}
```
- `method` names the pipeline that produced the prediction.
- `predicted_arcs` is `predicted_arc_per_obs` with consecutive repeats collapsed.
- `latency_ms` keys depend on the pipeline stages; `total` is always present.
- `top1_accuracy` is `null` until per-point labels from the alignment step exist.
- Errors: 404 unknown trip, 422 malformed trip ID, 500 pipeline failure.

The response never contains reference-route arcs or geometry. Metrics are computed on the server.

## Metric definitions (shared by the report and the API)
- **Top-1 accuracy:** share of GPS points whose highest-scoring candidate equals the aligned reference arc.
- **Candidate miss rate:** share of GPS points whose reference arc is not in the candidate set.
- **Route precision / recall / F1:** predicted arc set vs reference arc set.
- **Continuity:** share of consecutive distinct predicted arcs where the first arc's `to_node` equals the next arc's `from_node`.
