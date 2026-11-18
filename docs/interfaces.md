# Interfaces

How the parts of the pipeline connect. P1's detailed feature and label contract is in
[`HANDOVER.md`](../HANDOVER.md); this file covers what each role hands to the next.

## Conventions
- Raw files store **longitude, latitude** (in that order).
- Trips come from all over the world (24 UTM zones, both hemispheres). Each trip is projected into
  the **UTM zone of its own GPS track** (`pathsnap_p1.geometry.choose_utm`); `x_m` / `y_m` are
  metres. Never compute distances across trips or with one fixed CRS.
- Node IDs and arc IDs are **0-based row indices, local to each trip**, never renumbered. Never use
  them as model features. Raw lon/lat must not be features either.
- Most roads appear as **two arcs with identical geometry** (A→B and B→A). Distance alone cannot
  separate them; direction of travel (`heading_difference_deg`) must.
- `trip_id` is the 8-digit folder name as a string, e.g. `"00000040"`.
- `observation_id` is the original 0-based row in the trip's `.track` file (kept after cleaning).
  `sequence_index` is the cleaned chronological order. The API calls this field `obs_idx`.
- Pipeline settings live in `config.json`; data is read from `data/` unless `--data-root` /
  `DATA_DIR` says otherwise. Never hardcode absolute paths.

## 1. Shared P1 modules (used by everyone)

| What | Function | Notes |
|---|---|---|
| Load a trip | `pathsnap_p1.data.load_trip(dir, split, config, with_reference=True)` | Returns `Trip` with `observations`, `nodes`, `arcs`, `route`, `audit`. Prediction code must pass `with_reference=False` (the `.route` file is then never opened) |
| Candidates + features | `pathsnap_p1.features.generate_features(trip, config)` | One row per (observation, candidate arc). Model inputs are exactly `pathsnap_p1.features.FEATURES` |
| Per-point labels | `pathsnap_p1.alignment.align_reference(trip, config)` | Monotonic least-squares alignment of points to the route. Used for training labels **and** the API's Top-1 scoring |
| Train / compare models | `pathsnap_p1.models.train(config, estimators={...})` | Trip-grouped CV, student-test evaluation, saves a model bundle per estimator |

## 2. Model artifacts (P2, P3 → P4)
Both files go in `artifacts/`. The API loads them once at startup.

- `model.joblib`: the bundle saved by `pathsnap_p1.models.train` or `optimisation/select_model.py --export`:
  `{"pipeline": fitted sklearn Pipeline, "features": [...], "configuration": {...}, ...}`.
  The pipeline must have `predict_proba` with class `1` = "this candidate is the correct arc".
  Optionally add `"model_name": "<name>"` for display; otherwise the classifier's class name is shown.
  `configuration` is reused at serving time, so candidates and features are generated with exactly
  the training settings.
- `hmm_config.json`: emission / transition / gap parameters for the decoder.

| In `artifacts/` | API method |
|---|---|
| nothing | nearest-arc baseline |
| `model.joblib` | `<model_name>, classifier only`: highest-scoring candidate per point (ties: lower `distance_rank`, then lower arc ID, as in training Top-1) |
| `model.joblib` + `hmm_config.json` | `<model_name> + HMM/Viterbi` (refuses to start if `hmm/viterbi.py` is missing) |
| `hmm_config.json` only | refuses to start |

Train the final model with the pinned `requirements.txt`, or it may not load on the API laptop.

## 3. Decoder (P3 → P4)
`hmm/viterbi.py` must define

```python
def decode(candidates, probs, trip, cfg) -> dict[int, int]:
    """candidates: generate_features output for one trip (observation_id, sequence_index,
    candidate_arc_id, features, projected_x_m/y_m, segment_fraction, ...);
    probs: P(label = 1) per row of candidates; trip: pathsnap_p1 Trip (nodes, arcs);
    cfg: hmm_config.json. Returns {observation_id: arc_id}.
    Observations left out are reported as -1 (no arc)."""
```

## 4. Inference (P4)
`inference.pipeline.Predictor(artifacts, pipeline_config).predict(trip_dir) -> (prediction, trip)`.
Never reads `.route`. `python -m inference.cli <trip_dir>` runs it without the API.

## 5. API (P4)
`GET /health` → `{"status": "ok", "method": str, "model_loaded": bool, "n_trips": int, "model_info": {...} | null}`

`model_info` (null for the baseline) comes from the model bundle: `model_name`, `hyperparameters`, `selected_by`,
`selection_metrics` (`cv_top1`, `cv_log_loss`, `cv_macro_f1`, `folds`), `tuning` (`method`, `trials`, `objective`), `trained_on`.
The UI sidebar shows it under "Model details".

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
  "metrics": {"scored": true, "top1_accuracy": 0.0, "route_precision": 0.0, "route_recall": 0.0,
              "route_f1": 0.0, "continuity": 0.0},
  "latency_ms": {"load": 0, "matching": 0, "scoring": 0, "total": 0}
}
```
- `method` names the pipeline that produced the prediction (`"<model_name> + HMM/Viterbi"` once a model is loaded).
- `obs` and `predicted_arc_per_obs` follow the cleaned chronological order (exact duplicate GPS rows removed).
- `predicted_arc_per_obs` uses `-1` for points with no arc; `predicted_arcs` drops those and collapses repeats.
- `latency_ms` keys follow the pipeline stages (`candidates`, `model`, `viterbi` for the full pipeline); `total` is always present.
- Reference routes are read from `REFERENCE_DIR/<trip_id>/<trip_id>.route` (default: next to the track in `EVAL_DIR`).
  A trip without one is still predicted: `scored` is `false`, the accuracy fields are `null`, continuity is still given.
- Errors: 404 unknown trip, 422 malformed trip ID, 500 pipeline failure (including a reference route that names invalid arcs).
- `python -m api.e2e_check --api <url>` runs every trip through a running API and writes a CSV.

The response never contains reference-route arcs or geometry. Metrics are computed on the server.

## Metric definitions (shared by the report and the API)
- **Top-1 accuracy:** share of labelled GPS points whose predicted arc equals the reference arc aligned
  to that point (`align_reference`). A `-1` counts as wrong; candidate misses stay in the denominator.
- **Candidate miss rate:** share of labelled GPS points whose aligned reference arc is not in the candidate set.
- **Route precision / recall / F1:** predicted arc set vs reference arc set.
- **Continuity:** share of consecutive distinct predicted arcs where the first arc's `to_node` equals the next arc's `from_node`.
