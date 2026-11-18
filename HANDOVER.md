# P1 handover contract for P2, P3 and backend

## Identity

- A road arc is identified by `(trip_id, candidate_arc_id)` within its supplied partition. Do not treat local arc ID 10 in different networks as the same physical road.
- An observation is identified by `(split, trip_id, observation_id)`. `observation_id` is the original zero-based `.track` row ID, even after cleaning. `sequence_index` gives cleaned chronological order.
- A candidate row is identified by `(split, trip_id, observation_id, candidate_arc_id)`.
- Node and arc row IDs remain unchanged. Invalid arcs are marked/excluded, not renumbered.
- Every trip has its own EPSG. Do not compute distances across different projected coordinate systems.

## Model feature columns (strict allowlist)

Use `pathsnap_p1.features.FEATURES`, never all numeric columns automatically:

| Column | Unit / meaning |
|---|---|
| `distance_m` | Exact distance from GPS point to finite candidate segment, metres |
| `heading_difference_deg` | Wrapped absolute difference in [0,180] between GPS motion and directed arc orientation |
| `speed_mps` | Consecutive GPS displacement / positive dt; missing across large gaps |
| `arc_length_m` | Projected straight-line distance between supplied arc endpoints |
| `distance_rank` | One-based candidate distance rank; ties broken by original arc ID |
| `candidate_count` | Number of retained candidates for that observation |
| `time_gap_s` | Current timestamp minus preceding cleaned timestamp |
| `previous_nearest_distance_m` | Euclidean distance from previous observation's nearest-road projected point to this candidate's projected point; missing across gaps/invalid dt |
| `heading_missing` | Indicator that GPS heading is unavailable or unreliable |
| `speed_missing` | Indicator that speed is unavailable |
| `large_gap` | Indicator dt > configured threshold |

Direction is computed in projected x/y coordinates with atan2(dy,dx). Both GPS and road headings use the same convention. No speed/heading is invented for the first point. Arc-length and heading use the supplied straight segments; no external road geometry is downloaded.

`candidate_count` is normally eight under the fixed top-8 baseline; it is retained for a stable interface and becomes useful if the candidate policy later changes.

The prior-road feature is a deliberately documented geometric measure, **not network shortest-path distance**. P3 should use its graph for transition distances.

## Targets and quality metadata

- `reference_arc_id`: derived aligned reference arc, or -1 if unavailable.
- `label`: 1 for the candidate matching that reference, 0 for another candidate, -1 if no usable reference.
- `route_position`: occurrence index in `.route`, not an arc ID. Repeated arcs remain distinct occurrences.
- `alignment_distance_m`, `alignment_far`, `alignment_ambiguous`, `crosses_reference_break`: quality flags, NEVER model inputs.
- `reference_in_candidates` in the observations table: whether the candidate generator covered the aligned reference road.

Unlabelled rows (`label == -1`) are excluded from supervised metrics; their counts and unusable references must still be reported. All-negative candidate groups are retained. Do not insert missing reference roads into the candidate set. Pointwise Top-1 uses all labelled observations, including misses.

## P2 training

```python
from pathsnap_p1.models import train
from sklearn.ensemble import RandomForestClassifier

# config must contain resolved data_root and output_dir paths and the normal settings.
train(config, estimators={
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2)
})
```

Use a separate output/report naming strategy when combining teams' experiments: the harness writes model_comparison.csv for the models supplied in that call. Do not overwrite P1's comparison accidentally. Reuse cv_splits.json or GroupKFold by trip, and fit learned preprocessing within each training fold.

P1's candidate metrics use binary candidate labels; Top-1 is the chosen road per observation. Compare them consistently. The supplied harness does not implement Bayesian optimisation or sequence-level route evaluation.

## P3 decoder

For each trip, sort by `sequence_index`, group candidates by observation, and use model `predict_proba` with its stored feature allowlist. Candidate probabilities are independent binary scores, not already normalised across an observation's road choices. Define/calibrate your emission conversion explicitly.

Available geometry: `candidate_arc_id`, `segment_fraction` in [0,1], `projected_x_m`, `projected_y_m`, and per-trip `arcs.parquet` / `nodes.parquet`.

Preserve candidate-to-arc correspondence through decoding. If scoring transition distances between points on arcs, include partial-arc distances and arc direction. Handle long gaps/disconnected transitions explicitly. Do not use alignment labels as hidden-state inputs.

## Backend inference without reference files

```python
from pathsnap_p1.data import load_trip
from pathsnap_p1.features import generate_features
import joblib

trip = load_trip(trip_directory, "inference", config, with_reference=False)
candidates, _, _, _ = generate_features(trip, config)
artifact = joblib.load(model_path)  # Load only your team's trusted artifact.
scores = artifact["pipeline"].predict_proba(candidates[artifact["features"]])[:, 1]
```

This accepts a single explicitly selected trip. Development discovery still refuses evaluator partitions. The backend must isolate server-side scoring and references from this inference path. Never send evaluator reference labels to the UI.
