# Your P1 implementation: what was done and how to use it

## 1. Where the work lives

The P1 work lives in the team repository (`pathsnap_p1/`, `tests/`, `outputs/`). The dataset is read-only input: by default `config.json` reads it from `data/` in the repository root (`data/train/`, `data/student_test/`, gitignored). Pass `--data-root <folder>` if your copy lives elsewhere. The commands below are shown for Windows PowerShell; on Linux/macOS use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

The source package is `pathsnap_p1`. It has separate modules for raw data, geometry, alignment, features, plots, orchestration and training. `HANDOVER.md` defines the interface for teammates.

## 2. What was discovered before implementation

The supplied data contains 80 training trips with **208,417 raw GPS observations**, and 10 student-test trips with **19,160 raw GPS observations**. These are dataset counts, not model scores.

Two assignment assumptions needed attention:

1. **UTM 32N is not suitable for all trips.** Coordinates span multiple continents and both hemispheres. The implementation chooses a local WGS84 UTM zone per trip. A northern trip uses EPSG 326xx; a southern trip uses 327xx. GPS and road nodes share that trip's projection. EPSG codes are retained in the output.
2. **Some reference routes are disconnected.** Raw inspection found 13 broken consecutive-arc connections in training and one in student_test. They are reported and crossings are flagged; no invented road connections are inserted.

The repository initially had no P2 harness. A reusable harness was implemented for your two models. P2 can reuse the same folds, features and metrics or adapt the handover tables to its own harness.

## 3. Read and validate the raw files (`data.py`)

For each trip:

| File | Columns | ID rule |
|---|---|---|
| `.track` | longitude, latitude, timestamp | Original row becomes `observation_id` |
| `.nodes` | longitude, latitude | Original row becomes `node_id` |
| `.arcs` | from_node, to_node | Original row becomes `arc_id` |
| `.route` | chronological arc IDs | Row is a route occurrence, not a GPS observation |

Files are whitespace-separated without headers. The loader checks column counts, numeric values, coordinate ranges and endpoint references. Invalid arcs are marked, not renumbered.

It inventories file presence and size, observation counts, missing values, timestamps, geographical extent, node/arc counts, isolated nodes, weak/strong connected components, reference availability and reference-route connectivity.

**Inspect:** `outputs/dataset_inventory.csv`, `outputs/source_manifest.csv`, `outputs/EDA_REPORT.md`.

## 4. Clean and project observations (`data.py`, `geometry.py`)

- Remove GPS records with invalid required values.
- Remove exact duplicates of longitude, latitude and timestamp.
- Keep the same location at different timestamps: that can mean the vehicle stopped.
- Sort by timestamp, preserving original observation IDs and a cleaned `sequence_index`.
- Retain and flag duplicate timestamps; speed is unavailable where dt is not positive.
- Flag gaps greater than the configured **5 seconds**. This is a declared initial threshold, not a discovered optimum.
- Project GPS and road nodes to metres in the same trip-specific CRS.

`pyproj` is preferred. When it is unavailable, a sixth-order WGS84 transverse-Mercator formula provides a bounded local fallback. It rejects unsupported extents and is checked against published PROJ examples in both hemispheres. The actual engine used is recorded in metadata. Do not describe the fallback as pyproj execution.

Original longitude/latitude remains available for maps. Map display does not use UTM coordinates as latitude/longitude.

## 5. Generate candidate roads without reference labels (`geometry.py`, `features.py`)

Road segments are indexed using SciPy's KD-tree. To avoid the error of searching only road midpoints, each road is covered by short subsegments. The search radius includes a conservative half-spacing margin, and results are filtered by **exact distance to the original finite segment**.

This method finds a long nearby road even if its original midpoint is far away. Tests compare its radius and nearest-k results against brute-force segment distances.

The final baseline retains the nearest **8 arcs per observation** (or all arcs if fewer exist). Distance ties use original arc IDs for reproducibility. This is a fixed starting policy, not a claim that eight is optimal.

Coverage experiments compare radii **25, 50 and 100 metres**, and top-k values **3, 5 and 8**. Candidate miss rate is calculated against usable derived reference labels; empty radius results count as misses. The reference road is never inserted to improve coverage.

**Inspect:** `candidate_sweep_summary.csv`, `candidate_sweep_by_trip.csv`, `figures/candidate_miss_rates.png`.

## 6. Align reference routes to GPS points (`alignment.py`)

This stage creates supervised labels for development; it is not part of inference.

For each observation and each reference-route occurrence, calculate point-to-segment distance. Dynamic programming finds a sequence with minimum total squared distance subject to nondecreasing route positions:

`cost[t,j] = distance(t,j)^2 + min(cost[t-1,k] for k <= j)`

The path can remain on an arc for several observations or skip forward across a gap. It does not force the first/last observation onto the first/last reference entry. Repeated occurrences of the same arc stay separate. Numerical arc IDs need not increase.

Outputs include the chosen reference arc, route position, alignment distance, a flag for distance >50 m, a geometric ambiguity flag, and whether an assignment crosses a known broken reference connection.

**Limitations:** parallel roads, intersections, loops, GPS gaps and wrong references can produce incorrect point labels. The geometric alignment may favour distance-based models. Report the measured scores as agreement with derived reference labels; do not claim independently verified physical ground truth.

## 7. Build the candidate feature table (`features.py`)

Each GPS point produces one row per candidate road. An observation with eight candidates produces eight rows. Features are:

1. Exact point-to-segment distance in metres.
2. Wrapped heading difference in degrees.
3. GPS speed in metres/second.
4. Arc length in metres.
5. Candidate distance rank.
6. Number of retained candidates.
7. Time gap in seconds.
8. Distance from the previous point's nearest-road projection to the current candidate projection.
9. Heading-missing indicator.
10. Speed-missing indicator.
11. Large-gap indicator.

The previous-road measure is **Euclidean distance between projected points**, not shortest-path distance. This resolves the ambiguous wording in the task. P3 can use graph distances for transition scoring.

Heading is unavailable for negligible movement (<1 m by default). Speed, heading and prior-projection context are unavailable across >5 s gaps. Missing numerical features stay missing until model preprocessing; no scaler is fitted while generating tables.

`attach_labels` separately adds `label = 1` for a candidate matching the aligned reference arc, `0` otherwise, or `-1` if a reference cannot be used. All-negative groups are retained. Metadata such as reference IDs and route positions is excluded by the explicit `FEATURES` allowlist.

**Inspect:** `processed/train/<trip>/candidates.parquet`, `feature_schema.json`, `HANDOVER.md`.

## 8. Generate maps and EDA charts (`visuals.py`)

Actual-data charts cover observation counts, trajectory lengths, sampling intervals, road-network sizes, GPS gaps, components, coordinate distribution and candidate miss rates.

Representative static trajectory figures show the road network, raw GPS, reference route and gap markers. Each arc is drawn separately so a disconnected reference does not become an invented straight connector.

The Folium generator creates separate GPS, road, reference and gap layers, a layer control, tooltips, a legend and fitted map bounds. It shows a local network bounding box for readability; feature generation uses the complete supplied network.

If Folium cannot be imported, preparation explicitly records interactive maps as pending and still produces static figures. To complete that stage after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pathsnap_p1 maps
```

Open HTML files from `outputs/maps/` in a browser. Leaflet resources and OpenStreetMap tiles need internet. A static PNG is not an interactive Folium map and must not be reported as one.

## 9. Train KNN and Gaussian Naive Bayes (`models.py`)

The shared harness uses trip-grouped five-fold cross-validation on training trips. Every candidate of a trip stays in the same fold.

For every fold, the harness fits a new pipeline:

`median imputation -> StandardScaler -> classifier`

Neither imputation nor scaling learns from validation or student-test rows. After CV, the pipeline is refitted on all training rows and evaluated on student_test. There is no hidden negative downsampling.

Starting configurations:

- **KNN:** 15 feature-space neighbours, distance weighting, Euclidean distance, KD-tree search, leaf size 40. This 15 differs from the eight geographical road candidates.
- **Gaussian NB:** `var_smoothing = 1e-9`, empirical class priors.

The harness records candidate accuracy, macro precision/recall/F1, log loss, observation Top-1, mean per-trip Top-1, candidate miss rate, fitting time and prediction time. Top-1 picks the highest candidate score with deterministic distance-rank/arc-ID tie-breaking; missing reference candidates count as failures.

Training-set scores are diagnostic only and use a fixed random sample of up to 2,500 observations with all their candidates. Fitting, cross-validation and student-test evaluation still use all rows. KNN includes self-neighbours when predicting its stored training examples, so very high resubstitution scores do not establish generalisation.

Artifacts bundle the fitted imputer, scaler, classifier, feature names, settings and data-run manifest hash. No fabricated epoch-loss plots are generated: these baselines do not have epoch-wise histories.

**Inspect:** `model_comparison.csv`, `cv_results.csv`, `cv_splits.json`, `MODEL_REPORT.md`, `figures/model_comparison.png`, `figures/model_log_loss.png`, confusion matrices and per-trip CSVs.

### Completed full-data run

The full five-fold run completed on all 80 training trips and evaluated on all 10 student-test trips. These scores measure agreement with the **derived geometric labels** described above; they are not independently verified GPS ground truth.

| Model | CV macro F1 (mean ± fold SD) | Student-test macro F1 | Student-test observation Top-1 | Student-test inference time |
|---|---:|---:|---:|---:|
| KNN | 0.8326 ± 0.0540 | 0.8749 | 0.7978 | 120.17 s |
| Gaussian Naive Bayes | 0.7239 ± 0.0241 | 0.7228 | 0.7946 | 0.09 s |

KNN had the higher candidate-level macro F1; the two models had similar observation Top-1 accuracy on these ten trips. KNN prediction is much slower, which matters if it serves live requests. The student-test candidate miss rate was 0.344%; the training top-8 miss rate was 9.072%. Review the per-trip outputs and alignment warnings before interpreting the scores. These are two model baselines, not the team's final five-model comparison.

These are your two models. The remaining models, Bayesian tuning and HMM/Viterbi are other team responsibilities.

## 10. Verification

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Checks cover published projection examples, long-road candidate correctness, finite-segment distance, angle wraparound, alignment optimality against exhaustive enumeration, duplicate/stationary handling, feature independence from `.route`, candidate-miss scoring, training-only preprocessing, grouped splits and evaluator-partition rejection.

The full run records input hashes and package versions in `outputs/source_manifest.csv` and `outputs/run_manifest.json`; measured data findings are in `outputs/EDA_REPORT.md`, and model results are in `outputs/MODEL_REPORT.md`. The continuation completed model training but did not run the test suite. The repository includes `verify_outputs.py`, `verify_projection.py` and `validate_maps.mjs` for later audits.

## 11. How you should review your work

1. Read `outputs/EDA_REPORT.md` and locate the referenced CSV rows/figures.
2. Open typical trajectory figures and inspect a trip with route breaks or large gaps.
3. Read `feature_schema.json`; explain the units and missing-value rules for every feature.
4. Read the candidate-miss comparison and discuss the radius/count trade-off.
5. Trace one observation through its candidates and reference alignment.
6. Read `model_comparison.csv` and the two report subsections. Explain discrepancies among training, CV and student-test scores.
7. Share the handover contract, code and generated data with your own teammates through the agreed repository/workflow.
8. Incorporate verified findings into the team's single PDF. Use your own interpretation, disclose AI assistance as required, and avoid claiming work from other team roles is already done.

## 12. Re-run commands

From `IMPLEMENTATION/ML-Hackathon`:

```powershell
# Rebuild EDA, alignment, candidate tables, features and figures.
.\.venv\Scripts\python.exe -m pathsnap_p1 prepare

# Run five-fold experiments and save your two fitted models.
.\.venv\Scripts\python.exe -m pathsnap_p1 train

# Generate the interactive maps once Folium is installed.
.\.venv\Scripts\python.exe -m pathsnap_p1 maps
```

Use `--output outputs/smoke --limit 2 --folds 2` with `all` for an isolated quick run. Limited-run outputs are explicitly marked and must not be substituted for final full-data results. A new config file or `--data-root` supports another teammate's dataset location.

An interrupted training run can reuse completed cross-validation folds with `train --resume`. The harness checks the input-data manifest, model settings, feature list, configuration and trip folds before reuse. It refits final models and evaluates student_test normally.
