# PathSnap — P1 Data & Features

P1 implementation: EDA, cleaning, per-trip coordinate projection, monotonic reference-label alignment, exact road candidate search, feature tables, geographical visualisation, KNN and Gaussian Naive Bayes experiments.

Read **STEP_BY_STEP_GUIDE.md** for the complete walkthrough and **HANDOVER.md** for P2/P3 integration.

## Run on Windows

Open a terminal in this repository (`IMPLEMENTATION/ML-Hackathon`). The existing `.venv` uses the installed Python 3.14 scientific packages. Activation is unnecessary:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pathsnap_p1 prepare
.\.venv\Scripts\python.exe -m pathsnap_p1 train
.\.venv\Scripts\python.exe -m pathsnap_p1 maps
```

`prepare` and `train` together are also available as `python -m pathsnap_p1 all`.

For a new machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pathsnap_p1 all --data-root "C:\path\to\Student folder"
```

The data-root directory must contain `train` and `student_test`. The default in `config.json` matches this workspace's supplied folder. Paths in that file resolve relative to the config file, not an arbitrary terminal location.

For a small isolated run:

```powershell
.\.venv\Scripts\python.exe -m pathsnap_p1 all --limit 2 --output outputs/smoke --folds 2
```

## Main outputs

- `outputs/EDA_REPORT.md`: measured dataset findings, coverage and alignment limitations.
- `outputs/MODEL_REPORT.md`: actual KNN/Gaussian NB experiments and report section 3.2 drafts.
- `outputs/model_comparison.csv`, `cv_results.csv`, `cv_splits.json`: reproducible comparison evidence.
- `outputs/dataset_inventory.csv`: one row per trip, including file sizes, missing values, network connectivity and route breaks.
- `outputs/candidate_sweep_summary.csv`: radius/top-k coverage comparisons.
- `outputs/processed/<partition>/<trip>/candidates.parquet`: candidate features and separately attached labels.
- `outputs/processed/<partition>/<trip>/observations.parquet`: observation alignment and quality flags.
- `outputs/processed/<partition>/<trip>/nodes.parquet`, `arcs.parquet`: ID-preserving network handover.
- `outputs/models/*.joblib`: saved fitted preprocessing + model bundles.
- `outputs/predictions/`: student-test candidate probabilities and observation predictions.
- `outputs/figures/`: real EDA, trajectory, comparison and confusion-matrix plots.
- `outputs/maps/`: interactive Folium maps, when Folium is installed.
- `outputs/source_manifest.csv`, `run_manifest.json`, `feature_schema.json`: input hashes, versions, configurations and schema.

## Scope and important decisions

Only the 80 training and 10 student-test trips are used. There is no evaluator-data development path. The inference loader supports `with_reference=False`; tests verify that changing `.route` cannot change generated features.

The screenshot's fixed UTM 32N assumption is contradicted by the real dataset, which spans multiple continents and both hemispheres. The implementation chooses a local WGS84 UTM zone per trip. Metadata records the EPSG code and projection implementation. `pyproj` is preferred; a tested numerical fallback supports offline execution within a bounded local extent.

Reference labels are **derived**, using an ordered geometric alignment. They are not independently verified GPS-level truth. Scores can be biased toward geometric features. Ambiguity, long alignment distances and reference discontinuities are reported instead of hidden.

P2's existing harness was absent from the initially empty repository. `pathsnap_p1/models.py` therefore provides a reusable group-aware harness, not a claim of integration with an unseen external harness. P2 can supply additional estimators through its `train(config, estimators=...)` interface. P3 implements HMM/Viterbi; P1 does not claim decoder results.

## Optional dependency status

If Folium is missing, preparation still creates static geographical figures and explicitly records interactive maps as pending. Install the requirements when package access is available, then run `maps`; do not claim static figures are Folium maps. External Leaflet scripts and map tiles require internet when opening Folium HTML.

## Repository submission

Source code, small result tables, reports and plots can be committed. Raw data, the local virtual environment and large derived tables are excluded. Model artifacts are generated locally; inspect their sizes and use your team's agreed Git LFS/release strategy if needed for submission. Never publish evaluator reference labels or private credentials. Do not claim only these two models fulfil the team's minimum-five requirement.

AI assistance was used to implement and document this workflow. Verify the actual results and explain the choices in your own words; follow the event's disclosure policy.
