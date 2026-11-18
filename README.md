# PathSnap: Map-Matching & Route Reconstruction

Matches noisy GPS trajectories to a road network. For each GPS point we generate
candidate road segments, score them with a trained classifier, and decode a
connected route with an HMM + Viterbi. A FastAPI server (Laptop 1) runs the
pipeline; an evaluator UI (Laptop 2) selects a trip and shows the predicted route.

UE24CS352A Machine Learning, PES University RR Campus.

## Repository layout

| Path | Contents | Owner |
|---|---|---|
| `pathsnap_p1/` | Loading and cleaning, per-trip UTM projection, reference alignment (labels), candidate generation and features, grouped-CV harness, KNN and Gaussian NB, plots and Folium maps | P1 |
| `tests/` | Unit tests for `pathsnap_p1` (`pytest`) | P1 |
| `notebooks/` | Exploration notebook (reads generated results) | P1 |
| `models/` | Additional models, trained through `pathsnap_p1.models.train` | P2 |
| `optimisation/` | Bayesian hyperparameter optimisation (Optuna) | P2 |
| `hmm/` | HMM emission/transition models and Viterbi decoding (`hmm/viterbi.py`) | P3 |
| `inference/` | `Predictor`: the end-to-end pipeline used by the API | P4 |
| `api/` | FastAPI backend (Laptop 1) | P4 |
| `ui/` | Evaluator interface (Laptop 2) | P4 |
| `visualisations/` | Extra plotting scripts (P1's Folium code lives in `pathsnap_p1/visuals.py`) | P1, P3 |
| `outputs/` | Committed results: CSV tables, figures, Folium maps, EDA and model reports. `processed/`, `models/`, `predictions/` are regenerated locally and gitignored | All |
| `artifacts/` | Final model bundle and HMM configuration served by the API | P2, P3 |
| `docs/` | Interface definitions and report assets | All |
| `config.json` | Shared pipeline settings (data location, gaps, candidates, CV, seeds) | P1 |
| `HANDOVER.md` | P1's feature/label contract for P2, P3 and the backend | P1 |
| `STEP_BY_STEP_GUIDE.md` | Walkthrough of the P1 implementation and its findings | P1 |
| `verify_*.py`, `review_anomalies.py`, `build_notebook.py`, `validate_maps.mjs` | P1 audit and helper scripts (run from the repo root) | P1 |

## Setup

Python **3.13 or 3.14**. All versions in `requirements.txt` are pinned and install on Linux and
Windows; everyone must use them, because a model saved with one scikit-learn version is not
guaranteed to load in another.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q              # P1 unit tests
```

## Data

The raw dataset is not committed. Place the trip folders in the repository root like this:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
└── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

`config.json` reads `data/` by default; pass `--data-root <folder>` to the P1 commands (or set
`DATA_DIR` for the API) if your copy lives elsewhere.

Trips come from many countries (24 UTM zones, both hemispheres), so every trip is projected
into the UTM zone of its own GPS track (`pathsnap_p1/geometry.py`).

## Reproducing the results

1. **Exploration, cleaning, labels, candidates and features (P1)**
   ```bash
   python -m pathsnap_p1 prepare
   ```
   Writes `outputs/dataset_inventory.csv`, `outputs/EDA_REPORT.md`, figures, maps and the
   per-trip candidate tables under `outputs/processed/`.
2. **KNN and Gaussian NB with trip-grouped CV (P1)**
   ```bash
   python -m pathsnap_p1 train
   ```
   Writes `outputs/model_comparison.csv`, `outputs/cv_results.csv`, `outputs/MODEL_REPORT.md`.
3. **Feature selection evidence (P1)**: correlation, single-feature AUC, permutation importance and
   drop-one ablation on the training trips (about 9 minutes; prepares into a temporary folder so the
   committed `outputs/` files are not overwritten):
   ```bash
   python -m pathsnap_p1 prepare --output /tmp/pathsnap-prepared
   python -m pathsnap_p1.feature_selection --prepared /tmp/pathsnap-prepared --output outputs/feature_selection
   ```
   Writes `outputs/feature_selection/FEATURE_SELECTION.md` and its figures.
4. **Interactive maps (P1)**: `python -m pathsnap_p1 maps`
5. **Further models (P2)**: pass estimators to `pathsnap_p1.models.train` (see `HANDOVER.md`).
6. **Bayesian optimisation (P2)**: _TODO_
7. **HMM / Viterbi decoding and comparison (P3)**: _TODO_

Quick isolated check: `python -m pathsnap_p1 all --limit 2 --output outputs/smoke --folds 2`.

## Running the system

### API (Laptop 1)

From the repo root, with the venv active:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

- Serves the trips in `EVAL_DIR` (default `data/student_test`). For evaluation:
  `EVAL_DIR=/path/to/eval_trips uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log`
- Reference routes for server-side scoring are read from `REFERENCE_DIR` (default: the same trip folders).
  If they are supplied separately: `REFERENCE_DIR=/path/to/routes`. Trips without a reference route are
  still predicted; their accuracy metrics show as unavailable.
- Every request is logged in this terminal.
- Uses the model in `artifacts/` (`model.joblib`, `hmm_config.json`) when present; otherwise serves the nearest-arc baseline. `/health` shows which.
- Interactive docs: `http://<laptop1-ip>:8000/docs`
- Fedora firewall: `sudo firewall-cmd --add-port=8000/tcp`

Quick check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"trip_id": "00000040"}'
```

End-to-end check of every trip (run it on Laptop 2 for the two-laptop test):

```bash
python -m api.e2e_check --api http://<laptop1-ip>:8000 --out outputs/e2e_student_test.csv
```

Predict one trip without the API (debugging): `python -m inference.cli data/student_test/00000040`

### UI (Laptop 2)

From the repo root, with the venv active:

```bash
API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

- Open `http://localhost:8501` on Laptop 2.
- The API address can also be changed in the sidebar.
- The UI shows raw GPS points, the predicted route and the server-computed metrics. It never receives the reference route.

## AI-use disclosure

AI assistance was used to implement and document parts of this project (including the P1
pipeline and the API). Every member must verify the results and be able to explain the choices.
_TODO: final wording to match the report's Section 8._
