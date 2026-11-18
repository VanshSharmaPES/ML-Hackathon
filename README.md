# PathSnap: Map-Matching & Route Reconstruction

Matches noisy GPS trajectories to a road network. For each GPS point we generate
candidate road segments, score them with a trained classifier, and decode a
connected route with an HMM + Viterbi. A FastAPI server (Laptop 1) runs the
pipeline; an evaluator UI (Laptop 2) selects a trip and shows the predicted route.

UE24CS352A Machine Learning, PES University RR Campus.

## Repository layout

| Folder | Contents | Owner |
|---|---|---|
| `data_processing/` | Loading, cleaning, projection, label alignment | P1 |
| `features/` | Candidate generation and feature extraction | P1 |
| `models/` | Cross-validation harness and model training scripts | P2, P1 |
| `optimisation/` | Bayesian hyperparameter optimisation (Optuna) | P2 |
| `hmm/` | HMM emission/transition models and Viterbi decoding | P3 |
| `inference/` | End-to-end `predict_trip()` used by the API | P4 |
| `api/` | FastAPI backend | P4 |
| `ui/` | Evaluator interface | P4 |
| `visualisations/` | Folium maps and plotting scripts | P1, P3 |
| `results/` | Experiment logs, metrics CSVs, plots | All |
| `artifacts/` | Final model weights and configuration | P2, P3 |
| `docs/` | Interface definitions and report assets | All |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The raw dataset is not committed. Place the trip folders like this:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
└── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

Set `DATA_DIR` if the data lives elsewhere (default: `./data`).

Trips come from many countries (24 UTM zones, both hemispheres), so every trip
is projected into the UTM zone of its own GPS track (`inference/geo.py`).

## Reproducing the results

_TODO: exact commands, filled in as each stage is finished._

1. Exploration and preprocessing
2. Candidate generation and features
3. Model training and comparison
4. Bayesian optimisation
5. HMM / Viterbi decoding and comparison
6. Folium maps

## Running the system

### API (Laptop 1)

From the repo root, with the venv active:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

- Serves the trips in `EVAL_DIR` (default `data/student_test`). For evaluation:
  `EVAL_DIR=/path/to/eval_trips uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log`
- Every request is logged in this terminal.
- Interactive docs: `http://<laptop1-ip>:8000/docs`
- Fedora firewall: `sudo firewall-cmd --add-port=8000/tcp`

Quick check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"trip_id": "00000040"}'
```

### UI (Laptop 2)

_TODO_

## AI-use disclosure

_TODO: matches the report's Section 8._
