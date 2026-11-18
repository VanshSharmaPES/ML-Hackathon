"""Run the prediction pipeline on one trip folder without the API (debugging aid).

    python -m inference.cli data/student_test/00000040
    python -m inference.cli data/student_test/00000040 --json prediction.json

Uses the same Predictor as the API (model from artifacts/ if present, else the
nearest-arc baseline). Never reads the .route file and computes no accuracy.
"""
import argparse
import json
from pathlib import Path

from .artifacts import load_artifacts
from .pipeline import NO_ARC, Predictor

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one trip without the API")
    parser.add_argument("trip_dir", type=Path, help="Folder containing <trip_id>.track/.nodes/.arcs")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.json")
    parser.add_argument("--json", type=Path, help="Also write the full prediction to this JSON file")
    args = parser.parse_args()

    predictor = Predictor(load_artifacts(args.artifacts), json.loads(args.config.read_text(encoding="utf-8")))
    prediction, _ = predictor.predict(args.trip_dir)

    per_obs = prediction["predicted_arc_per_obs"]
    timings = ", ".join(f"{stage} {ms:.0f} ms" for stage, ms in prediction["timings"].items())
    print(f"trip        {prediction['trip_id']}")
    print(f"method      {prediction['method']}")
    print(f"points      {len(per_obs):,} ({per_obs.count(NO_ARC)} without an arc)")
    print(f"route       {len(prediction['predicted_arcs']):,} arcs, first: {prediction['predicted_arcs'][:10]}")
    print(f"timings     {timings}")
    if args.json:
        args.json.write_text(json.dumps(prediction), encoding="utf-8")
        print(f"wrote       {args.json}")


if __name__ == "__main__":
    main()
