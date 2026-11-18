"""API settings, read from environment variables.

DATA_DIR         folder holding train/ and student_test/   (default: <repo>/data)
EVAL_DIR         folder holding the trips the API serves   (default: DATA_DIR/student_test)
REFERENCE_DIR    folder holding <trip_id>/<trip_id>.route  (default: EVAL_DIR, next to the tracks)
ARTIFACTS_DIR    exported model + decoder config           (default: <repo>/artifacts)
PIPELINE_CONFIG  P1 pipeline settings for the baseline     (default: <repo>/config.json)

On evaluation day, point EVAL_DIR at the evaluator trips (and REFERENCE_DIR at their
reference routes if those are supplied separately). No code change needed.
"""
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).resolve()
EVAL_DIR = Path(os.environ.get("EVAL_DIR", DATA_DIR / "student_test")).resolve()
REFERENCE_DIR = Path(os.environ.get("REFERENCE_DIR", EVAL_DIR)).resolve()
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", REPO_ROOT / "artifacts")).resolve()
PIPELINE_CONFIG = Path(os.environ.get("PIPELINE_CONFIG", REPO_ROOT / "config.json")).resolve()

# Trip folders are 8-digit names, e.g. 00000040
TRIP_ID_PATTERN = re.compile(r"^\d{8}$")


def load_pipeline_config() -> dict:
    return json.loads(PIPELINE_CONFIG.read_text(encoding="utf-8"))
