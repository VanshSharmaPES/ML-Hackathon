"""API settings, read from environment variables.

DATA_DIR       folder holding train/ and student_test/   (default: <repo>/data)
EVAL_DIR       folder holding the trips the API serves   (default: DATA_DIR/student_test)
ARTIFACTS_DIR  exported model + configs                  (default: <repo>/artifacts)

On evaluation day, point EVAL_DIR at the evaluator trips. No code change needed.
"""
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")).resolve()
EVAL_DIR = Path(os.environ.get("EVAL_DIR", DATA_DIR / "student_test")).resolve()
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", REPO_ROOT / "artifacts")).resolve()

# Trip folders are 8-digit names, e.g. 00000040
TRIP_ID_PATTERN = re.compile(r"^\d{8}$")
