"""Load the exported model and decoder configuration once, at API startup.

Expected files in the artifacts folder (see docs/interfaces.md):
    model.joblib     bundle saved by pathsnap_p1.models.train (P1/P2):
                     {"pipeline": fitted sklearn Pipeline, "features": [...],
                      "configuration": {...pipeline settings used in training...}, ...}
    hmm_config.json  emission / transition / gap parameters for hmm.viterbi.decode (P3)

Neither present -> the API serves the nearest-arc baseline.
Only one present -> startup fails, so a half-exported model is never served.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib

REQUIRED_FILES = ("model.joblib", "hmm_config.json")
BUNDLE_KEYS = ("pipeline", "features", "configuration")


@dataclass
class Artifacts:
    bundle: dict | None = None
    hmm_config: dict = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def pipeline(self):
        return self.bundle["pipeline"]

    @property
    def features(self) -> list[str]:
        return list(self.bundle["features"])

    @property
    def feature_config(self) -> dict:
        """Pipeline settings the model was trained with (candidates, gaps, projection)."""
        return self.bundle["configuration"]

    @property
    def model_name(self) -> str:
        if not self.loaded:
            return "none"
        if self.bundle.get("model_name"):
            return self.bundle["model_name"]
        steps = getattr(self.pipeline, "steps", None)
        return type(steps[-1][1] if steps else self.pipeline).__name__


def load_artifacts(artifacts_dir: Path) -> Artifacts:
    present = [name for name in REQUIRED_FILES if (artifacts_dir / name).is_file()]
    if not present:
        return Artifacts()
    missing = [name for name in REQUIRED_FILES if name not in present]
    if missing:
        raise RuntimeError(f"Incomplete artifacts in {artifacts_dir}: missing {', '.join(missing)}")

    bundle = joblib.load(artifacts_dir / "model.joblib")
    if not isinstance(bundle, dict) or any(key not in bundle for key in BUNDLE_KEYS):
        raise RuntimeError(f"model.joblib must be a dict with keys {BUNDLE_KEYS} "
                           "(the format saved by pathsnap_p1.models.train)")
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict_proba") or 1 not in list(getattr(pipeline, "classes_", [])):
        raise RuntimeError("The model pipeline needs predict_proba and a positive class 1")
    if not bundle["features"]:
        raise RuntimeError("The model bundle has an empty feature list")
    return Artifacts(bundle=bundle, hmm_config=json.loads((artifacts_dir / "hmm_config.json").read_text()))
