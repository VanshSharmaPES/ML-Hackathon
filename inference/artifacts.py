"""Load the exported model and decoder configuration once, at API startup.

Files in the artifacts folder (see docs/interfaces.md):
    model.joblib     bundle saved by pathsnap_p1.models.train or optimisation/tuned_vs_untuned.py:
                     {"pipeline": fitted sklearn Pipeline, "features": [...],
                      "configuration": {...pipeline settings used in training...}, ...}
    hmm_config.json  emission / transition / gap parameters for hmm.viterbi.decode (P3)

    neither                         -> nearest-arc baseline
    model.joblib only               -> classifier only (highest-scoring candidate per point)
    model.joblib + hmm_config.json  -> classifier + HMM/Viterbi (needs hmm/viterbi.py)
    hmm_config.json only            -> startup fails (decoder settings without a model)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

MODEL_FILE = "model.joblib"
HMM_FILE = "hmm_config.json"
BUNDLE_KEYS = ("pipeline", "features", "configuration")


@dataclass
class Artifacts:
    bundle: dict | None = None
    hmm_config: dict | None = None

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @property
    def uses_hmm(self) -> bool:
        return self.loaded and self.hmm_config is not None

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
    def model_info(self) -> dict | None:
        """What the served model is and how it was chosen (from the bundle; no reference data)."""
        if not self.loaded:
            return None
        keys = ("hyperparameters", "selected_by", "selection_metrics", "tuning", "trained_on")
        return {"model_name": self.model_name, **{k: self.bundle[k] for k in keys if k in self.bundle}}

    @property
    def model_name(self) -> str:
        if not self.loaded:
            return "none"
        if self.bundle.get("model_name"):
            return self.bundle["model_name"]
        steps = getattr(self.pipeline, "steps", None)
        return type(steps[-1][1] if steps else self.pipeline).__name__


def load_artifacts(artifacts_dir: Path) -> Artifacts:
    has_model = (artifacts_dir / MODEL_FILE).is_file()
    has_hmm = (artifacts_dir / HMM_FILE).is_file()
    if not has_model:
        if has_hmm:
            raise RuntimeError(f"{HMM_FILE} is in {artifacts_dir} but {MODEL_FILE} is missing")
        return Artifacts()

    bundle = joblib.load(artifacts_dir / MODEL_FILE)
    if not isinstance(bundle, dict) or any(key not in bundle for key in BUNDLE_KEYS):
        raise RuntimeError(f"{MODEL_FILE} must be a dict with keys {BUNDLE_KEYS} "
                           "(the format saved by pathsnap_p1.models.train)")
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict_proba") or 1 not in list(getattr(pipeline, "classes_", [])):
        raise RuntimeError("The model pipeline needs predict_proba and a positive class 1")
    if not bundle["features"]:
        raise RuntimeError("The model bundle has an empty feature list")
    hmm_config = json.loads((artifacts_dir / HMM_FILE).read_text()) if has_hmm else None
    return Artifacts(bundle=bundle, hmm_config=hmm_config)
