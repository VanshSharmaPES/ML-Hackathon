"""Load the exported model and configuration once, at API startup.

Expected files in the artifacts folder (see docs/interfaces.md):
    model.joblib      fitted scikit-learn Pipeline with predict_proba (P2)
    features.json     {"features": [...], "model_name": str, "trained_on": str, "seed": int} (P2)
    hmm_config.json   candidate + emission/transition/gap parameters (P1/P3)

If none of them exist yet, the API runs the nearest-arc baseline.
If only some exist, startup fails, so a half-exported model is never served.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

REQUIRED_FILES = ("model.joblib", "features.json", "hmm_config.json")


@dataclass
class Artifacts:
    model: Any = None
    features: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    hmm_config: dict = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        return self.meta.get("model_name", type(self.model).__name__ if self.model is not None else "none")


def load_artifacts(artifacts_dir: Path) -> Artifacts:
    present = [name for name in REQUIRED_FILES if (artifacts_dir / name).is_file()]
    if not present:
        return Artifacts()
    missing = [name for name in REQUIRED_FILES if name not in present]
    if missing:
        raise RuntimeError(f"Incomplete artifacts in {artifacts_dir}: missing {', '.join(missing)}")

    meta = json.loads((artifacts_dir / "features.json").read_text())
    features = meta.get("features") or []
    if not features:
        raise RuntimeError("features.json has no 'features' list")
    model = joblib.load(artifacts_dir / "model.joblib")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"{type(model).__name__} in model.joblib has no predict_proba")
    return Artifacts(
        model=model,
        features=features,
        meta=meta,
        hmm_config=json.loads((artifacts_dir / "hmm_config.json").read_text()),
    )
