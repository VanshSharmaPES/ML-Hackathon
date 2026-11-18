import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap_p1.models import train

ROOT = Path(__file__).resolve().parent

with open(ROOT / "config.json", encoding="utf-8") as f:
    config = json.load(f)

config["data_root"] = str(
    (ROOT.parent / "Student folder" / "Student folder").resolve()
)
config["output_dir"] = str((ROOT / "outputs_p2").resolve())

estimators = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
}

print("P2 BASELINE TRAINING")
print("Models:", list(estimators.keys()))
print("CV folds:", config["cv_folds"])
print("Output:", config["output_dir"])

results = train(config, estimators=estimators)

print("\nP2 BASELINE RESULTS")
print(results.to_string(index=False))
