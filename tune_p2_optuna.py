import gc
import json
from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from pathsnap_p1.features import FEATURES
from pathsnap_p1.models import load_prepared, make_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_p2"
OPTUNA_DIR = OUT / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.json", encoding="utf-8") as f:
    config = json.load(f)

config["output_dir"] = str(OUT.resolve())

frame, observations, run = load_prepared(config)
train_df = frame.loc[frame.split == "train"].reset_index(drop=True)

X = train_df[FEATURES]
y = train_df.label.to_numpy(int)
groups = train_df.trip_id

folds = min(config["cv_folds"], groups.nunique())
splitter = GroupKFold(n_splits=folds)
splits = list(splitter.split(X, y, groups))

print("P2 OPTUNA BAYESIAN OPTIMISATION")
print("Training rows:", len(train_df))
print("Features:", len(FEATURES))
print("GroupKFold folds:", folds)
print("Trials per model: 50")
print("Student-test data is not used for tuning.")

def objective_factory(model_name):
    def objective(trial):

        if model_name == "logistic_regression":
            estimator = LogisticRegression(
                C=trial.suggest_float("C", 0.001, 100.0, log=True),
                class_weight=trial.suggest_categorical(
                    "class_weight", [None, "balanced"]
                ),
                max_iter=3000,
                solver="lbfgs",
                random_state=42
            )

        else:
            estimator = HistGradientBoostingClassifier(
                max_iter=trial.suggest_int("max_iter", 80, 200, step=20),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.03, 0.2, log=True
                ),
                max_leaf_nodes=trial.suggest_int(
                    "max_leaf_nodes", 15, 63, step=8
                ),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf", 20, 100, step=20
                ),
                l2_regularization=trial.suggest_float(
                    "l2_regularization", 1e-8, 10.0, log=True
                ),
                random_state=42
            )

        pipeline = make_pipeline(estimator)
        fold_scores = []

        for fold, (tr, va) in enumerate(splits, start=1):
            model = clone(pipeline)
            model.fit(X.iloc[tr], y[tr])

            probabilities = model.predict_proba(X.iloc[va])[:, 1]
            predictions = (probabilities > 0.5).astype(int)

            score = f1_score(
                y[va],
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0
            )
            fold_scores.append(score)

            del model, probabilities, predictions
            gc.collect()

        trial.set_user_attr("fold_scores", fold_scores)
        return sum(fold_scores) / len(fold_scores)

    return objective


def run_study(model_name):
    database = OPTUNA_DIR / "p2_optuna.db"
    storage = f"sqlite:///{database.resolve().as_posix()}"

    study = optuna.create_study(
        study_name=model_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        storage=storage,
        load_if_exists=True
    )

    completed = sum(
        t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )
    remaining = max(0, 50 - completed)

    csv_path = OPTUNA_DIR / f"{model_name}_trials.csv"
    json_path = OPTUNA_DIR / f"{model_name}_best.json"

    def save_progress(study, trial):
        study.trials_dataframe().to_csv(csv_path, index=False)

        if study.best_trial is not None:
            result = {
                "model": model_name,
                "best_macro_f1": study.best_value,
                "best_parameters": study.best_params,
                "best_fold_scores": study.best_trial.user_attrs.get(
                    "fold_scores", []
                ),
                "completed_trials": sum(
                    t.state == optuna.trial.TrialState.COMPLETE
                    for t in study.trials
                )
            }
            json_path.write_text(
                json.dumps(result, indent=4),
                encoding="utf-8"
            )

    print(f"\nStarting {model_name}")
    print(f"Already completed: {completed}; remaining: {remaining}")

    if remaining:
        study.optimize(
            objective_factory(model_name),
            n_trials=remaining,
            callbacks=[save_progress],
            gc_after_trial=True,
            show_progress_bar=True
        )

    study.trials_dataframe().to_csv(csv_path, index=False)

    result = {
        "model": model_name,
        "best_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "best_fold_scores": study.best_trial.user_attrs.get(
            "fold_scores", []
        ),
        "completed_trials": sum(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
    }

    json_path.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print(f"\n{model_name} best CV Macro F1: {study.best_value:.5f}")
    print("Best parameters:", study.best_params)


optuna.logging.set_verbosity(optuna.logging.WARNING)

run_study("logistic_regression")
run_study("hist_gradient_boosting")

print("\nBOTH OPTUNA STUDIES FINISHED.")
print("Results saved in:", OPTUNA_DIR)
