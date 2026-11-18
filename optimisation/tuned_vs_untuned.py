"""Tuned vs untuned comparison for P2's Optuna studies.

Reads the best hyperparameters found by tune_p2_optuna.py (outputs_p2/optuna/*_best.json)
and evaluates the untuned and the tuned configuration of each model with exactly the
same data, folds and metrics:

  * 5-fold GroupKFold by trip on the 80 training trips (the same folds as
    pathsnap_p1.models.train and the Optuna study);
  * a refit on all 80 training trips, evaluated once on the 10 student_test trips.

Per-fold Macro F1, Top-1 and log loss are kept, so the report shows where any tuning
gain comes from. Choosing and exporting the final model is done afterwards by
optimisation/select_model.py, across all five models. Evaluator trips are never read.

    python -m optimisation.tuned_vs_untuned --prepared <pathsnap_p1 prepare output dir>
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pathsnap_p1.features import FEATURES
from pathsnap_p1.models import evaluate_predictions, load_prepared, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
OPTUNA_DIR = ROOT / "outputs_p2" / "optuna"
FAR_SHARE = 0.5  # a training trip with more than this share of points >50 m from its route is flagged

# Untuned = P2's baseline settings (train_p2_baselines.py). Tuned = Optuna best values plus the
# same fixed settings the Optuna objective used (tune_p2_optuna.py).
MODELS = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "untuned": lambda: LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
        "tuned": lambda p: LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=3000,
                                              solver="lbfgs", random_state=42),
    },
    "hist_gradient_boosting": {
        "label": "HistGradientBoosting",
        "untuned": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=31,
                                                          random_state=42),
        "tuned": lambda p: HistGradientBoostingClassifier(random_state=42, **p),
    },
}
REPORTED_PARAMS = ("C", "class_weight", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                   "l2_regularization")

# Chart tokens (light surface); categorical slot 1 = untuned, slot 2 = tuned
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
UNTUNED, TUNED = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def evaluate(estimator, train, train_obs, test, test_obs, splits):
    """Grouped CV on training trips, then one refit evaluated on student_test."""
    pipeline = make_pipeline(estimator)
    folds = []
    for tr, va in splits:
        model = clone(pipeline).fit(train.iloc[tr][FEATURES], train.label.iloc[tr].to_numpy(int))
        val = train.iloc[va]
        m, _, _ = evaluate_predictions(val, model.predict_proba(val[FEATURES])[:, 1],
                                       train_obs.loc[train_obs.trip_id.isin(val.trip_id.unique())])
        folds.append(m)
    cv = pd.DataFrame(folds)

    final = clone(pipeline)
    start = time.perf_counter()
    final.fit(train[FEATURES], train.label.to_numpy(int))
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    proba = final.predict_proba(test[FEATURES])[:, 1]
    infer_s = time.perf_counter() - start
    st, _, _ = evaluate_predictions(test, proba, test_obs)
    return {
        "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_sd": cv.macro_f1.std(ddof=0),
        "cv_top1": cv.top1_accuracy.mean(), "cv_top1_sd": cv.top1_accuracy.std(ddof=0),
        "cv_log_loss": cv.log_loss.mean(),
        "student_macro_f1": st["macro_f1"], "student_precision": st["macro_precision"],
        "student_recall": st["macro_recall"], "student_top1": st["top1_accuracy"],
        "student_log_loss": st["log_loss"], "fit_seconds": fit_s, "student_inference_seconds": infer_s,
        "student_rows": len(test),
        "cv_fold_macro_f1": json.dumps([round(v, 4) for v in cv.macro_f1]),
        "cv_fold_top1": json.dumps([round(v, 4) for v in cv.top1_accuracy]),
        "cv_fold_log_loss": json.dumps([round(v, 4) for v in cv.log_loss]),
    }


def tuning_cost(name):
    trials = pd.read_csv(OPTUNA_DIR / f"{name}_trials.csv")
    duration = pd.to_timedelta(trials.duration)
    done = trials.state == "COMPLETE"
    return trials, {"trials_complete": int(done.sum()), "trials_failed": int((trials.state == "FAIL").sum()),
                    "search_minutes": round(duration.sum().total_seconds() / 60, 1),
                    "seconds_per_trial": round(duration[done].mean().total_seconds(), 1)}


def plot_history(histories, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(10, 3.8))
    for ax, (label, trials) in zip(np.atleast_1d(axes), histories):
        done = trials.loc[trials.state == "COMPLETE"].sort_values("number")
        ax.scatter(done.number, done.value, s=14, color=UNTUNED, alpha=0.75, label="Trial (CV Macro F1)", zorder=2)
        ax.plot(done.number, done.value.cummax(), color=TUNED, linewidth=2, label="Best so far", zorder=3)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Trial")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("5-fold CV Macro F1 (training trips)")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("Optuna TPE search history", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, "Training trips only, trip-grouped folds. Seed 42.", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_comparison(results, path):
    metrics = [("cv_macro_f1", "CV Macro F1"), ("student_macro_f1", "Student Macro F1"),
               ("cv_top1", "CV Top-1"), ("student_top1", "Student Top-1")]
    models = list(dict.fromkeys(results.label))
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.8), sharey=True)
    x = np.arange(len(metrics))
    for ax, label in zip(np.atleast_1d(axes), models):
        sub = results.loc[results.label == label].set_index("variant")
        for offset, variant, colour in [(-0.19, "untuned", UNTUNED), (0.19, "tuned", TUNED)]:
            ax.bar(x + offset, [sub.loc[variant, m] for m, _ in metrics], width=0.36, color=colour,
                   label=variant.capitalize(), edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(x, [n for _, n in metrics], fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(label, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.suptitle("Untuned vs tuned (same folds, same student_test trips)", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, "Axis starts at 0.5. Log loss and inference time are in the table (different units).",
             color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def flagged_trips(train, train_obs, splits):
    """Training trips whose GPS is mostly far (>50 m) from their own reference route, by fold."""
    far = train_obs.groupby("trip_id").alignment_far.mean()
    flagged = far[far > FAR_SHARE]
    out = {}
    for fold, (_, va) in enumerate(splits, 1):
        trips = set(train.iloc[va].trip_id.unique())
        out[fold] = {t: round(float(flagged[t]), 3) for t in sorted(trips & set(flagged.index))}
    return out


def main():
    parser = argparse.ArgumentParser(description="Tuned vs untuned comparison (P2 Optuna studies)")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--prepared", required=True, help="Output directory of `python -m pathsnap_p1 prepare`")
    parser.add_argument("--output", default=str(ROOT / "outputs_p2" / "tuning"))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["output_dir"] = str(Path(args.prepared).resolve())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()

    frame, observations, _ = load_prepared(config)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    test = frame.loc[frame.split == "student_test"].reset_index(drop=True)
    train_obs = observations.loc[observations.split == "train"]
    test_obs = observations.loc[observations.split == "student_test"]
    folds = min(config["cv_folds"], train.trip_id.nunique())
    splits = list(GroupKFold(n_splits=folds).split(train, train.label, groups=train.trip_id))
    print(f"Training rows {len(train):,} ({train.trip_id.nunique()} trips), student_test rows {len(test):,}", flush=True)

    rows, histories, costs = [], [], {}
    for name, spec in MODELS.items():
        best = json.loads((OPTUNA_DIR / f"{name}_best.json").read_text(encoding="utf-8"))
        trials, costs[name] = tuning_cost(name)
        histories.append((spec["label"], trials))
        for variant, estimator in [("untuned", spec["untuned"]()), ("tuned", spec["tuned"](best["best_parameters"]))]:
            print(f"  {spec['label']} ({variant}) ...", flush=True)
            row = evaluate(estimator, train, train_obs, test, test_obs, splits)
            params = {k: v for k, v in estimator.get_params().items() if k in REPORTED_PARAMS}
            rows.append({"model": name, "label": spec["label"], "variant": variant, **row,
                         "hyperparameters": json.dumps(params, default=str)})
            print(f"    CV Macro F1 {row['cv_macro_f1']:.4f} | CV Top-1 {row['cv_top1']:.4f} | "
                  f"student Macro F1 {row['student_macro_f1']:.4f} | student Top-1 {row['student_top1']:.4f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / "tuned_vs_untuned.csv", index=False)
    pd.DataFrame([{"model": k, **v} for k, v in costs.items()]).to_csv(out / "tuning_cost.csv", index=False)
    plot_history(histories, out / "optuna_history.png")
    plot_comparison(results, out / "tuned_vs_untuned.png")
    flags = flagged_trips(train, train_obs, splits)
    (out / "selection.json").unlink(missing_ok=True)  # selection now lives in outputs/model_selection/

    # Report
    f4 = lambda v: f"{v:.4f}"
    lines = ["# Bayesian hyperparameter optimisation: tuned vs untuned", "",
             "Search: Optuna TPE sampler (seed 42), 50 trials per model, objective = mean 5-fold trip-grouped CV "
             "Macro F1 on the 80 training trips (`tune_p2_optuna.py`). Student-test and evaluator trips were not "
             "used in the search.", "",
             "Comparison: untuned and tuned settings evaluated with the same folds, the same refit on all "
             "training trips and the same 10 student_test trips (`optimisation/tuned_vs_untuned.py`).", "",
             "| Model | Variant | CV Macro F1 (SD) | CV Top-1 (SD) | CV log loss | Student Macro F1 | Student Top-1 | Student log loss | Inference s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results.itertuples():
        lines.append(f"| {r.label} | {r.variant} | {r.cv_macro_f1:.4f} ({r.cv_macro_f1_sd:.4f}) | "
                     f"{r.cv_top1:.4f} ({r.cv_top1_sd:.4f}) | {f4(r.cv_log_loss)} | {f4(r.student_macro_f1)} | "
                     f"{f4(r.student_top1)} | {f4(r.student_log_loss)} | {r.student_inference_seconds:.2f} |")
    lines += ["", f"Inference time is `predict_proba` on {int(results.student_rows.iloc[0]):,} student_test candidate rows.",
              "", "## Where the tuning gain comes from (per fold)", "",
              "| Model | Metric | " + " | ".join(f"Fold {i}" for i in range(1, folds + 1)) + " |",
              "|---|---|" + "---:|" * folds]
    for name, spec in MODELS.items():
        u = results.loc[(results.model == name) & (results.variant == "untuned")].iloc[0]
        t = results.loc[(results.model == name) & (results.variant == "tuned")].iloc[0]
        for metric, col in [("Macro F1 gain", "cv_fold_macro_f1"), ("Top-1 gain", "cv_fold_top1")]:
            gains = [b - a for a, b in zip(json.loads(u[col]), json.loads(t[col]))]
            lines.append(f"| {spec['label']} | {metric} | " + " | ".join(f"{g:+.4f}" for g in gains) + " |")
    lines += ["", "Training trips with more than half of their GPS points over 50 m from their own reference "
              "route (data anomalies documented in `outputs/ALIGNMENT_ANOMALIES.md`), by validation fold:", ""]
    for fold, trips in flags.items():
        lines.append(f"- Fold {fold}: " + (", ".join(f"{t} ({share:.0%} far)" for t, share in trips.items()) or "none"))
    lines += ["", "If the gain is concentrated in the fold that holds these trips, the tuned settings mainly make "
              "the model more robust to off-network GPS rather than better everywhere. Report it that way.", "",
              "## Search cost", "", "| Model | Complete trials | Failed | Search time (min) | Seconds per trial |",
              "|---|---:|---:|---:|---:|"]
    for name, c in costs.items():
        lines.append(f"| {MODELS[name]['label']} | {c['trials_complete']} | {c['trials_failed']} | "
                     f"{c['search_minutes']} | {c['seconds_per_trial']} |")
    lines += ["", "## Hyperparameters", ""]
    for r in results.itertuples():
        lines.append(f"- {r.label}, {r.variant}: `{r.hyperparameters}`")
    lines += ["", "## Notes", "",
              "- The Optuna best value is the best of 50 scores on the same folds it was searched on, so it is "
              "optimistic. The student_test columns are the independent check of whether tuning helped.",
              "- The objective was candidate-level Macro F1 at a 0.5 threshold. The served pipeline picks the "
              "top-scoring candidate per point, which is Top-1; both are reported above.",
              "- Logistic Regression: the tuned C gives the same predictions as the default (1.6M scaled rows, so "
              "regularisation strength barely matters). The search confirms the default; it is not a tuning gain.",
              "- The final model is chosen across all five models in `outputs/model_selection/MODEL_SELECTION.md`.",
              "", "Figures: `optuna_history.png`, `tuned_vs_untuned.png`.", "",
              f"Runtime of this comparison: {time.perf_counter() - start_all:.0f} s."]
    (out / "TUNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
