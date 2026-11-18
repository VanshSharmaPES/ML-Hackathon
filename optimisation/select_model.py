"""Choose the served model across all five models, then export it to artifacts/model.joblib.

Inputs (all computed on the same 5 trip-grouped folds of the 80 training trips; checked):
  outputs/                  KNN, Gaussian NB                 (P1)
  outputs_p2/               Logistic Regression, HGB untuned (P2)
  outputs/random_forest/    Random Forest                    (P1)
  outputs_p2/tuning/        Logistic Regression, HGB tuned   (optimisation/tuned_vs_untuned.py)
plus the nearest-arc baseline the API falls back to, scored on the same validation folds
(it needs no training, so its "CV" score is simply its Top-1 on each fold's trips).

Selection rule (training trips only; student_test is reported, never used to choose):
  1. Primary: CV Top-1 accuracy. The served pipeline picks the top-scoring candidate per
     GPS point, so Top-1 is what it actually does.
  2. Models within TIE_BAND of the best CV Top-1 are treated as tied.
  3. Among tied models: lowest CV log loss. The HMM/Viterbi decoder will use these scores as
     emission probabilities, so probability quality decides.

    python -m optimisation.select_model --prepared <pathsnap_p1 prepare output dir>
    python -m optimisation.select_model --prepared <dir> --export   # refit the choice, write artifacts/model.joblib
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from inference.pipeline import _baseline_arcs
from optimisation.tuned_vs_untuned import MODELS as TUNABLE, OPTUNA_DIR, REPORTED_PARAMS
from pathsnap_p1.alignment import align_reference
from pathsnap_p1.data import load_trip
from pathsnap_p1.features import FEATURES
from pathsnap_p1.models import baseline_estimators, evaluate_predictions, load_prepared, make_pipeline

ROOT = Path(__file__).resolve().parents[1]
TIE_BAND = 0.005  # half a percentage point of Top-1
SOURCES = [("outputs", ["knn", "gaussian_nb"]),
           ("outputs_p2", ["logistic_regression", "hist_gradient_boosting"]),
           ("outputs/random_forest", ["random_forest"])]
LABELS = {"knn": "KNN", "gaussian_nb": "Gaussian NB", "logistic_regression": "Logistic Regression",
          "hist_gradient_boosting": "HistGradientBoosting", "random_forest": "Random Forest"}

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def random_forest(config):
    # Same settings as `python -m pathsnap_p1 train --model random_forest` (n_jobs does not change the model)
    return RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, max_features="sqrt",
                                  random_state=config["random_seed"], n_jobs=-1)


def committed_rows(splits_ref):
    rows = []
    for folder, models in SOURCES:
        d = ROOT / folder
        if json.loads((d / "cv_splits.json").read_text()) != splits_ref:
            raise RuntimeError(f"{folder} used different CV folds; results are not comparable")
        cv = pd.read_csv(d / "cv_results.csv")
        comp = pd.read_csv(d / "model_comparison.csv").set_index("model")
        for m in models:
            f = cv.loc[cv.model == m].sort_values("fold")
            c = comp.loc[m]
            file = d / "models" / f"{m}.joblib"
            rows.append({"key": m, "model": LABELS[m], "variant": "untuned", "source": folder,
                         "fold_top1": f.top1_accuracy.round(4).tolist(), "fold_macro_f1": f.macro_f1.round(4).tolist(),
                         "fold_log_loss": f.log_loss.round(4).tolist(),
                         "student_macro_f1": c.student_macro_f1, "student_top1": c.student_top1_accuracy,
                         "student_log_loss": c.student_log_loss, "student_inference_seconds": c.student_inference_seconds,
                         "file_mb": round(file.stat().st_size / 1e6, 2) if file.is_file() else np.nan})
    return rows


def tuned_rows():
    t = pd.read_csv(ROOT / "outputs_p2" / "tuning" / "tuned_vs_untuned.csv")
    rows = []
    for r in t.loc[t.variant == "tuned"].itertuples():
        rows.append({"key": f"{r.model}_tuned", "model": LABELS[r.model], "variant": "tuned",
                     "source": "outputs_p2/tuning", "fold_top1": json.loads(r.cv_fold_top1),
                     "fold_macro_f1": json.loads(r.cv_fold_macro_f1), "fold_log_loss": json.loads(r.cv_fold_log_loss),
                     "student_macro_f1": r.student_macro_f1, "student_top1": r.student_top1,
                     "student_log_loss": r.student_log_loss, "student_inference_seconds": r.student_inference_seconds,
                     "file_mb": np.nan})
    return rows


def baseline_row(config, data_root, splits_ref):
    """Nearest-arc baseline on each validation fold's trips (pooled Top-1, as the harness does)."""
    def pooled(split, trips):
        hits = total = 0
        for tid in trips:
            trip = load_trip(data_root / split / tid, split, config, with_reference=True)
            ref = align_reference(trip, config).reference_arc_id.to_numpy()
            pred = _baseline_arcs(trip)
            labelled = ref >= 0
            hits += int((pred[labelled] == ref[labelled]).sum())
            total += int(labelled.sum())
        return hits / total
    folds = [pooled("train", f["validation_trips"]) for f in splits_ref]
    student = pooled("student_test", sorted(p.name for p in (data_root / "student_test").iterdir() if p.is_dir()))
    return {"key": "nearest_arc_baseline", "model": "Nearest-arc baseline (no training)", "variant": "n/a",
            "source": "inference/pipeline.py", "fold_top1": [round(v, 4) for v in folds],
            "fold_macro_f1": None, "fold_log_loss": None, "student_macro_f1": np.nan, "student_top1": student,
            "student_log_loss": np.nan, "student_inference_seconds": np.nan, "file_mb": 0.0}


def estimator_for(key, config):
    if key in ("knn", "gaussian_nb"):
        return baseline_estimators(config)[key]
    if key == "random_forest":
        return random_forest(config)
    name = key.removesuffix("_tuned")
    spec = TUNABLE[name]
    if key.endswith("_tuned"):
        best = json.loads((OPTUNA_DIR / f"{name}_best.json").read_text(encoding="utf-8"))
        return spec["tuned"](best["best_parameters"])
    return spec["untuned"]()


def plot(table, chosen, baseline, path):
    t = table.sort_values("cv_top1")
    colours = [ORANGE if k == chosen else BLUE for k in t.key]
    names = [f"{m} ({v})" if v != "n/a" else m for m, v in zip(t.model, t.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    axes[0].scatter(t.cv_top1, names, color=colours, s=46, zorder=3)
    axes[0].axvline(baseline, color=INK_2, linewidth=1, linestyle="--", zorder=2)
    axes[0].text(baseline, len(t) - 0.4, " nearest-arc baseline", color=INK_2, fontsize=8, va="bottom")
    axes[0].set_xlabel("CV Top-1 accuracy (mean of 5 folds)")
    axes[0].set_title("Primary: CV Top-1", loc="left")
    ll = t.cv_log_loss.to_numpy()
    axes[1].scatter(ll, names, color=colours, s=46, zorder=3)
    axes[1].set_xlabel("CV log loss (lower is better)")
    axes[1].set_title("Tie-break: CV log loss", loc="left")
    for ax in axes:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Model selection on the 80 training trips (5 trip-grouped folds)", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, f"Orange: selected model. Models within {TIE_BAND} of the best CV Top-1 are tied; "
             "the tie is broken by CV log loss. The baseline has no probabilities, so no log loss.",
             color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Select the served model across all five models")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--prepared", required=True, help="Output directory of `python -m pathsnap_p1 prepare`")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="Folder with train/ and student_test/")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "model_selection"))
    parser.add_argument("--export", action="store_true", help="Refit the selected model and write artifacts/model.joblib")
    args = parser.parse_args()

    start_all = time.perf_counter()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    splits_ref = json.loads((ROOT / "outputs" / "cv_splits.json").read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = committed_rows(splits_ref) + tuned_rows()
    print("Scoring the nearest-arc baseline on the same folds ...", flush=True)
    rows.append(baseline_row(config, Path(args.data_root), splits_ref))
    table = pd.DataFrame(rows)
    table["cv_top1"] = table.fold_top1.apply(np.mean)
    table["cv_top1_sd"] = table.fold_top1.apply(lambda v: float(np.std(v)))
    table["cv_macro_f1"] = table.fold_macro_f1.apply(lambda v: np.nan if v is None else float(np.mean(v)))
    table["cv_log_loss"] = table.fold_log_loss.apply(lambda v: np.nan if v is None else float(np.mean(v)))

    models = table.loc[table.key != "nearest_arc_baseline"]
    best_top1 = models.cv_top1.max()
    tied = models.loc[models.cv_top1 >= best_top1 - TIE_BAND]
    chosen = tied.sort_values("cv_log_loss").iloc[0]
    baseline = table.loc[table.key == "nearest_arc_baseline"].iloc[0]
    table["tied_on_top1"] = table.key.isin(tied.key)
    table["selected"] = table.key == chosen.key
    table.sort_values("cv_top1", ascending=False).to_csv(out / "model_selection.csv", index=False)
    plot(table, chosen.key, baseline.cv_top1, out / "model_selection.png")
    fold_wins = int(sum(m > b for m, b in zip(chosen.fold_top1, baseline.fold_top1)))
    print(f"Selected: {chosen.model} ({chosen.variant}) | CV Top-1 {chosen.cv_top1:.4f} | "
          f"CV log loss {chosen.cv_log_loss:.4f} | tied with {len(tied) - 1} others", flush=True)

    export_note = "not exported (run with --export)"
    if args.export:
        config["output_dir"] = str(Path(args.prepared).resolve())
        frame, observations, _ = load_prepared(config)
        train = frame.loc[frame.split == "train"].reset_index(drop=True)
        test = frame.loc[frame.split == "student_test"].reset_index(drop=True)
        estimator = estimator_for(chosen.key, config)
        pipeline = make_pipeline(estimator).fit(train[FEATURES], train.label.to_numpy(int))
        check, _, _ = evaluate_predictions(test, pipeline.predict_proba(test[FEATURES])[:, 1],
                                           observations.loc[observations.split == "student_test"])
        params = {k: v for k, v in estimator.get_params().items() if k in REPORTED_PARAMS or k in
                  ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_neighbors", "weights", "var_smoothing")}
        bundle = {"pipeline": pipeline, "features": FEATURES,
                  "configuration": {k: v for k, v in config.items() if k not in ("data_root", "output_dir")},
                  "model_name": f"{chosen.model} ({chosen.variant})", "hyperparameters": params,
                  "label_definition": "Agreement with monotonic reference-route derived labels",
                  "trained_on": "all 80 training trips",
                  "selected_by": f"highest CV Top-1 (ties within {TIE_BAND} broken by lowest CV log loss), "
                                 "training trips only",
                  "selection_metrics": {"cv_top1": round(float(chosen.cv_top1), 4),
                                        "cv_log_loss": round(float(chosen.cv_log_loss), 4),
                                        "cv_macro_f1": round(float(chosen.cv_macro_f1), 4),
                                        "folds": len(splits_ref)}}
        if chosen.variant == "tuned":
            name = chosen.key.removesuffix("_tuned")
            n = int((pd.read_csv(OPTUNA_DIR / f"{name}_trials.csv").state == "COMPLETE").sum())
            bundle["tuning"] = {"method": "Optuna TPE (seed 42)", "trials": n,
                                "objective": "mean 5-fold trip-grouped CV Macro F1 (training trips)"}
        target = ROOT / "artifacts" / "model.joblib"
        joblib.dump(bundle, target, compress=3)
        export_note = (f"exported to `artifacts/model.joblib` ({target.stat().st_size / 1e6:.2f} MB); refit check: "
                       f"student Top-1 {check['top1_accuracy']:.4f} (table: {chosen.student_top1:.4f})")
        print(f"Exported {bundle['model_name']}: {export_note}", flush=True)

    f4 = lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"
    show = table.sort_values("cv_top1", ascending=False)
    lines = ["# Model selection", "",
             "All five models and the nearest-arc baseline, on the same 5 trip-grouped folds of the 80 training "
             "trips (fold assignments checked identical across `outputs/`, `outputs_p2/`, `outputs/random_forest/`).", "",
             "**Rule** (training trips only; student_test is reported but never used to choose):", "",
             "1. Primary metric: **CV Top-1**. The served pipeline picks the top-scoring candidate per GPS point, "
             "so Top-1 measures what it actually does (candidate-level Macro F1 uses a 0.5 threshold the "
             "pipeline never applies).",
             f"2. Models within **{TIE_BAND}** of the best CV Top-1 count as tied.",
             "3. Ties are broken by the **lowest CV log loss**, because the HMM/Viterbi decoder will use the "
             "scores as emission probabilities.", "",
             "| Model | Variant | CV Top-1 (SD) | CV log loss | CV Macro F1 | Student Top-1 | Student Macro F1 | Student log loss | File MB | Tied | Selected |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|"]
    for r in show.itertuples():
        lines.append(f"| {r.model} | {r.variant} | {r.cv_top1:.4f} ({r.cv_top1_sd:.4f}) | {f4(r.cv_log_loss)} | "
                     f"{f4(r.cv_macro_f1)} | {f4(r.student_top1)} | {f4(r.student_macro_f1)} | {f4(r.student_log_loss)} | "
                     f"{'n/a' if pd.isna(r.file_mb) else r.file_mb} | {'yes' if r.tied_on_top1 else ''} | "
                     f"{'**yes**' if r.selected else ''} |")
    lines += ["", f"## Selected: {chosen.model} ({chosen.variant})", "",
              f"- CV Top-1 {chosen.cv_top1:.4f}; tied (within {TIE_BAND}) with: "
              + ", ".join(f"{m} ({v})" for m, v in zip(tied.model, tied.variant) if f"{m}{v}" != f"{chosen.model}{chosen.variant}"),
              f"- Lowest CV log loss among the tied models: {chosen.cv_log_loss:.4f}",
              f"- Export: {export_note}", "",
              "## Against the nearest-arc baseline (per fold, Top-1)", "",
              "| | " + " | ".join(f"Fold {i}" for i in range(1, len(splits_ref) + 1)) + " | Mean |",
              "|---|" + "---:|" * (len(splits_ref) + 1),
              f"| {chosen.model} ({chosen.variant}) | " + " | ".join(f"{v:.4f}" for v in chosen.fold_top1) + f" | {chosen.cv_top1:.4f} |",
              "| Nearest-arc baseline | " + " | ".join(f"{v:.4f}" for v in baseline.fold_top1) + f" | {baseline.cv_top1:.4f} |",
              "", f"The selected model beats the baseline on {fold_wins} of {len(splits_ref)} folds. Student_test Top-1 "
              f"(points-weighted): model {chosen.student_top1:.4f} vs baseline {baseline.student_top1:.4f}.", "",
              "## Caveats", "",
              "- Labels come from aligning GPS points to the reference route by distance, so they partly favour "
              "distance-based rules; scores measure agreement with derived labels, not verified ground truth.",
              "- Student_test is easier than training (top-8 candidate miss rate 0.34% vs 9.1%), so student "
              "Top-1 (about 0.81) is far above CV Top-1 (about 0.70). Compare models on CV.",
              "- Random Forest was not tuned; Logistic Regression and HistGradientBoosting were (Optuna, 50 trials).",
              "- Inference times in the source files were measured on different laptops, so they are not compared here.",
              "- Most of the classifier's practical gain is route continuity, not Top-1: see "
              "`outputs/e2e_student_test.csv` vs `outputs/e2e_student_test_baseline.csv`.", "",
              "Figure: `model_selection.png`.", "", f"Runtime: {time.perf_counter() - start_all:.0f} s."]
    (out / "MODEL_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
