"""Feature-selection evidence for the P1 candidate features.

Answers four questions about the 11 features in pathsnap_p1.features.FEATURES:
  1. Redundancy  - are any features near-duplicates?        (Spearman correlation)
  2. Signal      - does each feature separate correct roads? (single-feature ROC AUC)
  3. Use         - does the model rely on it?                (permutation importance)
  4. Necessity   - does the model get worse without it?      (drop-one ablation)

Only the 80 training trips are used, with the same trip-grouped GroupKFold folds as
pathsnap_p1.models.train. The student_test trips are not touched, so no selection
decision can leak test information. Importance is model-specific, so the analysis can be run
for Gaussian NB (fast, the original check) or for the tuned HistGradientBoosting (the model
the pipeline serves): `--model gaussian_nb | hist_gradient_boosting_tuned`.

Run after `python -m pathsnap_p1 prepare`:
    python -m pathsnap_p1.feature_selection
    python -m pathsnap_p1.feature_selection --prepared <prepare output dir> --output <dir>
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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from pathsnap_p1.features import FEATURES
from pathsnap_p1.models import evaluate_predictions, load_prepared, make_pipeline

METRICS = ("macro_f1", "top1_accuracy", "log_loss")


def _tuned_hgb(config):
    best = json.loads((Path(__file__).resolve().parents[1] / "outputs_p2" / "optuna" /
                       "hist_gradient_boosting_best.json").read_text(encoding="utf-8"))
    return HistGradientBoostingClassifier(random_state=42, **best["best_parameters"])


MODEL_CHOICES = {
    "gaussian_nb": ("Gaussian NB", lambda config: GaussianNB(var_smoothing=config["gaussian_nb_var_smoothing"])),
    "hist_gradient_boosting_tuned": ("HistGradientBoosting (tuned)", _tuned_hgb),
}
CORRELATION_LIMIT = 0.9      # |rho| at or above this flags a redundant pair
CORRELATION_SAMPLE = 300_000  # candidate rows sampled for correlation / AUC

# Chart tokens (light surface)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, RED = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#c43d3c"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.grid": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def fold_metrics(pipeline, fit_rows, val_rows, val_obs, features, X_val=None):
    model = clone(pipeline).fit(fit_rows[features], fit_rows.label.to_numpy(int))
    proba = model.predict_proba(val_rows[features] if X_val is None else X_val)[:, 1]
    metrics, _, _ = evaluate_predictions(val_rows, proba, val_obs)
    return model, {k: metrics[k] for k in METRICS}


def evaluate(model, val_rows, val_obs, X_val):
    proba = model.predict_proba(X_val)[:, 1]
    metrics, _, _ = evaluate_predictions(val_rows, proba, val_obs)
    return {k: metrics[k] for k in METRICS}


def correlation(train, seed):
    sample = train.sample(n=min(CORRELATION_SAMPLE, len(train)), random_state=seed)
    constant = [f for f in FEATURES if sample[f].nunique(dropna=True) <= 1]
    varying = [f for f in FEATURES if f not in constant]
    rho = sample[varying].corr(method="spearman")
    pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(varying) for b in varying[i + 1:]
             if abs(rho.loc[a, b]) >= CORRELATION_LIMIT]
    return sample, rho, constant, pairs


def univariate(sample):
    rows = []
    y = sample.label.to_numpy(int)
    for f in FEATURES:
        x = sample[f].to_numpy(float)
        ok = np.isfinite(x)
        auc = roc_auc_score(y[ok], x[ok]) if ok.any() and np.unique(x[ok]).size > 1 and np.unique(y[ok]).size == 2 else np.nan
        rows.append({
            "feature": f, "missing_rate": float(1 - ok.mean()),
            "distinct_values": int(np.unique(x[ok]).size) if ok.any() else 0,
            "roc_auc": auc,
            "separability": np.nan if np.isnan(auc) else max(auc, 1 - auc),
            "direction": "" if np.isnan(auc) else ("higher = more likely correct" if auc >= 0.5 else "lower = more likely correct"),
        })
    return pd.DataFrame(rows).sort_values("separability", ascending=False, na_position="last")


def plot_correlation(rho, constant, path):
    n = len(rho)
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    im = ax.imshow(rho.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), rho.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), rho.index)
    for i in range(n):
        for j in range(n):
            v = rho.iat[i, j]
            if pd.isna(v):  # one feature is always missing when the other is set
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=INK_2)
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.65 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Spearman ρ")
    cbar.outline.set_visible(False)
    note = ("Training trips only. – : undefined (one feature is always missing when the other is set).")
    if constant:
        note += f"  Constant, excluded: {', '.join(constant)}."
    ax.set_title("Feature correlation (candidate rows)", loc="left")
    fig.text(0.01, 0.005, note, color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_bars(summary, value_col, sd_col, title, xlabel, path, diverging=False):
    data = summary.sort_values(value_col)
    colours = [BLUE if not diverging or v <= 0 else RED for v in data[value_col]] if diverging else BLUE
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(data.feature, data[value_col], xerr=data[sd_col], color=colours, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    note = "Training trips only, 5 trip-grouped folds. Bars: mean, whiskers: ±1 SD across folds."
    if diverging:
        note += "\nBlue: removing the feature lowers CV Macro F1 (it helps). Red: removing it raises CV Macro F1."
    fig.text(0.01, 0.005, note, color=INK_2, fontsize=7.5, va="bottom")
    fig.tight_layout(rect=(0, 0.08 if diverging else 0.05, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def table(frame, columns, fmt):
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" if c == columns[0] else "---:" for c in columns) + "|"
    rows = ["| " + " | ".join(fmt.get(c, str)(r[c]) for c in columns) + " |" for _, r in frame.iterrows()]
    return "\n".join([head, rule, *rows])


def main():
    parser = argparse.ArgumentParser(description="Feature-selection evidence (training trips only)")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--prepared", help="Output directory of `pathsnap_p1 prepare` (default: config output_dir)")
    parser.add_argument("--output", help="Where to write results (default: <prepared>/feature_selection)")
    parser.add_argument("--repeats", type=int, default=3, help="Permutation repeats per feature per fold")
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="gaussian_nb",
                        help="Model whose feature use is measured")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared = Path(args.prepared).resolve() if args.prepared else (config_path.parent / config["output_dir"]).resolve()
    config["output_dir"] = str(prepared)
    out = Path(args.output).resolve() if args.output else prepared / "feature_selection"
    out.mkdir(parents=True, exist_ok=True)
    seed = config["random_seed"]
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    frame, observations, _ = load_prepared(config)
    train = frame.loc[frame.split == "train"].reset_index(drop=True)
    train_obs = observations.loc[observations.split == "train"]
    print(f"Training candidate rows: {len(train):,} from {train.trip_id.nunique()} trips", flush=True)

    # 1-2. Redundancy and single-feature signal
    sample, rho, constant, pairs = correlation(train, seed)
    rho.to_csv(out / "feature_correlation.csv")
    plot_correlation(rho, constant, out / "feature_correlation.png")
    uni = univariate(sample)
    uni.to_csv(out / "feature_univariate.csv", index=False)

    # 3-4. Permutation importance and drop-one ablation, same folds as the model harness
    folds = min(config["cv_folds"], train.trip_id.nunique())
    splits = list(GroupKFold(n_splits=folds).split(train, train.label, groups=train.trip_id))
    label = MODEL_CHOICES[args.model][0]
    pipeline = make_pipeline(MODEL_CHOICES[args.model][1](config))
    perm_rows, abl_rows = [], []
    for fold, (tr, va) in enumerate(splits, 1):
        fit_rows, val_rows = train.iloc[tr], train.iloc[va]
        val_obs = train_obs.loc[train_obs.trip_id.isin(val_rows.trip_id.unique())]
        model, base = fold_metrics(pipeline, fit_rows, val_rows, val_obs, FEATURES)
        abl_rows.append({"fold": fold, "dropped": "(none)", **base})
        X_val = val_rows[FEATURES].copy()
        for f in FEATURES:
            drops = []
            for _ in range(args.repeats):
                X_perm = X_val.copy()
                X_perm[f] = rng.permutation(X_perm[f].to_numpy())
                m = evaluate(model, val_rows, val_obs, X_perm)
                drops.append({k: base[k] - m[k] for k in METRICS})
            perm_rows.append({"fold": fold, "feature": f,
                              **{f"drop_{k}": float(np.mean([d[k] for d in drops])) for k in METRICS}})
        for f in FEATURES:
            kept = [g for g in FEATURES if g != f]
            _, m = fold_metrics(pipeline, fit_rows, val_rows, val_obs, kept)
            abl_rows.append({"fold": fold, "dropped": f, **m})
        print(f"  fold {fold}/{folds} done: base Macro F1 {base['macro_f1']:.4f}, Top-1 {base['top1_accuracy']:.4f}", flush=True)

    perm = pd.DataFrame(perm_rows)
    perm.to_csv(out / "permutation_importance_by_fold.csv", index=False)
    perm_summary = perm.groupby("feature").agg(
        mean_drop_macro_f1=("drop_macro_f1", "mean"), sd_drop_macro_f1=("drop_macro_f1", "std"),
        mean_drop_top1=("drop_top1_accuracy", "mean"), sd_drop_top1=("drop_top1_accuracy", "std"),
        mean_drop_log_loss=("drop_log_loss", "mean"),
    ).reset_index().sort_values("mean_drop_macro_f1", ascending=False)
    perm_summary.to_csv(out / "permutation_importance.csv", index=False)

    abl = pd.DataFrame(abl_rows)
    abl.to_csv(out / "ablation_by_fold.csv", index=False)
    base_by_fold = abl.loc[abl.dropped == "(none)"].set_index("fold")
    deltas = abl.loc[abl.dropped != "(none)"].copy()
    for k in METRICS:
        deltas[f"delta_{k}"] = deltas[k].to_numpy() - base_by_fold.loc[deltas.fold, k].to_numpy()
    abl_summary = deltas.groupby("dropped").agg(
        cv_macro_f1=("macro_f1", "mean"), delta_macro_f1=("delta_macro_f1", "mean"),
        sd_delta_macro_f1=("delta_macro_f1", "std"), cv_top1=("top1_accuracy", "mean"),
        delta_top1=("delta_top1_accuracy", "mean"), delta_log_loss=("delta_log_loss", "mean"),
    ).reset_index().rename(columns={"dropped": "feature"}).sort_values("delta_macro_f1")
    abl_summary.to_csv(out / "ablation.csv", index=False)
    base_mean = base_by_fold[list(METRICS)].mean()

    plot_bars(perm_summary, "mean_drop_macro_f1", "sd_drop_macro_f1",
              f"Permutation importance ({label})",
              "Drop in CV Macro F1 when the feature is shuffled", out / "permutation_importance.png")
    plot_bars(abl_summary, "delta_macro_f1", "sd_delta_macro_f1",
              f"Drop-one ablation ({label})",
              "Change in CV Macro F1 when the feature is removed", out / "ablation.png", diverging=True)

    # Report
    noise = float(base_by_fold.macro_f1.std())
    helps = abl_summary.loc[abl_summary.delta_macro_f1 <= -noise, "feature"].tolist()
    hurts = abl_summary.loc[abl_summary.delta_macro_f1 >= noise, "feature"].tolist()
    neutral = [f for f in abl_summary.feature if f not in helps + hurts]
    f4 = lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"
    f3 = lambda v: "n/a" if pd.isna(v) else f"{v:.3f}"
    lines = [
        "# Feature selection evidence", "",
        f"Training partition only: {train.trip_id.nunique()} trips, {len(train):,} labelled candidate rows. "
        f"{folds}-fold GroupKFold by trip (same folds as `pathsnap_p1.models.train`). Student-test trips not used. "
        f"Model: {label} with fold-fitted median imputation and scaling. Seed {seed}.", "",
        f"All-feature baseline: CV Macro F1 {base_mean.macro_f1:.4f} (fold SD {noise:.4f}), "
        f"CV Top-1 {base_mean.top1_accuracy:.4f}, CV log loss {base_mean.log_loss:.4f}.", "",
        "## 1. Redundancy (Spearman correlation)", "",
        f"Sample of {len(sample):,} training candidate rows. Figure: `feature_correlation.png`.", "",
        f"- Constant features (no information): {', '.join(constant) if constant else 'none'}",
        f"- Pairs with |ρ| ≥ {CORRELATION_LIMIT}: " + (", ".join(f"{a} / {b} (ρ = {r:.2f})" for a, b, r in pairs) if pairs else "none"),
        "", "## 2. Single-feature signal (ROC AUC on the same sample)", "",
        "Separability = max(AUC, 1 − AUC); 0.5 means no signal on its own.", "",
        table(uni, ["feature", "separability", "roc_auc", "missing_rate", "distinct_values", "direction"],
              {"separability": f3, "roc_auc": f3, "missing_rate": f3, "distinct_values": str, "direction": str}),
        "", "## 3. Permutation importance", "",
        f"Mean drop when one feature is shuffled in the validation fold ({args.repeats} repeats per fold). "
        "Figure: `permutation_importance.png`.", "",
        table(perm_summary, ["feature", "mean_drop_macro_f1", "sd_drop_macro_f1", "mean_drop_top1", "sd_drop_top1"],
              {k: f4 for k in ["mean_drop_macro_f1", "sd_drop_macro_f1", "mean_drop_top1", "sd_drop_top1"]}),
        "", "## 4. Drop-one ablation", "",
        "Model refitted without one feature in every fold; change versus the all-feature model in the same fold. "
        "Figure: `ablation.png`.", "",
        table(abl_summary, ["feature", "cv_macro_f1", "delta_macro_f1", "sd_delta_macro_f1", "cv_top1", "delta_top1", "delta_log_loss"],
              {k: f4 for k in ["cv_macro_f1", "delta_macro_f1", "sd_delta_macro_f1", "cv_top1", "delta_top1", "delta_log_loss"]}),
        "", "## Reading the ablation", "",
        f"Changes are compared with the all-feature model's fold-to-fold SD in Macro F1 ({noise:.4f}).", "",
        f"- Removing these lowers CV Macro F1 by more than one SD (the model needs them): {', '.join(helps) or 'none'}",
        f"- Removing these raises CV Macro F1 by more than one SD (candidates to drop, then re-check on other models): {', '.join(hurts) or 'none'}",
        f"- Within one SD (no clear effect for {label}): {', '.join(neutral) or 'none'}",
        "", ("Gaussian NB assumes independent features, so a feature that is neutral or harmful here can still "
             "matter for KNN or tree models. Compare with the run for the served model before changing `FEATURES`."
             if args.model == "gaussian_nb" else
             "This is the model the pipeline serves, so these results are the ones to act on. Removing a "
             "feature means re-training and re-tuning every model; record any decision in the report."),
        "", f"Runtime: {time.perf_counter() - start:.0f} s.",
    ]
    (out / "FEATURE_SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({"model": args.model, "folds": folds, "seed": seed, "repeats": args.repeats, "training_rows": len(train),
               "training_trips": int(train.trip_id.nunique()), "features": FEATURES,
               "elapsed_seconds": time.perf_counter() - start},
              open(out / "feature_selection_run.json", "w"), indent=2)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
