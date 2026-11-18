"""Tune the HMM/Viterbi decoder costs on training trips, then compare classifier-only vs HMM.

1. Out-of-fold classifier scores for the 80 training trips: the served model's pipeline
   (artifacts/model.joblib) is refitted on 4 folds and scores the 5th (same trip-grouped folds
   as every other experiment), so no training trip is scored by a model that saw it.
2. Optuna TPE (seed 42) searches the decoder costs, maximising pooled out-of-fold Top-1 over
   all training points. Student_test and evaluator trips are not used here.
3. The served model (trained on all 80 training trips) scores the 10 student_test trips once;
   classifier-only and HMM decoding are compared there with the chosen costs.
4. Writes artifacts/hmm_config.json (the API then serves classifier + HMM) and outputs/hmm/.

    python -m hmm.tune_decoder --prepared <pathsnap_p1 prepare output dir> [--trials 40]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from api.scoring import continuity, route_overlap
from hmm.viterbi import DEFAULTS, prepare, viterbi
from pathsnap_p1.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
FLAGGED = ("00000054", "00000033")  # training trips mostly >50 m from their own route (ALIGNMENT_ANOMALIES.md)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CLASSIFIER, HMM = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_split(prepared: Path, split: str):
    cands, arcs = [], {}
    for d in sorted((prepared / "processed" / split).iterdir()):
        if d.is_dir():
            cands.append(pd.read_parquet(d / "candidates.parquet"))
            arcs[d.name] = pd.read_parquet(d / "arcs.parquet")
    return pd.concat(cands, ignore_index=True), arcs


def collapse(arcs):
    route = []
    for a in arcs:
        if a >= 0 and (not route or route[-1] != a):
            route.append(int(a))
    return route


def trip_metrics(prep, decoded, reference_by_obs, route, arcs_df):
    ref = np.array([reference_by_obs.get(int(o), -1) for o in prep["obs_ids"]])
    labelled = ref >= 0
    pred_route = collapse(decoded)
    overlap = route_overlap(pred_route, route)
    return {"points": int(labelled.sum()), "correct": int((decoded[labelled] == ref[labelled]).sum()),
            "route_f1": overlap["route_f1"], "continuity": continuity(pred_route, SimpleNamespace(arcs=arcs_df))}


def evaluate(prepped, cfg, refs, routes, arcs, hmm=True):
    rows = []
    for tid, prep in prepped.items():
        decoded = viterbi(prep, cfg) if hmm else prep["arc"][:, 0]
        rows.append({"trip_id": tid, **trip_metrics(prep, decoded, refs[tid], routes[tid], arcs[tid])})
    return pd.DataFrame(rows)


def pooled(df):
    return df.correct.sum() / df.points.sum()


def summarise(df, label):
    sub = df.loc[~df.trip_id.isin(FLAGGED)]
    return {"set": label, "top1": pooled(df), "top1_excl_flagged": pooled(sub),
            "route_f1_mean": df.route_f1.mean(), "continuity_mean": df.continuity.mean()}


def main():
    parser = argparse.ArgumentParser(description="Tune and evaluate the HMM/Viterbi decoder")
    parser.add_argument("--prepared", required=True, help="Output directory of `python -m pathsnap_p1 prepare`")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "hmm"))
    args = parser.parse_args()
    start_all = time.perf_counter()
    prepared, data_root, out = Path(args.prepared), Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(ROOT / "artifacts" / "model.joblib")
    pipeline, features = bundle["pipeline"], bundle["features"]
    gap = bundle["configuration"]["gap_seconds"]
    print(f"Served model: {bundle.get('model_name')} | features {len(features)}", flush=True)

    # 1. Out-of-fold scores on training trips
    train, train_arcs = load_split(prepared, "train")
    splits_ref = json.loads((ROOT / "outputs" / "cv_splits.json").read_text())
    fold_of = {t: f["fold"] for f in splits_ref for t in f["validation_trips"]}
    train["fold"] = train.trip_id.map(fold_of)
    train["p"] = np.nan
    fit_rows = train.label >= 0
    for fold in sorted(train.fold.unique()):
        tr = fit_rows & (train.fold != fold)
        va = train.fold == fold
        model = clone(pipeline).fit(train.loc[tr, features], train.loc[tr, "label"].to_numpy(int))
        train.loc[va, "p"] = model.predict_proba(train.loc[va, features])[:, 1]
        print(f"  out-of-fold scores: fold {fold} done", flush=True)

    def references(cands):
        r = cands.drop_duplicates(["trip_id", "observation_id"])
        return {t: dict(zip(g.observation_id.astype(int), g.reference_arc_id.astype(int))) for t, g in r.groupby("trip_id")}

    def routes_for(split, trips):
        return {t: [int(x) for x in (data_root / split / t / f"{t}.route").read_text().split()] for t in trips}

    t0 = time.perf_counter()
    prepped = {t: prepare(g, g.p.to_numpy(), train_arcs[t], gap) for t, g in train.groupby("trip_id")}
    refs, routes = references(train), routes_for("train", prepped)
    print(f"  transition structure built for {len(prepped)} trips in {time.perf_counter() - t0:.0f} s", flush=True)

    # 2. Optuna search on training trips only
    def objective(trial):
        cfg = {"emission_weight": trial.suggest_float("emission_weight", 0.2, 3.0, log=True),
               "cost_next": trial.suggest_float("cost_next", 0.0, 3.0),
               "cost_two_hop": trial.suggest_float("cost_two_hop", 0.0, 6.0),
               "cost_twin": trial.suggest_float("cost_twin", 0.0, 12.0),
               "cost_other": trial.suggest_float("cost_other", 0.0, 12.0)}
        return pooled(evaluate(prepped, cfg, refs, routes, train_arcs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({k: DEFAULTS[k] for k in ("emission_weight", "cost_next", "cost_two_hop", "cost_twin", "cost_other")})
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.perf_counter() - t0
    study.trials_dataframe().to_csv(out / "decoder_trials.csv", index=False)
    best = {**study.best_params, "gap_seconds": gap, "probability_floor": DEFAULTS["probability_floor"]}
    print(f"  best out-of-fold Top-1 {study.best_value:.4f} with {study.best_params}", flush=True)

    train_cls = evaluate(prepped, best, refs, routes, train_arcs, hmm=False)
    train_hmm = evaluate(prepped, best, refs, routes, train_arcs, hmm=True)
    per_fold = []
    for fold in sorted(set(fold_of.values())):
        trips = [t for t, f in fold_of.items() if f == fold]
        per_fold.append({"fold": fold, "classifier_only": pooled(train_cls[train_cls.trip_id.isin(trips)]),
                         "hmm": pooled(train_hmm[train_hmm.trip_id.isin(trips)])})
    per_fold = pd.DataFrame(per_fold)

    # 3. Student_test once, with the served model and the chosen costs
    student, student_arcs = load_split(prepared, "student_test")
    student["p"] = pipeline.predict_proba(student[features])[:, 1]
    s_prep = {t: prepare(g, g.p.to_numpy(), student_arcs[t], gap) for t, g in student.groupby("trip_id")}
    s_refs, s_routes = references(student), routes_for("student_test", s_prep)
    t0 = time.perf_counter()
    stu_hmm = evaluate(s_prep, best, s_refs, s_routes, student_arcs, hmm=True)
    decode_s = time.perf_counter() - t0
    stu_cls = evaluate(s_prep, best, s_refs, s_routes, student_arcs, hmm=False)

    # 4. Save config and results
    (ROOT / "artifacts" / "hmm_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    summary = pd.DataFrame([summarise(train_cls, "train out-of-fold, classifier only"),
                            summarise(train_hmm, "train out-of-fold, classifier + HMM"),
                            summarise(stu_cls, "student_test, classifier only"),
                            summarise(stu_hmm, "student_test, classifier + HMM")])
    summary.to_csv(out / "classifier_vs_hmm.csv", index=False)
    per_trip = stu_cls[["trip_id", "points"]].assign(
        top1_classifier=stu_cls.correct / stu_cls.points, top1_hmm=stu_hmm.correct / stu_hmm.points,
        route_f1_classifier=stu_cls.route_f1, route_f1_hmm=stu_hmm.route_f1,
        continuity_classifier=stu_cls.continuity, continuity_hmm=stu_hmm.continuity)
    per_trip.to_csv(out / "student_test_per_trip.csv", index=False)
    per_fold.to_csv(out / "train_per_fold.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    x = np.arange(len(per_fold))
    axes[0].bar(x - 0.19, per_fold.classifier_only, 0.36, color=CLASSIFIER, label="Classifier only", edgecolor=SURFACE, linewidth=2)
    axes[0].bar(x + 0.19, per_fold.hmm, 0.36, color=HMM, label="Classifier + HMM", edgecolor=SURFACE, linewidth=2)
    axes[0].set_xticks(x, [f"Fold {f}" for f in per_fold.fold])
    axes[0].set_title("Training trips, out-of-fold Top-1", loc="left")
    xs = np.arange(len(per_trip))
    axes[1].bar(xs - 0.19, per_trip.top1_classifier, 0.36, color=CLASSIFIER, edgecolor=SURFACE, linewidth=2)
    axes[1].bar(xs + 0.19, per_trip.top1_hmm, 0.36, color=HMM, edgecolor=SURFACE, linewidth=2)
    axes[1].set_xticks(xs, [t[-2:] for t in per_trip.trip_id])
    axes[1].set_xlabel("student_test trip (last two digits)")
    axes[1].set_title("student_test, Top-1 per trip", loc="left")
    for ax in axes:
        ax.set_ylim(0.3, 1.0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.suptitle("Classifier only vs classifier + HMM/Viterbi", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, "Axis starts at 0.3. Decoder costs tuned on training trips only; student_test scored once.",
             color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out / "classifier_vs_hmm.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    f4 = lambda v: f"{v:.4f}"
    lines = ["# HMM / Viterbi decoding: classifier only vs classifier + HMM", "",
             f"Classifier: {bundle.get('model_name')} (`artifacts/model.joblib`). Decoder: `hmm/viterbi.py`.", "",
             "**Hidden states**: each GPS point's candidate arcs (up to 8). **Emissions**: "
             "`emission_weight x log P(candidate correct)` from the classifier. **Transitions** by road connectivity "
             "between consecutive candidates: same arc (free), next connected arc, two hops ahead, reverse twin "
             "(direction flip), anything else. After a time gap above "
             f"{gap:g} s the sequence restarts with no transition cost. Viterbi finds the best sequence.", "",
             "**Costs** were tuned with Optuna TPE (seed 42, "
             f"{args.trials} trials, starting from the defaults) to maximise pooled Top-1 on **out-of-fold** scores "
             "of the 80 training trips (each trip scored by a model trained on the other folds). Student_test was "
             "scored once afterwards; evaluator trips were not used.", "",
             "| Set | Top-1 | Top-1 excl. 2 flagged trips | Route F1 (mean per trip) | Continuity (mean per trip) |",
             "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.set} | {f4(r.top1)} | {f4(r.top1_excl_flagged)} | {f4(r.route_f1_mean)} | {f4(r.continuity_mean)} |")
    lines += ["", f"Flagged trips (GPS mostly >50 m from their route): {', '.join(FLAGGED)}.", "",
              "## Chosen costs (`artifacts/hmm_config.json`)", "", "```json", json.dumps(best, indent=2), "```", "",
              "## Training folds (out-of-fold Top-1)", "", "| Fold | Classifier only | + HMM | Gain |", "|---|---:|---:|---:|"]
    for r in per_fold.itertuples():
        lines.append(f"| {r.fold} | {f4(r.classifier_only)} | {f4(r.hmm)} | {r.hmm - r.classifier_only:+.4f} |")
    lines += ["", "## student_test per trip", "", "| Trip | Points | Top-1 classifier | Top-1 HMM | Route F1 classifier | Route F1 HMM | Continuity classifier | Continuity HMM |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_trip.itertuples():
        lines.append(f"| {r.trip_id} | {r.points:,} | {f4(r.top1_classifier)} | {f4(r.top1_hmm)} | {f4(r.route_f1_classifier)} | "
                     f"{f4(r.route_f1_hmm)} | {f4(r.continuity_classifier)} | {f4(r.continuity_hmm)} |")
    worse = per_trip.loc[per_trip.top1_hmm < per_trip.top1_classifier, "trip_id"].tolist()
    lines += ["", f"HMM decoding lowered Top-1 on: {', '.join(worse) or 'no student_test trip'}.", "",
              "## Cost", "", f"- Decoder search: {args.trials} trials in {search_s:.0f} s.",
              f"- Decoding all 10 student_test trips: {decode_s:.2f} s.", "",
              "Labels are derived by aligning GPS points to the reference route, so Top-1 measures agreement with "
              "those labels, not verified ground truth.", "", "Figure: `classifier_vs_hmm.png`.", "",
              f"Runtime: {time.perf_counter() - start_all:.0f} s."]
    (out / "HMM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.round(4).to_string(index=False), flush=True)
    print(f"Wrote {out} and artifacts/hmm_config.json", flush=True)


if __name__ == "__main__":
    main()
