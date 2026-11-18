# Bayesian hyperparameter optimisation: tuned vs untuned

Search: Optuna TPE sampler (seed 42), 50 trials per model, objective = mean 5-fold trip-grouped CV Macro F1 on the 80 training trips (`tune_p2_optuna.py`). Student-test and evaluator trips were not used in the search.

Comparison: untuned and tuned settings evaluated with the same folds, the same refit on all training trips and the same 10 student_test trips (`optimisation/tuned_vs_untuned.py`).

| Model | Variant | CV Macro F1 (SD) | CV Top-1 (SD) | CV log loss | Student Macro F1 | Student Top-1 | Student log loss | Inference s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | untuned | 0.8297 (0.0220) | 0.6943 (0.1314) | 0.2034 | 0.8516 | 0.7993 | 0.1651 | 0.04 |
| Logistic Regression | tuned | 0.8297 (0.0220) | 0.6943 (0.1315) | 0.2034 | 0.8516 | 0.7993 | 0.1651 | 0.09 |
| HistGradientBoosting | untuned | 0.8396 (0.0560) | 0.6983 (0.1343) | 0.1657 | 0.8785 | 0.8105 | 0.1283 | 0.14 |
| HistGradientBoosting | tuned | 0.8551 (0.0309) | 0.6987 (0.1340) | 0.1525 | 0.8823 | 0.8144 | 0.1234 | 0.39 |

Inference time is `predict_proba` on 153,280 student_test candidate rows.

## Where the tuning gain comes from (per fold)

| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Macro F1 gain | +0.0000 | -0.0001 | +0.0000 | +0.0000 | -0.0001 |
| Logistic Regression | Top-1 gain | -0.0001 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| HistGradientBoosting | Macro F1 gain | +0.0726 | -0.0002 | +0.0021 | +0.0014 | +0.0015 |
| HistGradientBoosting | Top-1 gain | +0.0006 | -0.0029 | +0.0072 | -0.0024 | -0.0004 |

Training trips with more than half of their GPS points over 50 m from their own reference route (data anomalies documented in `outputs/ALIGNMENT_ANOMALIES.md`), by validation fold:

- Fold 1: 00000054 (89% far)
- Fold 2: none
- Fold 3: none
- Fold 4: 00000033 (59% far)
- Fold 5: none

If the gain is concentrated in the fold that holds these trips, the tuned settings mainly make the model more robust to off-network GPS rather than better everywhere. Report it that way.

## Search cost

| Model | Complete trials | Failed | Search time (min) | Seconds per trial |
|---|---:|---:|---:|---:|
| Logistic Regression | 50 | 0 | 18.4 | 22.1 |
| HistGradientBoosting | 50 | 1 | 62.5 | 74.4 |

## Hyperparameters

- Logistic Regression, untuned: `{"C": 1.0, "class_weight": null, "max_iter": 2000}`
- Logistic Regression, tuned: `{"C": 5.2908100069206485, "class_weight": null, "max_iter": 3000}`
- HistGradientBoosting, untuned: `{"class_weight": null, "l2_regularization": 0.0, "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20}`
- HistGradientBoosting, tuned: `{"class_weight": null, "l2_regularization": 5.089809701047425e-05, "learning_rate": 0.05097983315308033, "max_iter": 160, "max_leaf_nodes": 63, "min_samples_leaf": 80}`

## Notes

- The Optuna best value is the best of 50 scores on the same folds it was searched on, so it is optimistic. The student_test columns are the independent check of whether tuning helped.
- The objective was candidate-level Macro F1 at a 0.5 threshold. The served pipeline picks the top-scoring candidate per point, which is Top-1; both are reported above.
- Logistic Regression: the tuned C gives the same predictions as the default (1.6M scaled rows, so regularisation strength barely matters). The search confirms the default; it is not a tuning gain.
- The final model is chosen across all five models in `outputs/model_selection/MODEL_SELECTION.md`.

Figures: `optuna_history.png`, `tuned_vs_untuned.png`.

Runtime of this comparison: 380 s.
