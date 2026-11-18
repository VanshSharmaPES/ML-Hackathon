# Bayesian hyperparameter optimisation: tuned vs untuned

Search: Optuna TPE sampler (seed 42), 50 trials per model, objective = mean 5-fold trip-grouped CV Macro F1 on the 80 training trips (`tune_p2_optuna.py`). Student-test and evaluator trips were not used in the search.

Comparison: untuned and tuned settings evaluated with the same folds, the same refit on all training trips and the same 10 student_test trips (`optimisation/tuned_vs_untuned.py`).

| Model | Variant | CV Macro F1 (SD) | Student Macro F1 | Student Top-1 | Student log loss | Inference s |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | untuned | 0.8297 (0.0220) | 0.8516 | 0.7993 | 0.1651 | 0.06 |
| Logistic Regression | tuned | 0.8297 (0.0220) | 0.8516 | 0.7993 | 0.1651 | 0.03 |
| HistGradientBoosting | untuned | 0.8396 (0.0560) | 0.8785 | 0.8105 | 0.1283 | 0.12 |
| HistGradientBoosting | tuned | 0.8551 (0.0309) | 0.8823 | 0.8144 | 0.1234 | 0.23 |

Inference time is `predict_proba` on 153,280 student_test candidate rows.

## Search cost

| Model | Complete trials | Failed | Search time (min) | Seconds per trial |
|---|---:|---:|---:|---:|
| Logistic Regression | 50 | 0 | 18.4 | 22.1 |
| HistGradientBoosting | 50 | 1 | 62.5 | 74.4 |

## Hyperparameters

- Logistic Regression, untuned: `{"C": 1.0, "class_weight": null, "max_iter": 2000}`
- Logistic Regression, tuned: `{"C": 5.2908100069206485, "class_weight": null}`
- HistGradientBoosting, untuned: `{"class_weight": null, "l2_regularization": 0.0, "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20}`
- HistGradientBoosting, tuned: `{"l2_regularization": 5.089809701047425e-05, "learning_rate": 0.05097983315308033, "max_iter": 160, "max_leaf_nodes": 63, "min_samples_leaf": 80}`

## Selected model

**HistGradientBoosting (tuned)**, chosen by highest tuned 5-fold CV Macro F1 on training trips (0.8551). Its student_test numbers are reported above, not used to choose.

Note: the Optuna best value is the best of 50 scores on the same folds it was searched on, so it is optimistic. The student_test columns are the unbiased check of whether tuning helped.

Figures: `optuna_history.png`, `tuned_vs_untuned.png`.

Runtime of this comparison: 252 s.
