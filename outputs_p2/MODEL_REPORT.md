# Model report — logistic_regression, hist_gradient_boosting

Actual experiments from the shared training and evaluation harness. Scores use labels derived by monotonic alignment to the supplied reference routes; they are not independently verified GPS ground truth.

## Shared experimental design

- Training trips: 80; student-test trips: 10.
- GroupKFold keeps all points/candidates from each trip together. Median imputation and StandardScaler are fitted within each fold, then refitted on training only for final models.
- All generated labelled candidate rows are used; no negative downsampling or hidden reference-candidate injection.
- Candidate macro F1 averages the correct/incorrect classes. Observation Top-1 selects one road by score; ties use distance rank then arc ID. Candidate misses remain in the observation denominator.
- Scores measure agreement with geometry-derived reference alignment. This creates label bias, especially for distance-based methods; it is not independently verified ground truth.
- Model inference timings exclude upstream candidate generation and route decoding. Training and prediction code are shared in pathsnap_p1/models.py.
- Resubstitution diagnostics use a fixed random sample of up to 2,500 training observations (all candidates for each); fitting, cross-validation and student testing use all rows.

| Model | CV macro F1 | CV SD | Student macro F1 | Student Top-1 | Log loss | Inference seconds |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8297 | 0.0220 | 0.8516 | 0.7993 | 0.1651 | 0.03 |
| hist_gradient_boosting | 0.8396 | 0.0560 | 0.8785 | 0.8105 | 0.1283 | 0.21 |

## 3.2 — logistic_regression

**Family and suitability:** Linear probabilistic. Logistic Regression provides a fast, interpretable linear baseline for candidate correctness.

**Strengths and limitations:** Efficient and produces probabilities, but its linear decision boundary may miss nonlinear interactions among road features.

**Configuration:** `{"C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true, "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 2000, "n_jobs": null, "penalty": "deprecated", "random_state": 42, "solver": "lbfgs", "tol": 0.0001, "verbose": 0, "warm_start": false}`

**Objective / training rule:** Minimises regularised logistic loss; probabilities come from the fitted logistic function.

**Measured results:** CV macro F1 0.8297 (fold SD 0.0220); student-test macro precision 0.8741, macro recall 0.8325, macro F1 0.8516, Top-1 0.7993, log loss 0.1651.

**Generalisation evidence:** sampled training resubstitution macro F1 0.8410 (2,500 observations), CV 0.8297, student-test 0.8516. Sampled training minus CV gap is 0.0113; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.

**Efficiency:** final fit 6.14 seconds; student prediction 0.03 seconds over 153,280 candidate rows.

**Diagnostics:** see figures/logistic_regression_confusion.png and logistic_regression_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.

## 3.2 — hist_gradient_boosting

**Family and suitability:** Boosting ensemble. HistGradientBoosting can capture nonlinear feature relationships while using histogram-based split search.

**Strengths and limitations:** Models nonlinear interactions and can be efficient on large tables. Results depend on tree depth, learning rate and iteration count; probability calibration may vary.

**Configuration:** `{"categorical_features": "from_dtype", "class_weight": null, "early_stopping": "auto", "interaction_cst": null, "l2_regularization": 0.0, "learning_rate": 0.1, "loss": "log_loss", "max_bins": 255, "max_depth": null, "max_features": 1.0, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "monotonic_cst": null, "n_iter_no_change": 10, "random_state": 42, "scoring": "loss", "tol": 1e-07, "validation_fraction": 0.1, "verbose": 0, "warm_start": false}`

**Objective / training rule:** Adds shallow decision trees sequentially to reduce logistic loss; histogram bins speed up split search.

**Measured results:** CV macro F1 0.8396 (fold SD 0.0560); student-test macro precision 0.8879, macro recall 0.8698, macro F1 0.8785, Top-1 0.8105, log loss 0.1283.

**Generalisation evidence:** sampled training resubstitution macro F1 0.8836 (2,500 observations), CV 0.8396, student-test 0.8785. Sampled training minus CV gap is 0.0440; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.

**Efficiency:** final fit 9.80 seconds; student prediction 0.21 seconds over 153,280 candidate rows.

**Diagnostics:** see figures/hist_gradient_boosting_confusion.png and hist_gradient_boosting_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.

## Model comparison interpretation

Use the measured cross-validation and student-test metrics to compare these experiments. Model selection should consider generalisation, fold stability, probability quality, inference speed, route continuity and compatibility with the decoder. These results alone do not include Bayesian optimisation or HMM/Viterbi decoding.

Inspect the per-trip CSVs and maps before writing team conclusions. All members should verify and understand this generated results draft before submission.
