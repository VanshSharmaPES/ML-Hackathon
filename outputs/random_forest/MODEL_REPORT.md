# Model report — random_forest

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
| random_forest | 0.8522 | 0.0358 | 0.8839 | 0.8159 | 0.1240 | 1.45 |

## 3.2 — random_forest

**Family and suitability:** Tree ensemble. A Random Forest can model nonlinear interactions among distance, heading, speed and road-context features.

**Strengths and limitations:** Handles nonlinear feature effects and provides a useful ensemble baseline. A large forest can use substantial memory and prediction time; probabilities may require calibration.

**Configuration:** `{"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": 20, "max_features": "sqrt", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 2, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 100, "n_jobs": 2, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}`

**Objective / training rule:** Fits bootstrap decision trees and aggregates their class probabilities. The trees reduce variance compared with a single decision tree.

**Measured results:** CV macro F1 0.8522 (fold SD 0.0358); student-test macro precision 0.8939, macro recall 0.8746, macro F1 0.8839, Top-1 0.8159, log loss 0.1240.

**Generalisation evidence:** sampled training resubstitution macro F1 0.9185 (2,500 observations), CV 0.8522, student-test 0.8839. Sampled training minus CV gap is 0.0663; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.

**Efficiency:** final fit 366.95 seconds; student prediction 1.45 seconds over 153,280 candidate rows.

**Diagnostics:** see figures/random_forest_confusion.png and random_forest_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.

## Model comparison interpretation

Use the measured cross-validation and student-test metrics to compare these experiments. Model selection should consider generalisation, fold stability, probability quality, inference speed, route continuity and compatibility with the decoder. These results alone do not include Bayesian optimisation or HMM/Viterbi decoding.

Inspect the per-trip CSVs and maps before writing team conclusions. All members should verify and understand this generated results draft before submission.
