# P1 model report — KNN and Gaussian Naive Bayes

Actual experiments, not placeholder results. These are two of the team's minimum five models. P2 must add the remaining models and Bayesian optimisation; P3 must evaluate HMM/Viterbi.

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
| knn | 0.8326 | 0.0540 | 0.8749 | 0.7978 | 0.3258 | 120.17 |
| gaussian_nb | 0.7239 | 0.0241 | 0.7228 | 0.7946 | 0.5731 | 0.09 |

## 3.2 — knn

**Family and suitability:** Instance-based. Similar point/candidate feature patterns may have similar labels; distance and heading interactions need not be linear.

**Strengths and limitations:** Flexible local decisions, but sensitive to scaling and noisy labels. Storage and inference cost increase with training size. Distance-weighted voting can produce overconfident probabilities.

**Configuration:** `{"algorithm": "kd_tree", "leaf_size": 40, "metric": "minkowski", "metric_params": null, "n_jobs": 2, "n_neighbors": 15, "p": 2, "weights": "distance"}`

**Objective / training rule:** No iterative loss is minimised. Training stores transformed examples; prediction uses distance-weighted neighbour votes. Training-set self-neighbours make resubstitution performance optimistic.

**Measured results:** CV macro F1 0.8326 (fold SD 0.0540); student-test macro precision 0.8797, macro recall 0.8703, macro F1 0.8749, Top-1 0.7978, log loss 0.3258.

**Generalisation evidence:** sampled training resubstitution macro F1 1.0000 (2,500 observations), CV 0.8326, student-test 0.8749. Sampled training minus CV gap is 0.1674; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.

**Efficiency:** final fit 8.93 seconds; student prediction 120.17 seconds over 153,280 candidate rows.

**Diagnostics:** see figures/knn_confusion.png and knn_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.

## 3.2 — gaussian_nb

**Family and suitability:** Probabilistic. A fast probabilistic baseline tests whether simple class-conditional feature distributions can distinguish candidate roads.

**Strengths and limitations:** Fast and compact. Conditional independence and Gaussian assumptions are imperfect: distance, rank, heading and previous-distance features can be correlated or skewed. Probabilities may be poorly calibrated.

**Configuration:** `{"priors": null, "var_smoothing": 1e-09}`

**Objective / training rule:** Estimates class priors and per-class feature means/variances; predicts by the Gaussian naive Bayes likelihood rule. Variance smoothing stabilises calculations.

**Measured results:** CV macro F1 0.7239 (fold SD 0.0241); student-test macro precision 0.6946, macro recall 0.8686, macro F1 0.7228, Top-1 0.7946, log loss 0.5731.

**Generalisation evidence:** sampled training resubstitution macro F1 0.7257 (2,500 observations), CV 0.7239, student-test 0.7228. Sampled training minus CV gap is 0.0018; discuss this alongside fold variability and per-trip failures. A single gap does not prove overfitting; low scores alone do not prove underfitting.

**Efficiency:** final fit 3.05 seconds; student prediction 0.09 seconds over 153,280 candidate rows.

**Diagnostics:** see figures/gaussian_nb_confusion.png and gaussian_nb_per_trip.csv. These models do not produce epoch-wise histories, so no artificial training-loss curves are supplied.

## Model comparison interpretation

Use the measured CV and student-test table to compare these two baselines. Probability quality and Top-1 accuracy both matter for subsequent decoding. No team-wide final model is selected here because the other three or more model experiments are outside P1. No Bayesian optimisation or HMM results are claimed.

Inspect the per-trip CSVs and maps before writing team conclusions. All members should verify and understand this generated results draft before submission.
