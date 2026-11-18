# Model selection

All five models and the nearest-arc baseline, on the same 5 trip-grouped folds of the 80 training trips (fold assignments checked identical across `outputs/`, `outputs_p2/`, `outputs/random_forest/`).

**Rule** (training trips only; student_test is reported but never used to choose):

1. Primary metric: **CV Top-1**. The served pipeline picks the top-scoring candidate per GPS point, so Top-1 measures what it actually does (candidate-level Macro F1 uses a 0.5 threshold the pipeline never applies).
2. Models within **0.005** of the best CV Top-1 count as tied.
3. Ties are broken by the **lowest CV log loss**, because the HMM/Viterbi decoder will use the scores as emission probabilities.

| Model | Variant | CV Top-1 (SD) | CV log loss | CV Macro F1 | Student Top-1 | Student Macro F1 | Student log loss | File MB | Tied | Selected |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| Nearest-arc baseline (no training) | n/a | 0.7196 (0.1378) | n/a | n/a | 0.8248 | n/a | n/a | 0.0 |  |  |
| Random Forest | untuned | 0.6991 (0.1340) | 0.1604 | 0.8522 | 0.8159 | 0.8839 | 0.1240 | 74.65 | yes |  |
| HistGradientBoosting | tuned | 0.6987 (0.1340) | 0.1525 | 0.8551 | 0.8144 | 0.8823 | 0.1234 | n/a | yes | **yes** |
| HistGradientBoosting | untuned | 0.6983 (0.1343) | 0.1657 | 0.8396 | 0.8105 | 0.8785 | 0.1283 | 0.13 | yes |  |
| Logistic Regression | untuned | 0.6943 (0.1315) | 0.2034 | 0.8297 | 0.7993 | 0.8516 | 0.1651 | 0.0 | yes |  |
| Logistic Regression | tuned | 0.6943 (0.1315) | 0.2033 | 0.8297 | 0.7993 | 0.8516 | 0.1651 | n/a | yes |  |
| Gaussian NB | untuned | 0.6871 (0.1312) | 0.5745 | 0.7239 | 0.7946 | 0.7228 | 0.5731 | n/a |  |  |
| KNN | untuned | 0.6850 (0.1345) | 0.4902 | 0.8326 | 0.7978 | 0.8749 | 0.3258 | n/a |  |  |

## Selected: HistGradientBoosting (tuned)

- CV Top-1 0.6987; tied (within 0.005) with: Logistic Regression (untuned), HistGradientBoosting (untuned), Random Forest (untuned), Logistic Regression (tuned)
- Lowest CV log loss among the tied models: 0.1525
- Export: exported to `artifacts/model.joblib` (0.56 MB); refit check: student Top-1 0.8144 (table: 0.8144)

## Against the nearest-arc baseline (per fold, Top-1)

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting (tuned) | 0.4326 | 0.7464 | 0.7497 | 0.7743 | 0.7906 | 0.6987 |
| Nearest-arc baseline | 0.4477 | 0.7674 | 0.8127 | 0.7583 | 0.8119 | 0.7196 |

The selected model beats the baseline on 1 of 5 folds. Student_test Top-1 (points-weighted): model 0.8144 vs baseline 0.8248.

## Caveats

- Labels come from aligning GPS points to the reference route by distance, so they partly favour distance-based rules; scores measure agreement with derived labels, not verified ground truth.
- Student_test is easier than training (top-8 candidate miss rate 0.34% vs 9.1%), so student Top-1 (about 0.81) is far above CV Top-1 (about 0.70). Compare models on CV.
- Random Forest was not tuned; Logistic Regression and HistGradientBoosting were (Optuna, 50 trials).
- Inference times in the source files were measured on different laptops, so they are not compared here.
- Most of the classifier's practical gain is route continuity, not Top-1: see `outputs/e2e_student_test.csv` vs `outputs/e2e_student_test_baseline.csv`.

Figure: `model_selection.png`.

Runtime: 80 s.
