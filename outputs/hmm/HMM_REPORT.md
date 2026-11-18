# HMM / Viterbi decoding: classifier only vs classifier + HMM

Classifier: HistGradientBoosting (tuned) (`artifacts/model.joblib`). Decoder: `hmm/viterbi.py`.

**Hidden states**: each GPS point's candidate arcs (up to 8). **Emissions**: `emission_weight x log P(candidate correct)` from the classifier. **Transitions** by road connectivity between consecutive candidates: same arc (free), next connected arc, two hops ahead, reverse twin (direction flip), anything else. After a time gap above 5 s the sequence restarts with no transition cost. Viterbi finds the best sequence.

**Costs** were tuned with Optuna TPE (seed 42, 40 trials, starting from the defaults) to maximise pooled Top-1 on **out-of-fold** scores of the 80 training trips (each trip scored by a model trained on the other folds). Student_test was scored once afterwards; evaluator trips were not used.

| Set | Top-1 | Top-1 excl. 2 flagged trips | Route F1 (mean per trip) | Continuity (mean per trip) |
|---|---:|---:|---:|---:|
| train out-of-fold, classifier only | 0.6986 | 0.7813 | 0.7795 | 0.7036 |
| train out-of-fold, classifier + HMM | 0.7976 | 0.8929 | 0.9461 | 0.9544 |
| student_test, classifier only | 0.8144 | 0.8144 | 0.7985 | 0.7300 |
| student_test, classifier + HMM | 0.9312 | 0.9312 | 0.9594 | 0.9784 |

Flagged trips (GPS mostly >50 m from their route): 00000054, 00000033.

## Chosen costs (`artifacts/hmm_config.json`)

```json
{
  "emission_weight": 0.20198523699751048,
  "cost_next": 0.0017215506074260145,
  "cost_two_hop": 5.656867434182339,
  "cost_twin": 11.339657483166942,
  "cost_other": 8.65856558057097,
  "gap_seconds": 5.0,
  "probability_floor": 1e-06
}
```

## Training folds (out-of-fold Top-1)

| Fold | Classifier only | + HMM | Gain |
|---|---:|---:|---:|
| 1 | 0.4326 | 0.5021 | +0.0695 |
| 2 | 0.7464 | 0.8726 | +0.1262 |
| 3 | 0.7497 | 0.8725 | +0.1228 |
| 4 | 0.7743 | 0.8311 | +0.0568 |
| 5 | 0.7906 | 0.9100 | +0.1195 |

## student_test per trip

| Trip | Points | Top-1 classifier | Top-1 HMM | Route F1 classifier | Route F1 HMM | Continuity classifier | Continuity HMM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00000000 | 2,503 | 0.9213 | 0.9668 | 0.8010 | 0.9532 | 0.6676 | 0.9727 |
| 00000007 | 1,354 | 0.9202 | 0.9926 | 0.9490 | 0.9878 | 0.9691 | 0.9917 |
| 00000009 | 1,049 | 0.8122 | 0.8513 | 0.7887 | 0.9220 | 0.6740 | 0.9784 |
| 00000040 | 2,995 | 0.8711 | 0.9796 | 0.8477 | 1.0000 | 0.8036 | 1.0000 |
| 00000052 | 262 | 0.9275 | 0.9389 | 0.8132 | 0.9617 | 0.7586 | 0.9885 |
| 00000056 | 3,958 | 0.8600 | 0.9133 | 0.8783 | 0.9608 | 0.6993 | 0.9883 |
| 00000065 | 768 | 0.7852 | 0.8672 | 0.7855 | 0.9932 | 0.6875 | 0.9732 |
| 00000076 | 1,486 | 0.7705 | 0.8964 | 0.6667 | 0.9285 | 0.5951 | 0.9424 |
| 00000077 | 3,911 | 0.6369 | 0.9207 | 0.6963 | 0.9798 | 0.7993 | 0.9681 |
| 00000091 | 874 | 0.8066 | 0.9039 | 0.7585 | 0.9074 | 0.6461 | 0.9810 |

HMM decoding lowered Top-1 on: no student_test trip.

## Cost

- Decoder search: 40 trials in 88 s.
- Decoding all 10 student_test trips: 0.22 s.

Labels are derived by aligning GPS points to the reference route, so Top-1 measures agreement with those labels, not verified ground truth.

Figure: `classifier_vs_hmm.png`.

Runtime: 197 s.
