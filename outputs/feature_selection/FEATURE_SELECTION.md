# Feature selection evidence

Training partition only: 80 trips, 1,667,336 labelled candidate rows. 5-fold GroupKFold by trip (same folds as `pathsnap_p1.models.train`). Student-test trips not used. Model: Gaussian NB with fold-fitted median imputation and scaling. Seed 42.

All-feature baseline: CV Macro F1 0.7239 (fold SD 0.0270), CV Top-1 0.6870, CV log loss 0.5745.

## 1. Redundancy (Spearman correlation)

Sample of 300,000 training candidate rows. Figure: `feature_correlation.png`.

- Constant features (no information): candidate_count
- Pairs with |ρ| ≥ 0.9: speed_missing / large_gap (ρ = 0.97)

## 2. Single-feature signal (ROC AUC on the same sample)

Separability = max(AUC, 1 − AUC); 0.5 means no signal on its own.

| feature | separability | roc_auc | missing_rate | distinct_values | direction |
|---|---:|---:|---:|---:|---:|
| distance_rank | 0.891 | 0.109 | 0.000 | 8 | lower = more likely correct |
| distance_m | 0.873 | 0.127 | 0.000 | 272861 | lower = more likely correct |
| previous_nearest_distance_m | 0.755 | 0.245 | 0.006 | 251155 | lower = more likely correct |
| heading_difference_deg | 0.743 | 0.257 | 0.013 | 296027 | lower = more likely correct |
| arc_length_m | 0.649 | 0.649 | 0.000 | 55821 | higher = more likely correct |
| speed_mps | 0.521 | 0.521 | 0.006 | 164709 | higher = more likely correct |
| time_gap_s | 0.501 | 0.501 | 0.000 | 120 | higher = more likely correct |
| large_gap | 0.501 | 0.501 | 0.000 | 2 | higher = more likely correct |
| speed_missing | 0.500 | 0.500 | 0.000 | 2 | higher = more likely correct |
| heading_missing | 0.500 | 0.500 | 0.000 | 2 | higher = more likely correct |
| candidate_count | n/a | n/a | 0.000 | 1 |  |

## 3. Permutation importance

Mean drop when one feature is shuffled in the validation fold (3 repeats per fold). Figure: `permutation_importance.png`.

| feature | mean_drop_macro_f1 | sd_drop_macro_f1 | mean_drop_top1 | sd_drop_top1 |
|---|---:|---:|---:|---:|
| distance_rank | 0.1180 | 0.0118 | 0.3533 | 0.0814 |
| heading_difference_deg | 0.0494 | 0.0257 | 0.2723 | 0.0777 |
| distance_m | 0.0351 | 0.0378 | 0.1850 | 0.0182 |
| previous_nearest_distance_m | 0.0082 | 0.0155 | 0.1247 | 0.0231 |
| arc_length_m | 0.0034 | 0.0049 | 0.0466 | 0.0230 |
| large_gap | 0.0012 | 0.0014 | 0.0139 | 0.0151 |
| speed_missing | 0.0011 | 0.0013 | 0.0136 | 0.0142 |
| speed_mps | 0.0004 | 0.0006 | 0.0008 | 0.0011 |
| time_gap_s | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| candidate_count | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heading_missing | -0.0002 | 0.0005 | 0.0011 | 0.0015 |

## 4. Drop-one ablation

Model refitted without one feature in every fold; change versus the all-feature model in the same fold. Figure: `ablation.png`.

| feature | cv_macro_f1 | delta_macro_f1 | sd_delta_macro_f1 | cv_top1 | delta_top1 | delta_log_loss |
|---|---:|---:|---:|---:|---:|---:|
| distance_rank | 0.6492 | -0.0747 | 0.0093 | 0.5319 | -0.1551 | 0.0756 |
| heading_difference_deg | 0.6838 | -0.0401 | 0.0180 | 0.4857 | -0.2014 | 0.1202 |
| arc_length_m | 0.6990 | -0.0249 | 0.0128 | 0.6976 | 0.0106 | -0.0182 |
| time_gap_s | 0.7204 | -0.0035 | 0.0018 | 0.6870 | 0.0000 | 0.0079 |
| large_gap | 0.7223 | -0.0017 | 0.0011 | 0.6870 | 0.0000 | -0.0268 |
| speed_missing | 0.7223 | -0.0016 | 0.0010 | 0.6870 | 0.0000 | -0.0253 |
| speed_mps | 0.7238 | -0.0001 | 0.0008 | 0.6871 | 0.0000 | 0.0006 |
| candidate_count | 0.7239 | 0.0000 | 0.0000 | 0.6871 | 0.0000 | 0.0000 |
| heading_missing | 0.7242 | 0.0003 | 0.0003 | 0.6870 | 0.0000 | 0.0040 |
| distance_m | 0.7937 | 0.0697 | 0.0600 | 0.6824 | -0.0047 | -0.1366 |
| previous_nearest_distance_m | 0.7977 | 0.0738 | 0.0237 | 0.6867 | -0.0003 | -0.1470 |

## Reading the ablation

Changes are compared with the all-feature model's fold-to-fold SD in Macro F1 (0.0270).

- Removing these lowers CV Macro F1 by more than one SD (the model needs them): distance_rank, heading_difference_deg
- Removing these raises CV Macro F1 by more than one SD (candidates to drop, then re-check on other models): distance_m, previous_nearest_distance_m
- Within one SD (no clear effect for Gaussian NB): arc_length_m, time_gap_s, large_gap, speed_missing, speed_mps, candidate_count, heading_missing

Gaussian NB assumes independent features, so a feature that is neutral here can still matter for KNN or tree models. Confirm any removal with the model the team selects before changing `FEATURES`.

Runtime: 306 s.
