# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `robust_feature`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.725000 |
| final_old_mf1 | 0.712259 |
| old_aaa | 0.726875 |
| old_aaf1 | 0.709299 |
| old_fr | 0.025210 |
| mean_current_before_acc | 0.626562 |
| mean_current_after_acc | 0.635156 |
| mean_current_acc_gain | 0.008594 |
| mean_current_before_mf1 | 0.467175 |
| mean_current_after_mf1 | 0.487055 |
| mean_current_mf1_gain | 0.019881 |
| final_seen_acc | 0.639063 |
| final_seen_mf1 | 0.491589 |
| bwt_acc | 0.003906 |
| bwt_mf1 | 0.004534 |
| mean_pseudo_acc_diagnostic_only | 0.630768 |
| mean_pseudo_mf1_diagnostic_only | 0.589029 |
| pseudo_label_coverage | 1.000000 |
| robust_mean_protected_fraction | 0.169922 |
| robust_mean_lambda | 0.649914 |
| robust_mean_defense_loss | 0.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7937 | 0.7812 | 0.6414 | 0.6323 | 0.7302 | 0.7625 |
| 2 | 89 | 0.7781 | 0.7719 | 0.4694 | 0.4854 | 0.7750 | 0.7104 |
| 3 | 1 | 0.1719 | 0.2031 | 0.0665 | 0.0936 | 0.2349 | 0.1654 |
| 4 | 27 | 0.7625 | 0.7844 | 0.6915 | 0.7369 | 0.7830 | 0.7178 |
