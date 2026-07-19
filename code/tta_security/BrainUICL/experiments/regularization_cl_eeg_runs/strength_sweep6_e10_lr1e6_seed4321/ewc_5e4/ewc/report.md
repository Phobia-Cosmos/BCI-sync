# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.316826 |
| final_old_mf1 | 0.240275 |
| old_aaa | 0.562592 |
| old_aaf1 | 0.527174 |
| old_fr | 0.548973 |
| mean_current_before_acc | 0.501424 |
| mean_current_after_acc | 0.604433 |
| mean_current_acc_gain | 0.103009 |
| mean_current_before_mf1 | 0.413498 |
| mean_current_after_mf1 | 0.548199 |
| mean_current_mf1_gain | 0.134701 |
| final_seen_acc | 0.378600 |
| final_seen_mf1 | 0.207751 |
| bwt_acc | -0.225833 |
| bwt_mf1 | -0.340448 |
| mean_pseudo_acc_diagnostic_only | 0.605225 |
| mean_pseudo_mf1_diagnostic_only | 0.551315 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6620 | 0.6345 | 0.6213 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3198 | 0.0769 | 0.2695 | 0.3198 | 0.2706 |
| 4 | 27 | 0.3898 | 0.8625 | 0.2200 | 0.8263 | 0.8534 | 0.8193 |
| 5 | 60 | 0.6159 | 0.7443 | 0.5729 | 0.6934 | 0.7443 | 0.6942 |
| 6 | 5 | 0.4131 | 0.3345 | 0.2135 | 0.1523 | 0.3429 | 0.1683 |
