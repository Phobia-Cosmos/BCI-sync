# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.325030 |
| final_old_mf1 | 0.244180 |
| old_aaa | 0.563122 |
| old_aaf1 | 0.527146 |
| old_fr | 0.537294 |
| mean_current_before_acc | 0.502046 |
| mean_current_after_acc | 0.606228 |
| mean_current_acc_gain | 0.104182 |
| mean_current_before_mf1 | 0.414662 |
| mean_current_after_mf1 | 0.550950 |
| mean_current_mf1_gain | 0.136288 |
| final_seen_acc | 0.389126 |
| final_seen_mf1 | 0.216824 |
| bwt_acc | -0.217102 |
| bwt_mf1 | -0.334126 |
| mean_pseudo_acc_diagnostic_only | 0.606366 |
| mean_pseudo_mf1_diagnostic_only | 0.552746 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6600 | 0.6345 | 0.6198 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3198 | 0.0769 | 0.2706 | 0.3209 | 0.2726 |
| 4 | 27 | 0.3886 | 0.8682 | 0.2187 | 0.8295 | 0.8580 | 0.8246 |
| 5 | 60 | 0.6136 | 0.7455 | 0.5739 | 0.6957 | 0.7455 | 0.6968 |
| 6 | 5 | 0.4202 | 0.3405 | 0.2209 | 0.1637 | 0.3429 | 0.1670 |
