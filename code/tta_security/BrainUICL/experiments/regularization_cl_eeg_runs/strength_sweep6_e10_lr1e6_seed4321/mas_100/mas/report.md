# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.324970 |
| final_old_mf1 | 0.244476 |
| old_aaa | 0.563790 |
| old_aaf1 | 0.527657 |
| old_fr | 0.537380 |
| mean_current_before_acc | 0.502389 |
| mean_current_after_acc | 0.599077 |
| mean_current_acc_gain | 0.096688 |
| mean_current_before_mf1 | 0.415480 |
| mean_current_after_mf1 | 0.542460 |
| mean_current_mf1_gain | 0.126981 |
| final_seen_acc | 0.386461 |
| final_seen_mf1 | 0.212379 |
| bwt_acc | -0.212616 |
| bwt_mf1 | -0.330081 |
| mean_pseudo_acc_diagnostic_only | 0.603692 |
| mean_pseudo_mf1_diagnostic_only | 0.549701 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6620 | 0.6345 | 0.6216 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3058 | 0.0769 | 0.2528 | 0.3198 | 0.2707 |
| 4 | 27 | 0.3898 | 0.8545 | 0.2239 | 0.8198 | 0.8511 | 0.8171 |
| 5 | 60 | 0.6193 | 0.7341 | 0.5753 | 0.6820 | 0.7398 | 0.6913 |
| 6 | 5 | 0.4155 | 0.3345 | 0.2192 | 0.1523 | 0.3405 | 0.1637 |
