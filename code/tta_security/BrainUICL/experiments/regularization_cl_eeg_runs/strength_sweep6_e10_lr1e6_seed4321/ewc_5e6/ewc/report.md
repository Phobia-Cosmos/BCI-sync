# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.329581 |
| final_old_mf1 | 0.247819 |
| old_aaa | 0.563909 |
| old_aaf1 | 0.528169 |
| old_fr | 0.530816 |
| mean_current_before_acc | 0.503011 |
| mean_current_after_acc | 0.604675 |
| mean_current_acc_gain | 0.101663 |
| mean_current_before_mf1 | 0.415328 |
| mean_current_after_mf1 | 0.550286 |
| mean_current_mf1_gain | 0.134958 |
| final_seen_acc | 0.393079 |
| final_seen_mf1 | 0.219957 |
| bwt_acc | -0.211595 |
| bwt_mf1 | -0.330328 |
| mean_pseudo_acc_diagnostic_only | 0.608441 |
| mean_pseudo_mf1_diagnostic_only | 0.554786 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6460 | 0.6345 | 0.6080 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3209 | 0.0769 | 0.2728 | 0.3186 | 0.2694 |
| 4 | 27 | 0.3898 | 0.8682 | 0.2200 | 0.8285 | 0.8682 | 0.8329 |
| 5 | 60 | 0.6159 | 0.7466 | 0.5768 | 0.6974 | 0.7500 | 0.7035 |
| 6 | 5 | 0.4226 | 0.3429 | 0.2206 | 0.1686 | 0.3429 | 0.1674 |
