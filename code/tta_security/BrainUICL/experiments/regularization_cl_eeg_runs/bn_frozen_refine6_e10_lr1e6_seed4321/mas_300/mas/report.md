# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.608982 |
| final_old_mf1 | 0.588967 |
| old_aaa | 0.677562 |
| old_aaf1 | 0.657744 |
| old_fr | 0.133066 |
| mean_current_before_acc | 0.576273 |
| mean_current_after_acc | 0.592658 |
| mean_current_acc_gain | 0.016386 |
| mean_current_before_mf1 | 0.495548 |
| mean_current_after_mf1 | 0.513350 |
| mean_current_mf1_gain | 0.017802 |
| final_seen_acc | 0.554082 |
| final_seen_mf1 | 0.474837 |
| bwt_acc | -0.038576 |
| bwt_mf1 | -0.038513 |
| mean_pseudo_acc_diagnostic_only | 0.605245 |
| mean_pseudo_mf1_diagnostic_only | 0.550634 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7160 | 0.6939 | 0.6670 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1860 | 0.2000 | 0.0763 | 0.0707 | 0.3116 | 0.2623 |
| 4 | 27 | 0.8352 | 0.8705 | 0.7675 | 0.8153 | 0.8648 | 0.8213 |
| 5 | 60 | 0.6239 | 0.7284 | 0.5685 | 0.6779 | 0.7375 | 0.6926 |
| 6 | 5 | 0.3214 | 0.3167 | 0.1040 | 0.0977 | 0.3536 | 0.1789 |
