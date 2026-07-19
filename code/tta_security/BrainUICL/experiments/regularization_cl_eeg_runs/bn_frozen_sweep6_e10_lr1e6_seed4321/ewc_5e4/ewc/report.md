# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.711916 |
| final_old_mf1 | 0.695760 |
| old_aaa | 0.711856 |
| old_aaf1 | 0.696469 |
| old_fr | 0.013469 |
| mean_current_before_acc | 0.572777 |
| mean_current_after_acc | 0.574211 |
| mean_current_acc_gain | 0.001434 |
| mean_current_before_mf1 | 0.492131 |
| mean_current_after_mf1 | 0.488544 |
| mean_current_mf1_gain | -0.003587 |
| final_seen_acc | 0.568532 |
| final_seen_mf1 | 0.482181 |
| bwt_acc | -0.005679 |
| bwt_mf1 | -0.006363 |
| mean_pseudo_acc_diagnostic_only | 0.603298 |
| mean_pseudo_mf1_diagnostic_only | 0.548068 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7180 | 0.6939 | 0.6653 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1965 | 0.1919 | 0.0906 | 0.0785 | 0.3070 | 0.2577 |
| 4 | 27 | 0.8102 | 0.8205 | 0.7356 | 0.7427 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6341 | 0.6739 | 0.5607 | 0.5953 | 0.7386 | 0.6935 |
| 6 | 5 | 0.3048 | 0.3167 | 0.1088 | 0.0980 | 0.3488 | 0.1722 |
