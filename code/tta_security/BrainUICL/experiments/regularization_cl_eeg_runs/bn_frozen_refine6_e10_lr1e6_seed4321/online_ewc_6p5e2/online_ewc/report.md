# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.654910 |
| final_old_mf1 | 0.632126 |
| old_aaa | 0.692224 |
| old_aaf1 | 0.674544 |
| old_fr | 0.067684 |
| mean_current_before_acc | 0.570740 |
| mean_current_after_acc | 0.589476 |
| mean_current_acc_gain | 0.018736 |
| mean_current_before_mf1 | 0.490322 |
| mean_current_after_mf1 | 0.511691 |
| mean_current_mf1_gain | 0.021369 |
| final_seen_acc | 0.572303 |
| final_seen_mf1 | 0.492660 |
| bwt_acc | -0.017172 |
| bwt_mf1 | -0.019031 |
| mean_pseudo_acc_diagnostic_only | 0.604862 |
| mean_pseudo_mf1_diagnostic_only | 0.550660 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7130 | 0.6939 | 0.6661 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1826 | 0.1942 | 0.0758 | 0.0697 | 0.3105 | 0.2611 |
| 4 | 27 | 0.8273 | 0.8750 | 0.7601 | 0.8255 | 0.8614 | 0.8150 |
| 5 | 60 | 0.6057 | 0.7148 | 0.5498 | 0.6610 | 0.7398 | 0.7004 |
| 6 | 5 | 0.3179 | 0.3155 | 0.0992 | 0.0965 | 0.3536 | 0.1789 |
