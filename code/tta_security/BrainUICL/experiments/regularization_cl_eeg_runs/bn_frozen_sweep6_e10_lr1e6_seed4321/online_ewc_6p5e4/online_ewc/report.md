# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.713234 |
| final_old_mf1 | 0.697489 |
| old_aaa | 0.712010 |
| old_aaf1 | 0.696477 |
| old_fr | 0.015344 |
| mean_current_before_acc | 0.572592 |
| mean_current_after_acc | 0.568988 |
| mean_current_acc_gain | -0.003604 |
| mean_current_before_mf1 | 0.490733 |
| mean_current_after_mf1 | 0.484693 |
| mean_current_mf1_gain | -0.006040 |
| final_seen_acc | 0.567094 |
| final_seen_mf1 | 0.482637 |
| bwt_acc | -0.001894 |
| bwt_mf1 | -0.002057 |
| mean_pseudo_acc_diagnostic_only | 0.603100 |
| mean_pseudo_mf1_diagnostic_only | 0.547622 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7220 | 0.6939 | 0.6693 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1977 | 0.1919 | 0.0932 | 0.0782 | 0.3047 | 0.2543 |
| 4 | 27 | 0.8080 | 0.8136 | 0.7276 | 0.7354 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6341 | 0.6477 | 0.5610 | 0.5733 | 0.7398 | 0.6942 |
| 6 | 5 | 0.3048 | 0.3143 | 0.1057 | 0.1005 | 0.3488 | 0.1722 |
