# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.648563 |
| final_old_mf1 | 0.624290 |
| old_aaa | 0.689675 |
| old_aaf1 | 0.669430 |
| old_fr | 0.076720 |
| mean_current_before_acc | 0.575903 |
| mean_current_after_acc | 0.589827 |
| mean_current_acc_gain | 0.013924 |
| mean_current_before_mf1 | 0.493752 |
| mean_current_after_mf1 | 0.508659 |
| mean_current_mf1_gain | 0.014907 |
| final_seen_acc | 0.571255 |
| final_seen_mf1 | 0.488161 |
| bwt_acc | -0.018572 |
| bwt_mf1 | -0.020497 |
| mean_pseudo_acc_diagnostic_only | 0.605431 |
| mean_pseudo_mf1_diagnostic_only | 0.550514 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7150 | 0.6939 | 0.6645 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1884 | 0.1965 | 0.0747 | 0.0703 | 0.3081 | 0.2583 |
| 4 | 27 | 0.8273 | 0.8648 | 0.7575 | 0.8016 | 0.8648 | 0.8213 |
| 5 | 60 | 0.6273 | 0.7216 | 0.5694 | 0.6666 | 0.7409 | 0.6949 |
| 6 | 5 | 0.3214 | 0.3167 | 0.1040 | 0.0975 | 0.3548 | 0.1800 |
