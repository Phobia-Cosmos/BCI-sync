# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.544491 |
| final_old_mf1 | 0.530477 |
| old_aaa | 0.660796 |
| old_aaf1 | 0.642654 |
| old_fr | 0.224874 |
| mean_current_before_acc | 0.575155 |
| mean_current_after_acc | 0.596176 |
| mean_current_acc_gain | 0.021022 |
| mean_current_before_mf1 | 0.495950 |
| mean_current_after_mf1 | 0.520688 |
| mean_current_mf1_gain | 0.024738 |
| final_seen_acc | 0.518597 |
| final_seen_mf1 | 0.444384 |
| bwt_acc | -0.077580 |
| bwt_mf1 | -0.076304 |
| mean_pseudo_acc_diagnostic_only | 0.604682 |
| mean_pseudo_mf1_diagnostic_only | 0.550069 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7110 | 0.6939 | 0.6605 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1837 | 0.2012 | 0.0760 | 0.0700 | 0.3081 | 0.2584 |
| 4 | 27 | 0.8375 | 0.8807 | 0.7666 | 0.8373 | 0.8636 | 0.8188 |
| 5 | 60 | 0.6136 | 0.7443 | 0.5676 | 0.7056 | 0.7364 | 0.6952 |
| 6 | 5 | 0.3250 | 0.3155 | 0.1086 | 0.0993 | 0.3560 | 0.1795 |
