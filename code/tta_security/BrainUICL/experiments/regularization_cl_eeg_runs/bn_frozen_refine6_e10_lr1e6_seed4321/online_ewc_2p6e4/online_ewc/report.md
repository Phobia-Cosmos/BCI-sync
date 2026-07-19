# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.713713 |
| final_old_mf1 | 0.698488 |
| old_aaa | 0.712156 |
| old_aaf1 | 0.697015 |
| old_fr | 0.016026 |
| mean_current_before_acc | 0.573359 |
| mean_current_after_acc | 0.570851 |
| mean_current_acc_gain | -0.002507 |
| mean_current_before_mf1 | 0.491947 |
| mean_current_after_mf1 | 0.486113 |
| mean_current_mf1_gain | -0.005834 |
| final_seen_acc | 0.569808 |
| final_seen_mf1 | 0.485553 |
| bwt_acc | -0.001043 |
| bwt_mf1 | -0.000560 |
| mean_pseudo_acc_diagnostic_only | 0.603690 |
| mean_pseudo_mf1_diagnostic_only | 0.548778 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7160 | 0.6939 | 0.6646 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1977 | 0.1930 | 0.0921 | 0.0783 | 0.3058 | 0.2557 |
| 4 | 27 | 0.8102 | 0.8170 | 0.7345 | 0.7397 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6352 | 0.6580 | 0.5637 | 0.5840 | 0.7398 | 0.6959 |
| 6 | 5 | 0.3060 | 0.3167 | 0.1044 | 0.0987 | 0.3512 | 0.1761 |
