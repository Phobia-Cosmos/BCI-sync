# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.705329 |
| final_old_mf1 | 0.687696 |
| old_aaa | 0.709829 |
| old_aaf1 | 0.694564 |
| old_fr | 0.004092 |
| mean_current_before_acc | 0.574369 |
| mean_current_after_acc | 0.579618 |
| mean_current_acc_gain | 0.005249 |
| mean_current_before_mf1 | 0.491768 |
| mean_current_after_mf1 | 0.495714 |
| mean_current_mf1_gain | 0.003946 |
| final_seen_acc | 0.576495 |
| final_seen_mf1 | 0.494155 |
| bwt_acc | -0.003123 |
| bwt_mf1 | -0.001559 |
| mean_pseudo_acc_diagnostic_only | 0.604466 |
| mean_pseudo_mf1_diagnostic_only | 0.550310 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7140 | 0.6939 | 0.6650 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1930 | 0.1953 | 0.0876 | 0.0716 | 0.3081 | 0.2584 |
| 4 | 27 | 0.8159 | 0.8386 | 0.7459 | 0.7650 | 0.8625 | 0.8178 |
| 5 | 60 | 0.6295 | 0.6886 | 0.5613 | 0.6235 | 0.7398 | 0.6990 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0988 | 0.0976 | 0.3524 | 0.1781 |
