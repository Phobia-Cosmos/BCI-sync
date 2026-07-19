# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.715449 |
| final_old_mf1 | 0.697763 |
| old_aaa | 0.712823 |
| old_aaf1 | 0.695192 |
| old_fr | 0.018498 |
| mean_current_before_acc | 0.573629 |
| mean_current_after_acc | 0.573248 |
| mean_current_acc_gain | -0.000381 |
| mean_current_before_mf1 | 0.483890 |
| mean_current_after_mf1 | 0.483009 |
| mean_current_mf1_gain | -0.000882 |
| final_seen_acc | 0.574212 |
| final_seen_mf1 | 0.483742 |
| bwt_acc | 0.000965 |
| bwt_mf1 | 0.000734 |
| mean_pseudo_acc_diagnostic_only | 0.603465 |
| mean_pseudo_mf1_diagnostic_only | 0.546533 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7530 | 0.6939 | 0.6969 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1977 | 0.1977 | 0.0862 | 0.0810 | 0.3012 | 0.2483 |
| 4 | 27 | 0.8000 | 0.8011 | 0.7088 | 0.7097 | 0.8659 | 0.8192 |
| 5 | 60 | 0.6364 | 0.6466 | 0.5533 | 0.5609 | 0.7409 | 0.6897 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0981 | 0.0981 | 0.3488 | 0.1734 |
