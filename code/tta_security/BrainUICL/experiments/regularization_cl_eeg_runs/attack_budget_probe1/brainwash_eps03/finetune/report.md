# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.704790 |
| final_old_mf1 | 0.689759 |
| old_aaa | 0.703623 |
| old_aaf1 | 0.688896 |
| old_fr | 0.003325 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.720930 |
| mean_current_acc_gain | -0.008140 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.755382 |
| mean_current_mf1_gain | -0.007734 |
| final_seen_acc | 0.720930 |
| final_seen_mf1 | 0.755382 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.712791 |
| mean_pseudo_mf1_diagnostic_only | 0.746535 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7209 | 0.7631 | 0.7554 | 0.7128 | 0.7465 |
