# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.707186 |
| final_old_mf1 | 0.687881 |
| old_aaa | 0.704820 |
| old_aaf1 | 0.687957 |
| old_fr | 0.006734 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.711628 |
| mean_current_acc_gain | -0.017442 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.742371 |
| mean_current_mf1_gain | -0.020745 |
| final_seen_acc | 0.711628 |
| final_seen_mf1 | 0.742371 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.706977 |
| mean_pseudo_mf1_diagnostic_only | 0.736654 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7116 | 0.7631 | 0.7424 | 0.7070 | 0.7367 |
