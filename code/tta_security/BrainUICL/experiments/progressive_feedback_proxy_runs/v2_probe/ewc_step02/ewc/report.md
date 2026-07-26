# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.700240 |
| final_old_mf1 | 0.686283 |
| old_aaa | 0.701347 |
| old_aaf1 | 0.687158 |
| old_fr | 0.003154 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.724419 |
| mean_current_acc_gain | -0.004651 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.759419 |
| mean_current_mf1_gain | -0.003697 |
| final_seen_acc | 0.724419 |
| final_seen_mf1 | 0.759419 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.202326 |
| mean_pseudo_mf1_diagnostic_only | 0.109409 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7594 | 0.2023 | 0.1094 |
