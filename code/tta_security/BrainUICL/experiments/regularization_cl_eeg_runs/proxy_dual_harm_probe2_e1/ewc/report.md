# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.699341 |
| final_old_mf1 | 0.678340 |
| old_aaa | 0.698822 |
| old_aaf1 | 0.680362 |
| old_fr | 0.004433 |
| mean_current_before_acc | 0.739535 |
| mean_current_after_acc | 0.707442 |
| mean_current_acc_gain | -0.032093 |
| mean_current_before_mf1 | 0.723378 |
| mean_current_after_mf1 | 0.691945 |
| mean_current_mf1_gain | -0.031433 |
| final_seen_acc | 0.710930 |
| final_seen_mf1 | 0.694310 |
| bwt_acc | 0.003488 |
| bwt_mf1 | 0.002365 |
| mean_pseudo_acc_diagnostic_only | 0.581733 |
| mean_pseudo_mf1_diagnostic_only | 0.569531 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6849 | 0.7631 | 0.7216 | 0.5605 | 0.5796 |
| 2 | 89 | 0.7500 | 0.7300 | 0.6836 | 0.6623 | 0.6030 | 0.5595 |
