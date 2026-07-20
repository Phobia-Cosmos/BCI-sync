# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.604970 |
| final_old_mf1 | 0.577728 |
| old_aaa | 0.658253 |
| old_aaf1 | 0.637595 |
| old_fr | 0.138778 |
| mean_current_before_acc | 0.566298 |
| mean_current_after_acc | 0.589653 |
| mean_current_acc_gain | 0.023355 |
| mean_current_before_mf1 | 0.475969 |
| mean_current_after_mf1 | 0.511028 |
| mean_current_mf1_gain | 0.035058 |
| final_seen_acc | 0.548359 |
| final_seen_mf1 | 0.463455 |
| bwt_acc | -0.041294 |
| bwt_mf1 | -0.047573 |
| mean_pseudo_acc_diagnostic_only | 0.569453 |
| mean_pseudo_mf1_diagnostic_only | 0.512284 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7105 | 0.7631 | 0.7406 | 0.6360 | 0.6590 |
| 2 | 89 | 0.6950 | 0.6870 | 0.6325 | 0.6394 | 0.6150 | 0.5781 |
| 3 | 1 | 0.1860 | 0.2023 | 0.0745 | 0.0697 | 0.3291 | 0.2823 |
| 4 | 27 | 0.8477 | 0.8773 | 0.7868 | 0.8306 | 0.7898 | 0.7407 |
| 5 | 60 | 0.6136 | 0.7386 | 0.5726 | 0.6959 | 0.7000 | 0.6379 |
| 6 | 5 | 0.3607 | 0.3155 | 0.1503 | 0.0999 | 0.3524 | 0.1758 |
| 7 | 52 | 0.5864 | 0.8114 | 0.5091 | 0.7468 | 0.7295 | 0.6768 |
| 8 | 42 | 0.8026 | 0.6679 | 0.6943 | 0.6098 | 0.5974 | 0.5475 |
| 9 | 80 | 0.6419 | 0.6860 | 0.5098 | 0.6110 | 0.6616 | 0.6155 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2837 | 0.2092 |
