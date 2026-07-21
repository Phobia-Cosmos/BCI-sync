# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.684072 |
| final_old_mf1 | 0.657411 |
| old_aaa | 0.693691 |
| old_aaf1 | 0.672570 |
| old_fr | 0.026170 |
| mean_current_before_acc | 0.578997 |
| mean_current_after_acc | 0.591412 |
| mean_current_acc_gain | 0.012415 |
| mean_current_before_mf1 | 0.488168 |
| mean_current_after_mf1 | 0.508342 |
| mean_current_mf1_gain | 0.020174 |
| final_seen_acc | 0.571199 |
| final_seen_mf1 | 0.481639 |
| bwt_acc | -0.020213 |
| bwt_mf1 | -0.026703 |
| mean_pseudo_acc_diagnostic_only | 0.588286 |
| mean_pseudo_mf1_diagnostic_only | 0.540911 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7012 | 0.7631 | 0.7341 | 0.6581 | 0.6708 |
| 2 | 89 | 0.7080 | 0.6940 | 0.6488 | 0.6442 | 0.6610 | 0.6222 |
| 3 | 1 | 0.1860 | 0.1965 | 0.0740 | 0.0702 | 0.3070 | 0.2609 |
| 4 | 27 | 0.8273 | 0.8648 | 0.7536 | 0.8009 | 0.8614 | 0.8150 |
| 5 | 60 | 0.5977 | 0.6989 | 0.5454 | 0.6395 | 0.7420 | 0.7045 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0961 | 0.3548 | 0.1868 |
| 7 | 52 | 0.7580 | 0.8034 | 0.6840 | 0.7428 | 0.8080 | 0.7670 |
| 8 | 42 | 0.8103 | 0.7667 | 0.6981 | 0.7050 | 0.4744 | 0.4643 |
| 9 | 80 | 0.6570 | 0.6733 | 0.5504 | 0.5840 | 0.7384 | 0.7125 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2779 | 0.2051 |
