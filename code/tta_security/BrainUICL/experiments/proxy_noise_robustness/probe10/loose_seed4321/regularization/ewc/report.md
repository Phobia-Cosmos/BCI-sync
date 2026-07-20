# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.587425 |
| final_old_mf1 | 0.512482 |
| old_aaa | 0.625351 |
| old_aaf1 | 0.568476 |
| old_fr | 0.163754 |
| mean_current_before_acc | 0.488317 |
| mean_current_after_acc | 0.467154 |
| mean_current_acc_gain | -0.021163 |
| mean_current_before_mf1 | 0.404196 |
| mean_current_after_mf1 | 0.380947 |
| mean_current_mf1_gain | -0.023249 |
| final_seen_acc | 0.430968 |
| final_seen_mf1 | 0.336469 |
| bwt_acc | -0.036186 |
| bwt_mf1 | -0.044478 |
| mean_pseudo_acc_diagnostic_only | 0.279847 |
| mean_pseudo_mf1_diagnostic_only | 0.241777 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6186 | 0.7631 | 0.6465 | 0.3279 | 0.3039 |
| 2 | 89 | 0.5530 | 0.5000 | 0.4959 | 0.4380 | 0.2920 | 0.2822 |
| 3 | 1 | 0.1860 | 0.1744 | 0.0741 | 0.0970 | 0.2302 | 0.2134 |
| 4 | 27 | 0.7148 | 0.7159 | 0.6323 | 0.6372 | 0.2682 | 0.2369 |
| 5 | 60 | 0.5716 | 0.5386 | 0.4935 | 0.4398 | 0.2784 | 0.2507 |
| 6 | 5 | 0.2798 | 0.3095 | 0.1134 | 0.1180 | 0.3298 | 0.1975 |
| 7 | 52 | 0.6966 | 0.6955 | 0.5385 | 0.5318 | 0.3364 | 0.3133 |
| 8 | 42 | 0.3295 | 0.3167 | 0.4027 | 0.3888 | 0.2218 | 0.2061 |
| 9 | 80 | 0.6209 | 0.6023 | 0.4564 | 0.4456 | 0.2465 | 0.2228 |
| 10 | 26 | 0.2019 | 0.2000 | 0.0721 | 0.0667 | 0.2673 | 0.1910 |
