# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.570479 |
| final_old_mf1 | 0.489692 |
| old_aaa | 0.620822 |
| old_aaf1 | 0.562530 |
| old_fr | 0.187878 |
| mean_current_before_acc | 0.487684 |
| mean_current_after_acc | 0.465253 |
| mean_current_acc_gain | -0.022431 |
| mean_current_before_mf1 | 0.404078 |
| mean_current_after_mf1 | 0.382229 |
| mean_current_mf1_gain | -0.021849 |
| final_seen_acc | 0.422694 |
| final_seen_mf1 | 0.328346 |
| bwt_acc | -0.042560 |
| bwt_mf1 | -0.053883 |
| mean_pseudo_acc_diagnostic_only | 0.279899 |
| mean_pseudo_mf1_diagnostic_only | 0.242084 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6186 | 0.7631 | 0.6465 | 0.3279 | 0.3039 |
| 2 | 89 | 0.5530 | 0.5000 | 0.4959 | 0.4381 | 0.2920 | 0.2822 |
| 3 | 1 | 0.1860 | 0.1767 | 0.0741 | 0.0985 | 0.2291 | 0.2126 |
| 4 | 27 | 0.7125 | 0.7125 | 0.6308 | 0.6355 | 0.2682 | 0.2369 |
| 5 | 60 | 0.5682 | 0.5386 | 0.4873 | 0.4395 | 0.2784 | 0.2508 |
| 6 | 5 | 0.2845 | 0.2833 | 0.1134 | 0.1149 | 0.3298 | 0.1976 |
| 7 | 52 | 0.6932 | 0.6932 | 0.5317 | 0.5298 | 0.3364 | 0.3134 |
| 8 | 42 | 0.3231 | 0.3218 | 0.3973 | 0.4002 | 0.2192 | 0.2047 |
| 9 | 80 | 0.6186 | 0.6058 | 0.4532 | 0.4472 | 0.2488 | 0.2244 |
| 10 | 26 | 0.2087 | 0.2019 | 0.0941 | 0.0721 | 0.2692 | 0.1942 |
