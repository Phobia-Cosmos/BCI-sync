# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.372275 |
| final_old_mf1 | 0.294237 |
| old_aaa | 0.534333 |
| old_aaf1 | 0.461532 |
| old_fr | 0.470037 |
| mean_current_before_acc | 0.442613 |
| mean_current_after_acc | 0.402221 |
| mean_current_acc_gain | -0.040392 |
| mean_current_before_mf1 | 0.371815 |
| mean_current_after_mf1 | 0.318529 |
| mean_current_mf1_gain | -0.053285 |
| final_seen_acc | 0.288426 |
| final_seen_mf1 | 0.186417 |
| bwt_acc | -0.113795 |
| bwt_mf1 | -0.132113 |
| mean_pseudo_acc_diagnostic_only | 0.280318 |
| mean_pseudo_mf1_diagnostic_only | 0.241563 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6186 | 0.7631 | 0.6465 | 0.3279 | 0.3039 |
| 2 | 89 | 0.5530 | 0.4530 | 0.4959 | 0.3827 | 0.2920 | 0.2822 |
| 3 | 1 | 0.1593 | 0.1500 | 0.0876 | 0.0977 | 0.2267 | 0.2089 |
| 4 | 27 | 0.6239 | 0.6023 | 0.5668 | 0.5503 | 0.2682 | 0.2357 |
| 5 | 60 | 0.4432 | 0.4307 | 0.3540 | 0.3277 | 0.2784 | 0.2506 |
| 6 | 5 | 0.2048 | 0.3131 | 0.1263 | 0.1036 | 0.3321 | 0.1986 |
| 7 | 52 | 0.6659 | 0.6682 | 0.4992 | 0.4863 | 0.3341 | 0.3123 |
| 8 | 42 | 0.2718 | 0.2410 | 0.3063 | 0.2856 | 0.2295 | 0.2117 |
| 9 | 80 | 0.5570 | 0.3453 | 0.3945 | 0.2382 | 0.2488 | 0.2246 |
| 10 | 26 | 0.2183 | 0.2000 | 0.1245 | 0.0667 | 0.2654 | 0.1870 |
