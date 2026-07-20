# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.655569 |
| final_old_mf1 | 0.614623 |
| old_aaa | 0.664850 |
| old_aaf1 | 0.628646 |
| old_fr | 0.066746 |
| mean_current_before_acc | 0.526858 |
| mean_current_after_acc | 0.511259 |
| mean_current_acc_gain | -0.015598 |
| mean_current_before_mf1 | 0.436839 |
| mean_current_after_mf1 | 0.421107 |
| mean_current_mf1_gain | -0.015731 |
| final_seen_acc | 0.505161 |
| final_seen_mf1 | 0.414616 |
| bwt_acc | -0.006099 |
| bwt_mf1 | -0.006492 |
| mean_pseudo_acc_diagnostic_only | 0.283596 |
| mean_pseudo_mf1_diagnostic_only | 0.242991 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6186 | 0.7631 | 0.6465 | 0.3279 | 0.3039 |
| 2 | 89 | 0.5530 | 0.5390 | 0.4959 | 0.4809 | 0.2920 | 0.2822 |
| 3 | 1 | 0.1860 | 0.1826 | 0.0743 | 0.0725 | 0.2326 | 0.2124 |
| 4 | 27 | 0.7398 | 0.7352 | 0.6575 | 0.6517 | 0.2727 | 0.2359 |
| 5 | 60 | 0.6489 | 0.6477 | 0.5656 | 0.5643 | 0.2852 | 0.2528 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0976 | 0.0976 | 0.3405 | 0.1998 |
| 7 | 52 | 0.7364 | 0.7307 | 0.6308 | 0.6225 | 0.3250 | 0.2985 |
| 8 | 42 | 0.5321 | 0.5154 | 0.5316 | 0.5229 | 0.2346 | 0.2147 |
| 9 | 80 | 0.6267 | 0.6267 | 0.4854 | 0.4855 | 0.2581 | 0.2342 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2673 | 0.1955 |
