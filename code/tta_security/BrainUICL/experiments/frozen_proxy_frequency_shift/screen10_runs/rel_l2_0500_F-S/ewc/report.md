# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.679581 |
| final_old_mf1 | 0.650359 |
| old_aaa | 0.690250 |
| old_aaf1 | 0.667510 |
| old_fr | 0.032563 |
| mean_current_before_acc | 0.576560 |
| mean_current_after_acc | 0.590149 |
| mean_current_acc_gain | 0.013589 |
| mean_current_before_mf1 | 0.485214 |
| mean_current_after_mf1 | 0.505388 |
| mean_current_mf1_gain | 0.020174 |
| final_seen_acc | 0.567869 |
| final_seen_mf1 | 0.477979 |
| bwt_acc | -0.022280 |
| bwt_mf1 | -0.027410 |
| mean_pseudo_acc_diagnostic_only | 0.583537 |
| mean_pseudo_mf1_diagnostic_only | 0.534620 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7012 | 0.7631 | 0.7328 | 0.6558 | 0.6612 |
| 2 | 89 | 0.7000 | 0.6960 | 0.6409 | 0.6457 | 0.6590 | 0.6200 |
| 3 | 1 | 0.1849 | 0.1988 | 0.0736 | 0.0703 | 0.3093 | 0.2625 |
| 4 | 27 | 0.8284 | 0.8659 | 0.7555 | 0.8023 | 0.8636 | 0.8187 |
| 5 | 60 | 0.5989 | 0.7068 | 0.5473 | 0.6488 | 0.7432 | 0.7046 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0966 | 0.3512 | 0.1786 |
| 7 | 52 | 0.7443 | 0.7977 | 0.6671 | 0.7349 | 0.8102 | 0.7673 |
| 8 | 42 | 0.8064 | 0.7603 | 0.6917 | 0.6973 | 0.4295 | 0.4215 |
| 9 | 80 | 0.6570 | 0.6593 | 0.5487 | 0.5584 | 0.7337 | 0.7039 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2798 | 0.2079 |
