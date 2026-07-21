# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.689341 |
| final_old_mf1 | 0.665168 |
| old_aaa | 0.697997 |
| old_aaf1 | 0.678878 |
| old_fr | 0.018668 |
| mean_current_before_acc | 0.585085 |
| mean_current_after_acc | 0.597002 |
| mean_current_acc_gain | 0.011918 |
| mean_current_before_mf1 | 0.494523 |
| mean_current_after_mf1 | 0.513136 |
| mean_current_mf1_gain | 0.018613 |
| final_seen_acc | 0.575820 |
| final_seen_mf1 | 0.488548 |
| bwt_acc | -0.021182 |
| bwt_mf1 | -0.024588 |
| mean_pseudo_acc_diagnostic_only | 0.592476 |
| mean_pseudo_mf1_diagnostic_only | 0.546813 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7221 | 0.7631 | 0.7507 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7610 | 0.7050 | 0.6972 | 0.6556 | 0.6650 | 0.6268 |
| 3 | 1 | 0.1872 | 0.1965 | 0.0769 | 0.0703 | 0.3081 | 0.2623 |
| 4 | 27 | 0.8227 | 0.8636 | 0.7505 | 0.8010 | 0.8625 | 0.8172 |
| 5 | 60 | 0.6011 | 0.6966 | 0.5481 | 0.6382 | 0.7420 | 0.7046 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0961 | 0.3548 | 0.1868 |
| 7 | 52 | 0.7602 | 0.8080 | 0.6895 | 0.7498 | 0.8102 | 0.7708 |
| 8 | 42 | 0.8077 | 0.7872 | 0.6948 | 0.7157 | 0.4718 | 0.4667 |
| 9 | 80 | 0.6651 | 0.6756 | 0.5608 | 0.5873 | 0.7372 | 0.7111 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2731 | 0.1994 |
