# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.686826 |
| final_old_mf1 | 0.662739 |
| old_aaa | 0.696249 |
| old_aaf1 | 0.676569 |
| old_fr | 0.022249 |
| mean_current_before_acc | 0.583359 |
| mean_current_after_acc | 0.599689 |
| mean_current_acc_gain | 0.016330 |
| mean_current_before_mf1 | 0.492397 |
| mean_current_after_mf1 | 0.517001 |
| mean_current_mf1_gain | 0.024604 |
| final_seen_acc | 0.581182 |
| final_seen_mf1 | 0.493144 |
| bwt_acc | -0.018507 |
| bwt_mf1 | -0.023856 |
| mean_pseudo_acc_diagnostic_only | 0.601155 |
| mean_pseudo_mf1_diagnostic_only | 0.552528 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7279 | 0.7631 | 0.7592 | 0.6895 | 0.7120 |
| 2 | 89 | 0.7500 | 0.6980 | 0.6870 | 0.6487 | 0.6610 | 0.6227 |
| 3 | 1 | 0.1860 | 0.1977 | 0.0765 | 0.0703 | 0.3058 | 0.2573 |
| 4 | 27 | 0.8261 | 0.8670 | 0.7525 | 0.8060 | 0.8625 | 0.8172 |
| 5 | 60 | 0.6045 | 0.6989 | 0.5497 | 0.6402 | 0.7420 | 0.7045 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0962 | 0.3548 | 0.1842 |
| 7 | 52 | 0.7568 | 0.8023 | 0.6841 | 0.7421 | 0.8091 | 0.7678 |
| 8 | 42 | 0.8038 | 0.8013 | 0.6865 | 0.7303 | 0.5731 | 0.5462 |
| 9 | 80 | 0.6605 | 0.6884 | 0.5604 | 0.6104 | 0.7349 | 0.7072 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2788 | 0.2063 |
