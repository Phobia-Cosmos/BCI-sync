# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.713293 |
| final_old_mf1 | 0.700015 |
| old_aaa | 0.706012 |
| old_aaf1 | 0.692177 |
| old_fr | 0.015429 |
| mean_current_before_acc | 0.614980 |
| mean_current_after_acc | 0.617783 |
| mean_current_acc_gain | 0.002802 |
| mean_current_before_mf1 | 0.563311 |
| mean_current_after_mf1 | 0.567853 |
| mean_current_mf1_gain | 0.004542 |
| final_seen_acc | 0.620108 |
| final_seen_mf1 | 0.570312 |
| bwt_acc | 0.002326 |
| bwt_mf1 | 0.002459 |
| mean_pseudo_acc_diagnostic_only | 0.504708 |
| mean_pseudo_mf1_diagnostic_only | 0.420791 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7594 | 0.2070 | 0.1127 |
| 2 | 89 | 0.7910 | 0.7840 | 0.7254 | 0.7190 | 0.7650 | 0.7022 |
| 3 | 1 | 0.2035 | 0.2093 | 0.0989 | 0.1095 | 0.2616 | 0.1433 |
| 4 | 27 | 0.7364 | 0.7534 | 0.6658 | 0.6835 | 0.7852 | 0.7249 |
