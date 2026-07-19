# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.225210 |
| final_old_mf1 | 0.085939 |
| old_aaa | 0.555090 |
| old_aaf1 | 0.514228 |
| old_fr | 0.679396 |
| mean_current_before_acc | 0.501555 |
| mean_current_after_acc | 0.609252 |
| mean_current_acc_gain | 0.107697 |
| mean_current_before_mf1 | 0.394358 |
| mean_current_after_mf1 | 0.550589 |
| mean_current_mf1_gain | 0.156230 |
| final_seen_acc | 0.303483 |
| final_seen_mf1 | 0.141012 |
| bwt_acc | -0.305770 |
| bwt_mf1 | -0.409577 |
| mean_pseudo_acc_diagnostic_only | 0.604399 |
| mean_pseudo_mf1_diagnostic_only | 0.551047 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6650 | 0.6345 | 0.6244 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3035 | 0.0769 | 0.2499 | 0.3209 | 0.2720 |
| 4 | 27 | 0.3886 | 0.8545 | 0.2225 | 0.8220 | 0.8511 | 0.8193 |
| 5 | 60 | 0.6205 | 0.7341 | 0.5761 | 0.6810 | 0.7375 | 0.6896 |
| 6 | 5 | 0.4071 | 0.3298 | 0.2091 | 0.1429 | 0.3405 | 0.1644 |
| 7 | 52 | 0.4239 | 0.8159 | 0.2532 | 0.7800 | 0.8102 | 0.7737 |
| 8 | 42 | 0.8000 | 0.6833 | 0.6225 | 0.5875 | 0.6077 | 0.5385 |
| 9 | 80 | 0.5895 | 0.7279 | 0.5165 | 0.6925 | 0.7291 | 0.6942 |
| 10 | 26 | 0.1962 | 0.2750 | 0.0690 | 0.1994 | 0.2760 | 0.2033 |
