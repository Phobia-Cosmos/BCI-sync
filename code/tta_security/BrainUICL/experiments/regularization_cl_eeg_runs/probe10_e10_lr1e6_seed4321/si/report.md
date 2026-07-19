# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.225449 |
| final_old_mf1 | 0.085861 |
| old_aaa | 0.555090 |
| old_aaf1 | 0.514159 |
| old_fr | 0.679055 |
| mean_current_before_acc | 0.501383 |
| mean_current_after_acc | 0.609252 |
| mean_current_acc_gain | 0.107869 |
| mean_current_before_mf1 | 0.393823 |
| mean_current_after_mf1 | 0.550453 |
| mean_current_mf1_gain | 0.156630 |
| final_seen_acc | 0.303253 |
| final_seen_mf1 | 0.140785 |
| bwt_acc | -0.306000 |
| bwt_mf1 | -0.409668 |
| mean_pseudo_acc_diagnostic_only | 0.604973 |
| mean_pseudo_mf1_diagnostic_only | 0.551630 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6650 | 0.6345 | 0.6243 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3035 | 0.0769 | 0.2498 | 0.3209 | 0.2720 |
| 4 | 27 | 0.3886 | 0.8523 | 0.2225 | 0.8183 | 0.8500 | 0.8171 |
| 5 | 60 | 0.6216 | 0.7341 | 0.5747 | 0.6810 | 0.7386 | 0.6902 |
| 6 | 5 | 0.4083 | 0.3298 | 0.2101 | 0.1429 | 0.3405 | 0.1637 |
| 7 | 52 | 0.4239 | 0.8182 | 0.2532 | 0.7821 | 0.8136 | 0.7781 |
| 8 | 42 | 0.8000 | 0.6833 | 0.6207 | 0.5866 | 0.6077 | 0.5397 |
| 9 | 80 | 0.5884 | 0.7279 | 0.5142 | 0.6925 | 0.7314 | 0.6967 |
| 10 | 26 | 0.1933 | 0.2750 | 0.0683 | 0.2007 | 0.2760 | 0.2033 |
