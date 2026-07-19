# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.225150 |
| final_old_mf1 | 0.085777 |
| old_aaa | 0.554976 |
| old_aaf1 | 0.514179 |
| old_fr | 0.679482 |
| mean_current_before_acc | 0.501108 |
| mean_current_after_acc | 0.609125 |
| mean_current_acc_gain | 0.108018 |
| mean_current_before_mf1 | 0.393746 |
| mean_current_after_mf1 | 0.550438 |
| mean_current_mf1_gain | 0.156693 |
| final_seen_acc | 0.302544 |
| final_seen_mf1 | 0.140133 |
| bwt_acc | -0.306581 |
| bwt_mf1 | -0.410305 |
| mean_pseudo_acc_diagnostic_only | 0.604898 |
| mean_pseudo_mf1_diagnostic_only | 0.551610 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6660 | 0.6345 | 0.6253 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3023 | 0.0769 | 0.2484 | 0.3198 | 0.2706 |
| 4 | 27 | 0.3886 | 0.8523 | 0.2225 | 0.8183 | 0.8500 | 0.8171 |
| 5 | 60 | 0.6205 | 0.7341 | 0.5761 | 0.6810 | 0.7375 | 0.6896 |
| 6 | 5 | 0.4060 | 0.3298 | 0.2090 | 0.1437 | 0.3405 | 0.1643 |
| 7 | 52 | 0.4239 | 0.8159 | 0.2532 | 0.7800 | 0.8125 | 0.7772 |
| 8 | 42 | 0.8000 | 0.6833 | 0.6207 | 0.5866 | 0.6115 | 0.5419 |
| 9 | 80 | 0.5872 | 0.7291 | 0.5126 | 0.6954 | 0.7302 | 0.6968 |
| 10 | 26 | 0.1952 | 0.2750 | 0.0688 | 0.1994 | 0.2760 | 0.2033 |
