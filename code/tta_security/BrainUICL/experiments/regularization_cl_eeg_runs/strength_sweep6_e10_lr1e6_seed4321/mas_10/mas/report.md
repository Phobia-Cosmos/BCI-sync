# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.319880 |
| final_old_mf1 | 0.242027 |
| old_aaa | 0.563507 |
| old_aaf1 | 0.527713 |
| old_fr | 0.544625 |
| mean_current_before_acc | 0.501604 |
| mean_current_after_acc | 0.598414 |
| mean_current_acc_gain | 0.096810 |
| mean_current_before_mf1 | 0.414642 |
| mean_current_after_mf1 | 0.540707 |
| mean_current_mf1_gain | 0.126065 |
| final_seen_acc | 0.379682 |
| final_seen_mf1 | 0.206527 |
| bwt_acc | -0.218732 |
| bwt_mf1 | -0.334180 |
| mean_pseudo_acc_diagnostic_only | 0.603692 |
| mean_pseudo_mf1_diagnostic_only | 0.549891 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6640 | 0.6345 | 0.6234 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3023 | 0.0769 | 0.2484 | 0.3198 | 0.2706 |
| 4 | 27 | 0.3886 | 0.8534 | 0.2225 | 0.8192 | 0.8511 | 0.8180 |
| 5 | 60 | 0.6193 | 0.7375 | 0.5753 | 0.6845 | 0.7398 | 0.6912 |
| 6 | 5 | 0.4119 | 0.3298 | 0.2155 | 0.1424 | 0.3405 | 0.1641 |
