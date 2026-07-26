# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.709581 |
| final_old_mf1 | 0.695610 |
| old_aaa | 0.706660 |
| old_aaf1 | 0.692600 |
| old_fr | 0.010144 |
| mean_current_before_acc | 0.630371 |
| mean_current_after_acc | 0.630158 |
| mean_current_acc_gain | -0.000213 |
| mean_current_before_mf1 | 0.540556 |
| mean_current_after_mf1 | 0.542539 |
| mean_current_mf1_gain | 0.001983 |
| final_seen_acc | 0.629020 |
| final_seen_mf1 | 0.541183 |
| bwt_acc | -0.001138 |
| bwt_mf1 | -0.001356 |
| mean_pseudo_acc_diagnostic_only | 0.513554 |
| mean_pseudo_mf1_diagnostic_only | 0.354307 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7604 | 0.3688 | 0.1470 |
| 2 | 89 | 0.7880 | 0.7830 | 0.7224 | 0.7182 | 0.7610 | 0.6993 |
| 3 | 1 | 0.2070 | 0.2058 | 0.0995 | 0.0995 | 0.3563 | 0.1371 |
| 4 | 27 | 0.7375 | 0.7511 | 0.6690 | 0.6818 | 0.7852 | 0.7249 |
| 5 | 60 | 0.6955 | 0.6966 | 0.6110 | 0.6119 | 0.3781 | 0.1479 |
| 6 | 5 | 0.3048 | 0.2988 | 0.1200 | 0.1203 | 0.2476 | 0.1229 |
| 7 | 52 | 0.7966 | 0.7943 | 0.7534 | 0.7485 | 0.4115 | 0.2093 |
| 8 | 42 | 0.7846 | 0.7872 | 0.5861 | 0.5997 | 0.8000 | 0.6460 |
