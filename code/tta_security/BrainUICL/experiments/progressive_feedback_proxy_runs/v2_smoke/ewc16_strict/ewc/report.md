# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.710419 |
| final_old_mf1 | 0.695231 |
| old_aaa | 0.707365 |
| old_aaf1 | 0.693052 |
| old_fr | 0.011337 |
| mean_current_before_acc | 0.620962 |
| mean_current_after_acc | 0.621088 |
| mean_current_acc_gain | 0.000125 |
| mean_current_before_mf1 | 0.538231 |
| mean_current_after_mf1 | 0.539831 |
| mean_current_mf1_gain | 0.001599 |
| final_seen_acc | 0.622435 |
| final_seen_mf1 | 0.541728 |
| bwt_acc | 0.001347 |
| bwt_mf1 | 0.001897 |
| mean_pseudo_acc_diagnostic_only | 0.474471 |
| mean_pseudo_mf1_diagnostic_only | 0.324051 |
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
| 9 | 80 | 0.6814 | 0.6767 | 0.6357 | 0.6293 | 0.3990 | 0.1925 |
| 10 | 26 | 0.1942 | 0.1962 | 0.0882 | 0.0959 | 0.1817 | 0.1049 |
| 11 | 91 | 0.8104 | 0.8104 | 0.7748 | 0.7748 | 0.3802 | 0.1802 |
| 12 | 22 | 0.3390 | 0.3366 | 0.1301 | 0.1292 | 0.2878 | 0.1395 |
| 13 | 61 | 0.7561 | 0.7561 | 0.7451 | 0.7451 | 0.3729 | 0.1774 |
| 14 | 85 | 0.6884 | 0.6919 | 0.6163 | 0.6203 | 0.7302 | 0.6705 |
| 15 | 17 | 0.6927 | 0.6939 | 0.6101 | 0.6119 | 0.3698 | 0.1724 |
| 16 | 36 | 0.7302 | 0.7344 | 0.6870 | 0.6904 | 0.7615 | 0.7130 |
