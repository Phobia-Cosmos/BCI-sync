# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.626407 |
| final_old_mf1 | 0.583307 |
| old_aaa | 0.638035 |
| old_aaf1 | 0.598404 |
| old_fr | 0.108260 |
| mean_current_before_acc | 0.496611 |
| mean_current_after_acc | 0.478338 |
| mean_current_acc_gain | -0.018273 |
| mean_current_before_mf1 | 0.409146 |
| mean_current_after_mf1 | 0.390454 |
| mean_current_mf1_gain | -0.018692 |
| final_seen_acc | 0.471145 |
| final_seen_mf1 | 0.382667 |
| bwt_acc | -0.007192 |
| bwt_mf1 | -0.007787 |
| mean_pseudo_acc_diagnostic_only | 0.392894 |
| mean_pseudo_mf1_diagnostic_only | 0.335246 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5698 | 0.7631 | 0.6025 | 0.3593 | 0.3468 |
| 2 | 89 | 0.5000 | 0.4720 | 0.4515 | 0.4167 | 0.4060 | 0.3720 |
| 3 | 1 | 0.1826 | 0.1826 | 0.0802 | 0.0779 | 0.3047 | 0.2832 |
| 4 | 27 | 0.7125 | 0.7239 | 0.6206 | 0.6309 | 0.5068 | 0.4369 |
| 5 | 60 | 0.6125 | 0.6114 | 0.5080 | 0.5087 | 0.4830 | 0.4195 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0977 | 0.0976 | 0.3583 | 0.2081 |
| 7 | 52 | 0.7182 | 0.7205 | 0.6012 | 0.6035 | 0.4705 | 0.4148 |
| 8 | 42 | 0.3923 | 0.3821 | 0.4375 | 0.4343 | 0.3615 | 0.3125 |
| 9 | 80 | 0.6023 | 0.6047 | 0.4648 | 0.4659 | 0.4058 | 0.3519 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2731 | 0.2069 |
