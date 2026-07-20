# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.584910 |
| final_old_mf1 | 0.511336 |
| old_aaa | 0.601464 |
| old_aaf1 | 0.540500 |
| old_fr | 0.167334 |
| mean_current_before_acc | 0.463453 |
| mean_current_after_acc | 0.463988 |
| mean_current_acc_gain | 0.000535 |
| mean_current_before_mf1 | 0.377641 |
| mean_current_after_mf1 | 0.367476 |
| mean_current_mf1_gain | -0.010165 |
| final_seen_acc | 0.425872 |
| final_seen_mf1 | 0.318606 |
| bwt_acc | -0.038116 |
| bwt_mf1 | -0.048870 |
| mean_pseudo_acc_diagnostic_only | 0.394682 |
| mean_pseudo_mf1_diagnostic_only | 0.337398 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5698 | 0.7631 | 0.6025 | 0.3593 | 0.3468 |
| 2 | 89 | 0.5000 | 0.4440 | 0.4515 | 0.3734 | 0.4060 | 0.3720 |
| 3 | 1 | 0.1360 | 0.1907 | 0.0793 | 0.0829 | 0.3081 | 0.2868 |
| 4 | 27 | 0.6943 | 0.7511 | 0.6102 | 0.6433 | 0.5125 | 0.4421 |
| 5 | 60 | 0.4716 | 0.6205 | 0.4087 | 0.5195 | 0.4852 | 0.4188 |
| 6 | 5 | 0.3155 | 0.3155 | 0.0961 | 0.0989 | 0.3536 | 0.2026 |
| 7 | 52 | 0.6602 | 0.6875 | 0.4936 | 0.5367 | 0.4614 | 0.4032 |
| 8 | 42 | 0.3359 | 0.3167 | 0.3722 | 0.3465 | 0.3577 | 0.3123 |
| 9 | 80 | 0.5977 | 0.5442 | 0.4347 | 0.4044 | 0.4174 | 0.3673 |
| 10 | 26 | 0.1942 | 0.2000 | 0.0669 | 0.0667 | 0.2856 | 0.2220 |
