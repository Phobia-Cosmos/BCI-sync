# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.610719 |
| final_old_mf1 | 0.564798 |
| old_aaa | 0.627921 |
| old_aaf1 | 0.586419 |
| old_fr | 0.130594 |
| mean_current_before_acc | 0.487161 |
| mean_current_after_acc | 0.466038 |
| mean_current_acc_gain | -0.021123 |
| mean_current_before_mf1 | 0.399341 |
| mean_current_after_mf1 | 0.376359 |
| mean_current_mf1_gain | -0.022982 |
| final_seen_acc | 0.451816 |
| final_seen_mf1 | 0.361641 |
| bwt_acc | -0.014222 |
| bwt_mf1 | -0.014718 |
| mean_pseudo_acc_diagnostic_only | 0.393175 |
| mean_pseudo_mf1_diagnostic_only | 0.335589 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5698 | 0.7631 | 0.6025 | 0.3593 | 0.3468 |
| 2 | 89 | 0.5000 | 0.4390 | 0.4515 | 0.3809 | 0.4060 | 0.3720 |
| 3 | 1 | 0.1814 | 0.1802 | 0.0799 | 0.0773 | 0.3035 | 0.2826 |
| 4 | 27 | 0.7091 | 0.7159 | 0.6164 | 0.6203 | 0.5091 | 0.4384 |
| 5 | 60 | 0.5841 | 0.6023 | 0.4799 | 0.4966 | 0.4841 | 0.4195 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0962 | 0.3583 | 0.2078 |
| 7 | 52 | 0.7045 | 0.7023 | 0.5800 | 0.5781 | 0.4705 | 0.4148 |
| 8 | 42 | 0.3526 | 0.3436 | 0.4078 | 0.3958 | 0.3615 | 0.3129 |
| 9 | 80 | 0.5942 | 0.5919 | 0.4505 | 0.4491 | 0.4035 | 0.3501 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2760 | 0.2110 |
