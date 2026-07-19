# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.713174 |
| final_old_mf1 | 0.693812 |
| old_aaa | 0.712284 |
| old_aaf1 | 0.693793 |
| old_fr | 0.015259 |
| mean_current_before_acc | 0.574390 |
| mean_current_after_acc | 0.572538 |
| mean_current_acc_gain | -0.001852 |
| mean_current_before_mf1 | 0.488021 |
| mean_current_after_mf1 | 0.486829 |
| mean_current_mf1_gain | -0.001192 |
| final_seen_acc | 0.572335 |
| final_seen_mf1 | 0.486290 |
| bwt_acc | -0.000203 |
| bwt_mf1 | -0.000539 |
| mean_pseudo_acc_diagnostic_only | 0.604993 |
| mean_pseudo_mf1_diagnostic_only | 0.548732 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7590 | 0.6939 | 0.7002 | 0.6640 | 0.6261 |
| 3 | 1 | 0.2105 | 0.2081 | 0.1050 | 0.1043 | 0.3047 | 0.2521 |
| 4 | 27 | 0.7920 | 0.7886 | 0.6968 | 0.6942 | 0.8716 | 0.8292 |
| 5 | 60 | 0.6420 | 0.6443 | 0.5527 | 0.5543 | 0.7409 | 0.6896 |
| 6 | 5 | 0.3107 | 0.3107 | 0.1166 | 0.1166 | 0.3488 | 0.1729 |
