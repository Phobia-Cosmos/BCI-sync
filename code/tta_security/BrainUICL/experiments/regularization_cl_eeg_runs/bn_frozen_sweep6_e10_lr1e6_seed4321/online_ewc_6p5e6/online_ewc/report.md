# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.714311 |
| final_old_mf1 | 0.694933 |
| old_aaa | 0.713080 |
| old_aaf1 | 0.694561 |
| old_fr | 0.016878 |
| mean_current_before_acc | 0.574417 |
| mean_current_after_acc | 0.574954 |
| mean_current_acc_gain | 0.000537 |
| mean_current_before_mf1 | 0.485664 |
| mean_current_after_mf1 | 0.484598 |
| mean_current_mf1_gain | -0.001066 |
| final_seen_acc | 0.573638 |
| final_seen_mf1 | 0.482356 |
| bwt_acc | -0.001316 |
| bwt_mf1 | -0.002242 |
| mean_pseudo_acc_diagnostic_only | 0.605399 |
| mean_pseudo_mf1_diagnostic_only | 0.549214 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7630 | 0.6939 | 0.7027 | 0.6640 | 0.6261 |
| 3 | 1 | 0.2105 | 0.2105 | 0.1050 | 0.0994 | 0.3116 | 0.2624 |
| 4 | 27 | 0.7932 | 0.7943 | 0.6939 | 0.6989 | 0.8670 | 0.8225 |
| 5 | 60 | 0.6375 | 0.6420 | 0.5520 | 0.5552 | 0.7409 | 0.6887 |
| 6 | 5 | 0.3143 | 0.3155 | 0.1060 | 0.1000 | 0.3488 | 0.1731 |
