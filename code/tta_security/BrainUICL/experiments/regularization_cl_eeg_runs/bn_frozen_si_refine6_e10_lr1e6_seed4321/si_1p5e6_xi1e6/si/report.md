# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.716707 |
| final_old_mf1 | 0.699582 |
| old_aaa | 0.713088 |
| old_aaf1 | 0.695906 |
| old_fr | 0.020288 |
| mean_current_before_acc | 0.573052 |
| mean_current_after_acc | 0.573774 |
| mean_current_acc_gain | 0.000722 |
| mean_current_before_mf1 | 0.484047 |
| mean_current_after_mf1 | 0.484428 |
| mean_current_mf1_gain | 0.000381 |
| final_seen_acc | 0.577647 |
| final_seen_mf1 | 0.488821 |
| bwt_acc | 0.003874 |
| bwt_mf1 | 0.004393 |
| mean_pseudo_acc_diagnostic_only | 0.604241 |
| mean_pseudo_mf1_diagnostic_only | 0.548196 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7380 | 0.6939 | 0.6828 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1953 | 0.1965 | 0.0835 | 0.0744 | 0.2988 | 0.2469 |
| 4 | 27 | 0.8000 | 0.8148 | 0.7092 | 0.7272 | 0.8693 | 0.8272 |
| 5 | 60 | 0.6352 | 0.6523 | 0.5567 | 0.5728 | 0.7409 | 0.6888 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0980 | 0.0979 | 0.3524 | 0.1776 |
