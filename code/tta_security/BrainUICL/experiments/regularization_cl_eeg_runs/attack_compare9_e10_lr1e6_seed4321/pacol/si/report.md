# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.714192 |
| final_old_mf1 | 0.696608 |
| old_aaa | 0.713521 |
| old_aaf1 | 0.696232 |
| old_fr | 0.016708 |
| mean_current_before_acc | 0.637741 |
| mean_current_after_acc | 0.638008 |
| mean_current_acc_gain | 0.000267 |
| mean_current_before_mf1 | 0.549572 |
| mean_current_after_mf1 | 0.549966 |
| mean_current_mf1_gain | 0.000394 |
| final_seen_acc | 0.635074 |
| final_seen_mf1 | 0.547778 |
| bwt_acc | -0.002934 |
| bwt_mf1 | -0.002188 |
| mean_pseudo_acc_diagnostic_only | 0.645084 |
| mean_pseudo_mf1_diagnostic_only | 0.594642 |
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
| 7 | 52 | 0.8261 | 0.8284 | 0.7662 | 0.7701 | 0.8159 | 0.7751 |
| 8 | 42 | 0.7962 | 0.7885 | 0.6673 | 0.6573 | 0.6423 | 0.5904 |
| 9 | 80 | 0.6791 | 0.6826 | 0.6084 | 0.6157 | 0.7221 | 0.6971 |
