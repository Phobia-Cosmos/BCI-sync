# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.663892 |
| final_old_mf1 | 0.639827 |
| old_aaa | 0.693661 |
| old_aaf1 | 0.675844 |
| old_fr | 0.054897 |
| mean_current_before_acc | 0.570929 |
| mean_current_after_acc | 0.590025 |
| mean_current_acc_gain | 0.019096 |
| mean_current_before_mf1 | 0.490297 |
| mean_current_after_mf1 | 0.512084 |
| mean_current_mf1_gain | 0.021787 |
| final_seen_acc | 0.573513 |
| final_seen_mf1 | 0.494587 |
| bwt_acc | -0.016513 |
| bwt_mf1 | -0.017497 |
| mean_pseudo_acc_diagnostic_only | 0.604659 |
| mean_pseudo_mf1_diagnostic_only | 0.550473 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7140 | 0.6939 | 0.6664 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1826 | 0.1953 | 0.0757 | 0.0699 | 0.3093 | 0.2597 |
| 4 | 27 | 0.8295 | 0.8739 | 0.7619 | 0.8246 | 0.8614 | 0.8150 |
| 5 | 60 | 0.6045 | 0.7170 | 0.5479 | 0.6637 | 0.7409 | 0.7026 |
| 6 | 5 | 0.3179 | 0.3155 | 0.0992 | 0.0965 | 0.3524 | 0.1770 |
