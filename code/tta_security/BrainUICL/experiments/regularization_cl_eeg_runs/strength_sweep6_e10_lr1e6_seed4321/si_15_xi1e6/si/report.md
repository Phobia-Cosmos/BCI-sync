# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.320719 |
| final_old_mf1 | 0.242181 |
| old_aaa | 0.563858 |
| old_aaf1 | 0.527857 |
| old_fr | 0.543432 |
| mean_current_before_acc | 0.500991 |
| mean_current_after_acc | 0.598206 |
| mean_current_acc_gain | 0.097215 |
| mean_current_before_mf1 | 0.413786 |
| mean_current_after_mf1 | 0.540950 |
| mean_current_mf1_gain | 0.127164 |
| final_seen_acc | 0.379844 |
| final_seen_mf1 | 0.205869 |
| bwt_acc | -0.218362 |
| bwt_mf1 | -0.335080 |
| mean_pseudo_acc_diagnostic_only | 0.603701 |
| mean_pseudo_mf1_diagnostic_only | 0.549937 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6650 | 0.6345 | 0.6244 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3035 | 0.0769 | 0.2499 | 0.3221 | 0.2732 |
| 4 | 27 | 0.3886 | 0.8534 | 0.2225 | 0.8211 | 0.8511 | 0.8177 |
| 5 | 60 | 0.6216 | 0.7341 | 0.5770 | 0.6810 | 0.7375 | 0.6895 |
| 6 | 5 | 0.4060 | 0.3298 | 0.2087 | 0.1429 | 0.3405 | 0.1638 |
