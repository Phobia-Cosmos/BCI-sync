# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.320359 |
| final_old_mf1 | 0.241936 |
| old_aaa | 0.563892 |
| old_aaf1 | 0.528079 |
| old_fr | 0.543943 |
| mean_current_before_acc | 0.501009 |
| mean_current_after_acc | 0.598211 |
| mean_current_acc_gain | 0.097202 |
| mean_current_before_mf1 | 0.413353 |
| mean_current_after_mf1 | 0.540623 |
| mean_current_mf1_gain | 0.127271 |
| final_seen_acc | 0.379844 |
| final_seen_mf1 | 0.205890 |
| bwt_acc | -0.218367 |
| bwt_mf1 | -0.334733 |
| mean_pseudo_acc_diagnostic_only | 0.603119 |
| mean_pseudo_mf1_diagnostic_only | 0.549365 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6650 | 0.6345 | 0.6243 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3047 | 0.0769 | 0.2514 | 0.3209 | 0.2720 |
| 4 | 27 | 0.3886 | 0.8523 | 0.2225 | 0.8183 | 0.8511 | 0.8180 |
| 5 | 60 | 0.6193 | 0.7341 | 0.5729 | 0.6810 | 0.7364 | 0.6887 |
| 6 | 5 | 0.4083 | 0.3298 | 0.2102 | 0.1424 | 0.3393 | 0.1620 |
