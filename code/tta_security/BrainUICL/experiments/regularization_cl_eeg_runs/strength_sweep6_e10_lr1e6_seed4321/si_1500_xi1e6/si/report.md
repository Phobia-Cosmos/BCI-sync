# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.325090 |
| final_old_mf1 | 0.245063 |
| old_aaa | 0.563858 |
| old_aaf1 | 0.527723 |
| old_fr | 0.537209 |
| mean_current_before_acc | 0.500829 |
| mean_current_after_acc | 0.598504 |
| mean_current_acc_gain | 0.097676 |
| mean_current_before_mf1 | 0.413147 |
| mean_current_after_mf1 | 0.541650 |
| mean_current_mf1_gain | 0.128503 |
| final_seen_acc | 0.386466 |
| final_seen_mf1 | 0.212757 |
| bwt_acc | -0.212039 |
| bwt_mf1 | -0.328893 |
| mean_pseudo_acc_diagnostic_only | 0.604084 |
| mean_pseudo_mf1_diagnostic_only | 0.550308 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6620 | 0.6345 | 0.6216 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3047 | 0.0769 | 0.2514 | 0.3209 | 0.2720 |
| 4 | 27 | 0.3875 | 0.8523 | 0.2173 | 0.8167 | 0.8523 | 0.8181 |
| 5 | 60 | 0.6182 | 0.7341 | 0.5721 | 0.6819 | 0.7386 | 0.6904 |
| 6 | 5 | 0.4095 | 0.3345 | 0.2150 | 0.1520 | 0.3417 | 0.1659 |
