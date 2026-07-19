# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.228263 |
| final_old_mf1 | 0.086803 |
| old_aaa | 0.554954 |
| old_aaf1 | 0.514460 |
| old_fr | 0.675049 |
| mean_current_before_acc | 0.499193 |
| mean_current_after_acc | 0.618286 |
| mean_current_acc_gain | 0.119093 |
| mean_current_before_mf1 | 0.392101 |
| mean_current_after_mf1 | 0.561071 |
| mean_current_mf1_gain | 0.168970 |
| final_seen_acc | 0.315471 |
| final_seen_mf1 | 0.153758 |
| bwt_acc | -0.302815 |
| bwt_mf1 | -0.407314 |
| mean_pseudo_acc_diagnostic_only | 0.610033 |
| mean_pseudo_mf1_diagnostic_only | 0.557614 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6620 | 0.6345 | 0.6218 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3128 | 0.0769 | 0.2612 | 0.3209 | 0.2719 |
| 4 | 27 | 0.3886 | 0.8614 | 0.2187 | 0.8254 | 0.8511 | 0.8175 |
| 5 | 60 | 0.6125 | 0.7409 | 0.5671 | 0.6910 | 0.7432 | 0.6936 |
| 6 | 5 | 0.4143 | 0.3345 | 0.2182 | 0.1528 | 0.3429 | 0.1677 |
| 7 | 52 | 0.4261 | 0.8250 | 0.2543 | 0.7887 | 0.8182 | 0.7841 |
| 8 | 42 | 0.7897 | 0.7128 | 0.6062 | 0.6165 | 0.6269 | 0.5609 |
| 9 | 80 | 0.5872 | 0.7453 | 0.5085 | 0.7156 | 0.7453 | 0.7170 |
| 10 | 26 | 0.1837 | 0.2846 | 0.0735 | 0.2115 | 0.2808 | 0.2079 |
