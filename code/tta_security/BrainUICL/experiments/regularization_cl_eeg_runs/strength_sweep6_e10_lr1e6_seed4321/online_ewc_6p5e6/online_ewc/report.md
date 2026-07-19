# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.329162 |
| final_old_mf1 | 0.246293 |
| old_aaa | 0.563388 |
| old_aaf1 | 0.527387 |
| old_fr | 0.531412 |
| mean_current_before_acc | 0.502831 |
| mean_current_after_acc | 0.605446 |
| mean_current_acc_gain | 0.102615 |
| mean_current_before_mf1 | 0.415283 |
| mean_current_after_mf1 | 0.550843 |
| mean_current_mf1_gain | 0.135560 |
| final_seen_acc | 0.392908 |
| final_seen_mf1 | 0.219762 |
| bwt_acc | -0.212538 |
| bwt_mf1 | -0.331081 |
| mean_pseudo_acc_diagnostic_only | 0.608251 |
| mean_pseudo_mf1_diagnostic_only | 0.554256 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6450 | 0.6345 | 0.6071 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3209 | 0.0769 | 0.2729 | 0.3186 | 0.2694 |
| 4 | 27 | 0.3875 | 0.8739 | 0.2173 | 0.8344 | 0.8670 | 0.8303 |
| 5 | 60 | 0.6159 | 0.7477 | 0.5777 | 0.6981 | 0.7500 | 0.7030 |
| 6 | 5 | 0.4238 | 0.3417 | 0.2222 | 0.1662 | 0.3429 | 0.1674 |
