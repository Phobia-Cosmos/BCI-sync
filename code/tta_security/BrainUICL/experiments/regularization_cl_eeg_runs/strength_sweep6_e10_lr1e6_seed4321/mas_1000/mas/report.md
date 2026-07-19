# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.330120 |
| final_old_mf1 | 0.247429 |
| old_aaa | 0.563935 |
| old_aaf1 | 0.527708 |
| old_fr | 0.530049 |
| mean_current_before_acc | 0.502606 |
| mean_current_after_acc | 0.604777 |
| mean_current_acc_gain | 0.102171 |
| mean_current_before_mf1 | 0.415044 |
| mean_current_after_mf1 | 0.549664 |
| mean_current_mf1_gain | 0.134620 |
| final_seen_acc | 0.392105 |
| final_seen_mf1 | 0.218361 |
| bwt_acc | -0.212671 |
| bwt_mf1 | -0.331304 |
| mean_pseudo_acc_diagnostic_only | 0.605974 |
| mean_pseudo_mf1_diagnostic_only | 0.552280 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6570 | 0.6345 | 0.6165 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3186 | 0.0768 | 0.2692 | 0.3198 | 0.2708 |
| 4 | 27 | 0.3886 | 0.8648 | 0.2187 | 0.8262 | 0.8591 | 0.8243 |
| 5 | 60 | 0.6182 | 0.7443 | 0.5775 | 0.6961 | 0.7443 | 0.6970 |
| 6 | 5 | 0.4190 | 0.3405 | 0.2196 | 0.1637 | 0.3417 | 0.1662 |
