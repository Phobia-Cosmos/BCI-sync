# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.326766 |
| final_old_mf1 | 0.246483 |
| old_aaa | 0.563353 |
| old_aaf1 | 0.527649 |
| old_fr | 0.534822 |
| mean_current_before_acc | 0.502822 |
| mean_current_after_acc | 0.605399 |
| mean_current_acc_gain | 0.102577 |
| mean_current_before_mf1 | 0.414960 |
| mean_current_after_mf1 | 0.550552 |
| mean_current_mf1_gain | 0.135592 |
| final_seen_acc | 0.390280 |
| final_seen_mf1 | 0.217800 |
| bwt_acc | -0.215119 |
| bwt_mf1 | -0.332753 |
| mean_pseudo_acc_diagnostic_only | 0.608066 |
| mean_pseudo_mf1_diagnostic_only | 0.554274 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6550 | 0.6345 | 0.6160 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3209 | 0.0769 | 0.2729 | 0.3198 | 0.2706 |
| 4 | 27 | 0.3898 | 0.8670 | 0.2200 | 0.8297 | 0.8648 | 0.8293 |
| 5 | 60 | 0.6148 | 0.7455 | 0.5750 | 0.6947 | 0.7500 | 0.7042 |
| 6 | 5 | 0.4226 | 0.3405 | 0.2201 | 0.1637 | 0.3429 | 0.1661 |
