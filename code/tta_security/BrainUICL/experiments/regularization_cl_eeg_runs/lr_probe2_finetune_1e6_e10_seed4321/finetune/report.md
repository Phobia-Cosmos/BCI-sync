# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.686407 |
| final_old_mf1 | 0.670216 |
| old_aaa | 0.691617 |
| old_aaf1 | 0.674056 |
| old_fr | 0.022845 |
| mean_current_before_acc | 0.699535 |
| mean_current_after_acc | 0.684744 |
| mean_current_acc_gain | -0.014791 |
| mean_current_before_mf1 | 0.698803 |
| mean_current_after_mf1 | 0.675832 |
| mean_current_mf1_gain | -0.022971 |
| final_seen_acc | 0.689395 |
| final_seen_mf1 | 0.683365 |
| bwt_acc | 0.004651 |
| bwt_mf1 | 0.007533 |
| mean_pseudo_acc_diagnostic_only | 0.685500 |
| mean_pseudo_mf1_diagnostic_only | 0.677729 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6660 | 0.6345 | 0.6253 | 0.6710 | 0.6329 |
