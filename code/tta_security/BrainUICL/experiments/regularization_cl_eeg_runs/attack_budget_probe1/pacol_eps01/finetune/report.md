# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.704072 |
| final_old_mf1 | 0.688915 |
| old_aaa | 0.703263 |
| old_aaf1 | 0.688474 |
| old_fr | 0.002302 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.719767 |
| mean_current_acc_gain | -0.009302 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.754474 |
| mean_current_mf1_gain | -0.008643 |
| final_seen_acc | 0.719767 |
| final_seen_mf1 | 0.754474 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.716279 |
| mean_pseudo_mf1_diagnostic_only | 0.749741 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7198 | 0.7631 | 0.7545 | 0.7163 | 0.7497 |
