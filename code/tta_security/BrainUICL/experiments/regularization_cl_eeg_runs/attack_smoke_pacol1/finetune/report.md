# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.703653 |
| final_old_mf1 | 0.688272 |
| old_aaa | 0.703054 |
| old_aaf1 | 0.688153 |
| old_fr | 0.001705 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.718605 |
| mean_current_acc_gain | -0.010465 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.753515 |
| mean_current_mf1_gain | -0.009602 |
| final_seen_acc | 0.718605 |
| final_seen_mf1 | 0.753515 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.702326 |
| mean_pseudo_mf1_diagnostic_only | 0.727732 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7186 | 0.7631 | 0.7535 | 0.7023 | 0.7277 |
