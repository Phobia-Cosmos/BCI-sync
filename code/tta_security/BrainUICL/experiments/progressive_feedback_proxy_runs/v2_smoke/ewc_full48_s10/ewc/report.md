# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.705329 |
| final_old_mf1 | 0.691324 |
| old_aaa | 0.702535 |
| old_aaf1 | 0.688292 |
| old_fr | 0.004092 |
| mean_current_before_acc | 0.758535 |
| mean_current_after_acc | 0.755209 |
| mean_current_acc_gain | -0.003326 |
| mean_current_before_mf1 | 0.742754 |
| mean_current_after_mf1 | 0.740249 |
| mean_current_mf1_gain | -0.002504 |
| final_seen_acc | 0.757535 |
| final_seen_mf1 | 0.742271 |
| bwt_acc | 0.002326 |
| bwt_mf1 | 0.002021 |
| mean_pseudo_acc_diagnostic_only | 0.574771 |
| mean_pseudo_mf1_diagnostic_only | 0.421168 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7598 | 0.3885 | 0.1430 |
| 2 | 89 | 0.7880 | 0.7860 | 0.7224 | 0.7207 | 0.7610 | 0.6993 |
