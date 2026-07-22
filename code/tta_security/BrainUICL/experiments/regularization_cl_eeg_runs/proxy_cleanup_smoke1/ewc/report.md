# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.707545 |
| final_old_mf1 | 0.693761 |
| old_aaa | 0.705000 |
| old_aaf1 | 0.690897 |
| old_fr | 0.007246 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.738372 |
| mean_current_acc_gain | 0.009302 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.771123 |
| mean_current_mf1_gain | 0.008006 |
| final_seen_acc | 0.738372 |
| final_seen_mf1 | 0.771123 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.665217 |
| mean_pseudo_mf1_diagnostic_only | 0.701175 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7384 | 0.7631 | 0.7711 | 0.6652 | 0.7012 |
