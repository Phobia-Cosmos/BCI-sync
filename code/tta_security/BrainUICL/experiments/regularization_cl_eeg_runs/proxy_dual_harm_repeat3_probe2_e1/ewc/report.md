# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.629701 |
| final_old_mf1 | 0.589904 |
| old_aaa | 0.652914 |
| old_aaf1 | 0.623399 |
| old_fr | 0.103572 |
| mean_current_before_acc | 0.616535 |
| mean_current_after_acc | 0.536093 |
| mean_current_acc_gain | -0.080442 |
| mean_current_before_mf1 | 0.615358 |
| mean_current_after_mf1 | 0.524498 |
| mean_current_mf1_gain | -0.090860 |
| final_seen_acc | 0.522140 |
| final_seen_mf1 | 0.512770 |
| bwt_acc | -0.013953 |
| bwt_mf1 | -0.011728 |
| mean_pseudo_acc_diagnostic_only | 0.265500 |
| mean_pseudo_mf1_diagnostic_only | 0.258599 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5942 | 0.7631 | 0.6222 | 0.3000 | 0.3037 |
| 2 | 89 | 0.5040 | 0.4780 | 0.4676 | 0.4268 | 0.2310 | 0.2135 |
