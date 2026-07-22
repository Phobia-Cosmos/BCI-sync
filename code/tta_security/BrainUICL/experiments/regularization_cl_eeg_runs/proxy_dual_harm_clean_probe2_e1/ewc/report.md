# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.710180 |
| final_old_mf1 | 0.693162 |
| old_aaa | 0.705389 |
| old_aaf1 | 0.688653 |
| old_fr | 0.010997 |
| mean_current_before_acc | 0.753535 |
| mean_current_after_acc | 0.731581 |
| mean_current_acc_gain | -0.021953 |
| mean_current_before_mf1 | 0.737758 |
| mean_current_after_mf1 | 0.718059 |
| mean_current_mf1_gain | -0.019699 |
| final_seen_acc | 0.738558 |
| final_seen_mf1 | 0.726361 |
| bwt_acc | 0.006977 |
| bwt_mf1 | 0.008303 |
| mean_pseudo_acc_diagnostic_only | 0.713244 |
| mean_pseudo_mf1_diagnostic_only | 0.701701 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7012 | 0.7631 | 0.7336 | 0.7035 | 0.7325 |
| 2 | 89 | 0.7780 | 0.7620 | 0.7124 | 0.7025 | 0.7230 | 0.6709 |
