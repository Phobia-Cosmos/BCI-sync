# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.703054 |
| final_old_mf1 | 0.688033 |
| old_aaa | 0.702754 |
| old_aaf1 | 0.688033 |
| old_fr | 0.000852 |
| mean_current_before_acc | 0.729070 |
| mean_current_after_acc | 0.719767 |
| mean_current_acc_gain | -0.009302 |
| mean_current_before_mf1 | 0.763116 |
| mean_current_after_mf1 | 0.754925 |
| mean_current_mf1_gain | -0.008191 |
| final_seen_acc | 0.719767 |
| final_seen_mf1 | 0.754925 |
| bwt_acc | 0.000000 |
| bwt_mf1 | 0.000000 |
| mean_pseudo_acc_diagnostic_only | 0.256977 |
| mean_pseudo_mf1_diagnostic_only | 0.244827 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7198 | 0.7631 | 0.7549 | 0.2570 | 0.2448 |
