# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `t2t`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.718750 |
| final_old_mf1 | 0.698248 |
| old_aaa | 0.723750 |
| old_aaf1 | 0.704300 |
| old_fr | 0.033613 |
| mean_current_before_acc | 0.624219 |
| mean_current_after_acc | 0.626563 |
| mean_current_acc_gain | 0.002344 |
| mean_current_before_mf1 | 0.466408 |
| mean_current_after_mf1 | 0.477352 |
| mean_current_mf1_gain | 0.010944 |
| final_seen_acc | 0.616406 |
| final_seen_mf1 | 0.469361 |
| bwt_acc | -0.010156 |
| bwt_mf1 | -0.007991 |
| mean_pseudo_acc_diagnostic_only | 0.630909 |
| mean_pseudo_mf1_diagnostic_only | 0.589256 |
| pseudo_label_coverage | 1.000000 |
| t2t_valid_scores | 2.000000 |
| t2t_detected_pairs | 1.000000 |
| t2t_rejected_updates | 2.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7937 | 0.7781 | 0.6414 | 0.6300 | 0.7302 | 0.7625 |
| 2 | 89 | 0.7812 | 0.7656 | 0.4766 | 0.4795 | 0.7790 | 0.7135 |
| 3 | 1 | 0.1688 | 0.2031 | 0.0655 | 0.0935 | 0.2337 | 0.1647 |
| 4 | 27 | 0.7531 | 0.7594 | 0.6821 | 0.7063 | 0.7807 | 0.7163 |
