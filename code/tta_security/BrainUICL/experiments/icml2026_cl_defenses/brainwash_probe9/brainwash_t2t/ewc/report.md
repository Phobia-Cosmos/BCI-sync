# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `t2t`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.710060 |
| final_old_mf1 | 0.698868 |
| old_aaa | 0.709551 |
| old_aaf1 | 0.695373 |
| old_fr | 0.010826 |
| mean_current_before_acc | 0.632996 |
| mean_current_after_acc | 0.645274 |
| mean_current_acc_gain | 0.012278 |
| mean_current_before_mf1 | 0.549553 |
| mean_current_after_mf1 | 0.568662 |
| mean_current_mf1_gain | 0.019109 |
| final_seen_acc | 0.629820 |
| final_seen_mf1 | 0.555241 |
| bwt_acc | -0.015454 |
| bwt_mf1 | -0.013421 |
| mean_pseudo_acc_diagnostic_only | 0.645319 |
| mean_pseudo_mf1_diagnostic_only | 0.598048 |
| pseudo_label_coverage | 1.000000 |
| t2t_valid_scores | 6.000000 |
| t2t_detected_pairs | 1.000000 |
| t2t_rejected_updates | 2.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7150 | 0.6939 | 0.6659 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1884 | 0.1942 | 0.0841 | 0.0708 | 0.3058 | 0.2557 |
| 4 | 27 | 0.8159 | 0.8568 | 0.7464 | 0.7927 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6080 | 0.6943 | 0.5453 | 0.6331 | 0.7375 | 0.6979 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0986 | 0.0985 | 0.3524 | 0.1784 |
| 7 | 52 | 0.8136 | 0.8330 | 0.7501 | 0.7807 | 0.8159 | 0.7726 |
| 8 | 42 | 0.7936 | 0.7859 | 0.6818 | 0.7026 | 0.6372 | 0.6026 |
| 9 | 80 | 0.6698 | 0.6872 | 0.5827 | 0.6221 | 0.7337 | 0.7103 |
