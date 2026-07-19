# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.713293 |
| final_old_mf1 | 0.695845 |
| old_aaa | 0.712275 |
| old_aaf1 | 0.695205 |
| old_fr | 0.015429 |
| mean_current_before_acc | 0.571284 |
| mean_current_after_acc | 0.569561 |
| mean_current_acc_gain | -0.001723 |
| mean_current_before_mf1 | 0.486084 |
| mean_current_after_mf1 | 0.481288 |
| mean_current_mf1_gain | -0.004796 |
| final_seen_acc | 0.568241 |
| final_seen_mf1 | 0.480876 |
| bwt_acc | -0.001321 |
| bwt_mf1 | -0.000412 |
| mean_pseudo_acc_diagnostic_only | 0.603677 |
| mean_pseudo_mf1_diagnostic_only | 0.547692 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7400 | 0.6939 | 0.6834 | 0.6640 | 0.6261 |
| 3 | 1 | 0.2000 | 0.1988 | 0.0929 | 0.0838 | 0.3023 | 0.2517 |
| 4 | 27 | 0.7943 | 0.7966 | 0.7026 | 0.7040 | 0.8648 | 0.8205 |
| 5 | 60 | 0.6364 | 0.6420 | 0.5582 | 0.5625 | 0.7398 | 0.6898 |
| 6 | 5 | 0.3060 | 0.3155 | 0.1058 | 0.1026 | 0.3512 | 0.1755 |
