# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.530778 |
| final_old_mf1 | 0.516125 |
| old_aaa | 0.656424 |
| old_aaf1 | 0.638204 |
| old_fr | 0.244395 |
| mean_current_before_acc | 0.574776 |
| mean_current_after_acc | 0.595631 |
| mean_current_acc_gain | 0.020855 |
| mean_current_before_mf1 | 0.496013 |
| mean_current_after_mf1 | 0.520788 |
| mean_current_mf1_gain | 0.024775 |
| final_seen_acc | 0.506023 |
| final_seen_mf1 | 0.431320 |
| bwt_acc | -0.089607 |
| bwt_mf1 | -0.089468 |
| mean_pseudo_acc_diagnostic_only | 0.605052 |
| mean_pseudo_mf1_diagnostic_only | 0.550840 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7100 | 0.6939 | 0.6628 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1837 | 0.2012 | 0.0761 | 0.0697 | 0.3105 | 0.2610 |
| 4 | 27 | 0.8386 | 0.8795 | 0.7710 | 0.8360 | 0.8648 | 0.8213 |
| 5 | 60 | 0.6102 | 0.7432 | 0.5631 | 0.7055 | 0.7375 | 0.6966 |
| 6 | 5 | 0.3250 | 0.3155 | 0.1089 | 0.0993 | 0.3536 | 0.1775 |
