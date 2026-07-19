# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.712515 |
| final_old_mf1 | 0.694649 |
| old_aaa | 0.712002 |
| old_aaf1 | 0.694893 |
| old_fr | 0.014321 |
| mean_current_before_acc | 0.571478 |
| mean_current_after_acc | 0.569679 |
| mean_current_acc_gain | -0.001799 |
| mean_current_before_mf1 | 0.487304 |
| mean_current_after_mf1 | 0.484207 |
| mean_current_mf1_gain | -0.003097 |
| final_seen_acc | 0.569542 |
| final_seen_mf1 | 0.484943 |
| bwt_acc | -0.000136 |
| bwt_mf1 | 0.000735 |
| mean_pseudo_acc_diagnostic_only | 0.603479 |
| mean_pseudo_mf1_diagnostic_only | 0.547145 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7410 | 0.6939 | 0.6840 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1988 | 0.1977 | 0.0915 | 0.0888 | 0.3000 | 0.2483 |
| 4 | 27 | 0.7932 | 0.7932 | 0.7049 | 0.7048 | 0.8659 | 0.8209 |
| 5 | 60 | 0.6386 | 0.6523 | 0.5579 | 0.5711 | 0.7398 | 0.6898 |
| 6 | 5 | 0.3071 | 0.3095 | 0.1125 | 0.1050 | 0.3512 | 0.1753 |
