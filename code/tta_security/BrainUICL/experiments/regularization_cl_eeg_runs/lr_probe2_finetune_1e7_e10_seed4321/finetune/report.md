# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.683772 |
| final_old_mf1 | 0.667554 |
| old_aaa | 0.689780 |
| old_aaf1 | 0.672533 |
| old_fr | 0.026596 |
| mean_current_before_acc | 0.697535 |
| mean_current_after_acc | 0.675488 |
| mean_current_acc_gain | -0.022047 |
| mean_current_before_mf1 | 0.697468 |
| mean_current_after_mf1 | 0.667455 |
| mean_current_mf1_gain | -0.030013 |
| final_seen_acc | 0.678977 |
| final_seen_mf1 | 0.673220 |
| bwt_acc | 0.003488 |
| bwt_mf1 | 0.005765 |
| mean_pseudo_acc_diagnostic_only | 0.681000 |
| mean_pseudo_mf1_diagnostic_only | 0.673322 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7070 | 0.7631 | 0.7294 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6660 | 0.6440 | 0.6318 | 0.6055 | 0.6620 | 0.6241 |
