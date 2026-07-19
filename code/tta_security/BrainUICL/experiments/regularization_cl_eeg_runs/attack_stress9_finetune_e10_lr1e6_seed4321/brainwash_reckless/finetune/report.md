# Regularization-only EEG CL: finetune

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.681138 |
| final_old_mf1 | 0.667992 |
| old_aaa | 0.658437 |
| old_aaf1 | 0.639001 |
| old_fr | 0.030347 |
| mean_current_before_acc | 0.607240 |
| mean_current_after_acc | 0.635755 |
| mean_current_acc_gain | 0.028515 |
| mean_current_before_mf1 | 0.522106 |
| mean_current_after_mf1 | 0.566932 |
| mean_current_mf1_gain | 0.044825 |
| final_seen_acc | 0.623388 |
| final_seen_mf1 | 0.550242 |
| bwt_acc | -0.012367 |
| bwt_mf1 | -0.016690 |
| mean_pseudo_acc_diagnostic_only | 0.643531 |
| mean_pseudo_mf1_diagnostic_only | 0.593753 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7100 | 0.6939 | 0.6628 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1837 | 0.2012 | 0.0761 | 0.0696 | 0.3105 | 0.2610 |
| 4 | 27 | 0.8386 | 0.8818 | 0.7710 | 0.8386 | 0.8625 | 0.8172 |
| 5 | 60 | 0.6114 | 0.7432 | 0.5664 | 0.7055 | 0.7375 | 0.6951 |
| 6 | 5 | 0.3250 | 0.3167 | 0.1089 | 0.1046 | 0.3548 | 0.1776 |
| 7 | 52 | 0.5648 | 0.8000 | 0.4940 | 0.7344 | 0.8091 | 0.7640 |
| 8 | 42 | 0.8064 | 0.6538 | 0.7067 | 0.6110 | 0.6372 | 0.5945 |
| 9 | 80 | 0.6442 | 0.6907 | 0.5188 | 0.6244 | 0.7163 | 0.6858 |
