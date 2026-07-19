# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.714371 |
| final_old_mf1 | 0.695236 |
| old_aaa | 0.712267 |
| old_aaf1 | 0.693657 |
| old_fr | 0.016964 |
| mean_current_before_acc | 0.574792 |
| mean_current_after_acc | 0.574251 |
| mean_current_acc_gain | -0.000540 |
| mean_current_before_mf1 | 0.483654 |
| mean_current_after_mf1 | 0.482533 |
| mean_current_mf1_gain | -0.001121 |
| final_seen_acc | 0.574580 |
| final_seen_mf1 | 0.483528 |
| bwt_acc | 0.000329 |
| bwt_mf1 | 0.000994 |
| mean_pseudo_acc_diagnostic_only | 0.605412 |
| mean_pseudo_mf1_diagnostic_only | 0.548960 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7600 | 0.6939 | 0.6975 | 0.6640 | 0.6261 |
| 3 | 1 | 0.2070 | 0.2070 | 0.0946 | 0.0933 | 0.3105 | 0.2589 |
| 4 | 27 | 0.7966 | 0.7966 | 0.6987 | 0.6987 | 0.8682 | 0.8241 |
| 5 | 60 | 0.6386 | 0.6420 | 0.5515 | 0.5541 | 0.7386 | 0.6864 |
| 6 | 5 | 0.3155 | 0.3155 | 0.1002 | 0.1001 | 0.3512 | 0.1758 |
