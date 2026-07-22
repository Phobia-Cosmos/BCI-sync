# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.703473 |
| final_old_mf1 | 0.680158 |
| old_aaa | 0.699989 |
| old_aaf1 | 0.679005 |
| old_fr | 0.001449 |
| mean_current_before_acc | 0.584510 |
| mean_current_after_acc | 0.581409 |
| mean_current_acc_gain | -0.003101 |
| mean_current_before_mf1 | 0.489830 |
| mean_current_after_mf1 | 0.486047 |
| mean_current_mf1_gain | -0.003783 |
| final_seen_acc | 0.576729 |
| final_seen_mf1 | 0.483294 |
| bwt_acc | -0.004680 |
| bwt_mf1 | -0.002753 |
| mean_pseudo_acc_diagnostic_only | 0.292027 |
| mean_pseudo_mf1_diagnostic_only | 0.253491 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6930 | 0.7631 | 0.7253 | 0.3291 | 0.3369 |
| 2 | 89 | 0.7580 | 0.7180 | 0.6942 | 0.6495 | 0.2920 | 0.2792 |
| 3 | 1 | 0.2128 | 0.2140 | 0.1096 | 0.1063 | 0.1791 | 0.1470 |
| 4 | 27 | 0.7420 | 0.7545 | 0.6420 | 0.6526 | 0.4420 | 0.3930 |
| 5 | 60 | 0.6659 | 0.6727 | 0.5675 | 0.5721 | 0.2102 | 0.1644 |
| 6 | 5 | 0.3119 | 0.3274 | 0.1065 | 0.1275 | 0.3036 | 0.2112 |
| 7 | 52 | 0.8045 | 0.8125 | 0.7540 | 0.7613 | 0.3466 | 0.3072 |
| 8 | 42 | 0.7795 | 0.7692 | 0.6168 | 0.6049 | 0.1474 | 0.1321 |
| 9 | 80 | 0.6500 | 0.6547 | 0.5703 | 0.5742 | 0.3337 | 0.2757 |
| 10 | 26 | 0.1913 | 0.1981 | 0.0744 | 0.0869 | 0.3365 | 0.2882 |
