# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.327605 |
| final_old_mf1 | 0.245366 |
| old_aaa | 0.563276 |
| old_aaf1 | 0.527302 |
| old_fr | 0.533629 |
| mean_current_before_acc | 0.502443 |
| mean_current_after_acc | 0.606207 |
| mean_current_acc_gain | 0.103763 |
| mean_current_before_mf1 | 0.414810 |
| mean_current_after_mf1 | 0.551516 |
| mean_current_mf1_gain | 0.136706 |
| final_seen_acc | 0.391015 |
| final_seen_mf1 | 0.218080 |
| bwt_acc | -0.215192 |
| bwt_mf1 | -0.333437 |
| mean_pseudo_acc_diagnostic_only | 0.608256 |
| mean_pseudo_mf1_diagnostic_only | 0.554574 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6530 | 0.6345 | 0.6141 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3221 | 0.0769 | 0.2743 | 0.3198 | 0.2707 |
| 4 | 27 | 0.3875 | 0.8716 | 0.2173 | 0.8333 | 0.8670 | 0.8333 |
| 5 | 60 | 0.6148 | 0.7466 | 0.5750 | 0.6974 | 0.7489 | 0.7014 |
| 6 | 5 | 0.4226 | 0.3405 | 0.2220 | 0.1637 | 0.3429 | 0.1666 |
