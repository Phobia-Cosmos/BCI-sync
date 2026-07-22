# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.718623 |
| final_old_mf1 | 0.702160 |
| old_aaa | 0.711426 |
| old_aaf1 | 0.694869 |
| old_fr | 0.023016 |
| mean_current_before_acc | 0.592211 |
| mean_current_after_acc | 0.591226 |
| mean_current_acc_gain | -0.000985 |
| mean_current_before_mf1 | 0.502697 |
| mean_current_after_mf1 | 0.504752 |
| mean_current_mf1_gain | 0.002055 |
| final_seen_acc | 0.596212 |
| final_seen_mf1 | 0.514307 |
| bwt_acc | 0.004986 |
| bwt_mf1 | 0.009556 |
| mean_pseudo_acc_diagnostic_only | 0.608577 |
| mean_pseudo_mf1_diagnostic_only | 0.561957 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7012 | 0.7631 | 0.7336 | 0.7035 | 0.7325 |
| 2 | 89 | 0.7780 | 0.7620 | 0.7124 | 0.7025 | 0.7230 | 0.6709 |
| 3 | 1 | 0.2058 | 0.2035 | 0.0900 | 0.0966 | 0.1570 | 0.1008 |
| 4 | 27 | 0.7591 | 0.7716 | 0.6749 | 0.6857 | 0.8545 | 0.8124 |
| 5 | 60 | 0.6761 | 0.6898 | 0.5885 | 0.5976 | 0.7330 | 0.6710 |
| 6 | 5 | 0.3048 | 0.3024 | 0.1118 | 0.1202 | 0.2131 | 0.1578 |
| 7 | 52 | 0.8011 | 0.8080 | 0.7511 | 0.7626 | 0.8273 | 0.7903 |
| 8 | 42 | 0.7859 | 0.7859 | 0.6060 | 0.6104 | 0.7987 | 0.7094 |
| 9 | 80 | 0.6860 | 0.6919 | 0.6330 | 0.6425 | 0.7430 | 0.7136 |
| 10 | 26 | 0.1962 | 0.1962 | 0.0962 | 0.0959 | 0.3327 | 0.2610 |
