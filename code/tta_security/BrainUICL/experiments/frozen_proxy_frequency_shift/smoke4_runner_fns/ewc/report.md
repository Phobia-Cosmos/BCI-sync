# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.711976 |
| final_old_mf1 | 0.697179 |
| old_aaa | 0.707880 |
| old_aaf1 | 0.693225 |
| old_fr | 0.013554 |
| mean_current_before_acc | 0.614352 |
| mean_current_after_acc | 0.618601 |
| mean_current_acc_gain | 0.004249 |
| mean_current_before_mf1 | 0.558070 |
| mean_current_after_mf1 | 0.567111 |
| mean_current_mf1_gain | 0.009042 |
| final_seen_acc | 0.620013 |
| final_seen_mf1 | 0.568061 |
| bwt_acc | 0.001413 |
| bwt_mf1 | 0.000949 |
| mean_pseudo_acc_diagnostic_only | 0.624053 |
| mean_pseudo_mf1_diagnostic_only | 0.581765 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7233 | 0.7631 | 0.7573 | 0.7093 | 0.7374 |
| 2 | 89 | 0.7920 | 0.7850 | 0.7257 | 0.7198 | 0.7770 | 0.7118 |
| 3 | 1 | 0.1988 | 0.2105 | 0.0875 | 0.1123 | 0.2360 | 0.1663 |
| 4 | 27 | 0.7375 | 0.7557 | 0.6559 | 0.6790 | 0.7739 | 0.7115 |
