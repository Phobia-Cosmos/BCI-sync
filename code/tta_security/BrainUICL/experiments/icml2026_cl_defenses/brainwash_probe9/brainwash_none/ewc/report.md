# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.705928 |
| final_old_mf1 | 0.690298 |
| old_aaa | 0.706653 |
| old_aaf1 | 0.690513 |
| old_fr | 0.004944 |
| mean_current_before_acc | 0.628984 |
| mean_current_after_acc | 0.643918 |
| mean_current_acc_gain | 0.014934 |
| mean_current_before_mf1 | 0.543270 |
| mean_current_after_mf1 | 0.564532 |
| mean_current_mf1_gain | 0.021262 |
| final_seen_acc | 0.633217 |
| final_seen_mf1 | 0.551576 |
| bwt_acc | -0.010702 |
| bwt_mf1 | -0.012956 |
| mean_pseudo_acc_diagnostic_only | 0.643959 |
| mean_pseudo_mf1_diagnostic_only | 0.594910 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7150 | 0.6939 | 0.6659 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1884 | 0.1942 | 0.0841 | 0.0708 | 0.3058 | 0.2557 |
| 4 | 27 | 0.8159 | 0.8568 | 0.7464 | 0.7927 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6080 | 0.6943 | 0.5453 | 0.6331 | 0.7375 | 0.6979 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0986 | 0.0975 | 0.3524 | 0.1784 |
| 7 | 52 | 0.7727 | 0.8091 | 0.7057 | 0.7454 | 0.8114 | 0.7696 |
| 8 | 42 | 0.8077 | 0.7987 | 0.6981 | 0.7137 | 0.6295 | 0.5817 |
| 9 | 80 | 0.6605 | 0.6860 | 0.5543 | 0.6102 | 0.7337 | 0.7060 |
