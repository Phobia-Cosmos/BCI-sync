# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.708743 |
| final_old_mf1 | 0.687898 |
| old_aaa | 0.710904 |
| old_aaf1 | 0.692271 |
| old_fr | 0.008951 |
| mean_current_before_acc | 0.636762 |
| mean_current_after_acc | 0.637356 |
| mean_current_acc_gain | 0.000594 |
| mean_current_before_mf1 | 0.548590 |
| mean_current_after_mf1 | 0.549981 |
| mean_current_mf1_gain | 0.001392 |
| final_seen_acc | 0.633251 |
| final_seen_mf1 | 0.547371 |
| bwt_acc | -0.004105 |
| bwt_mf1 | -0.002611 |
| mean_pseudo_acc_diagnostic_only | 0.646786 |
| mean_pseudo_mf1_diagnostic_only | 0.596190 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7290 | 0.6939 | 0.6727 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1953 | 0.1930 | 0.0801 | 0.0706 | 0.3035 | 0.2524 |
| 4 | 27 | 0.8080 | 0.8216 | 0.7294 | 0.7438 | 0.8670 | 0.8239 |
| 5 | 60 | 0.6398 | 0.6648 | 0.5660 | 0.5895 | 0.7398 | 0.6890 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0980 | 0.0978 | 0.3524 | 0.1774 |
| 7 | 52 | 0.8080 | 0.8102 | 0.7370 | 0.7404 | 0.8159 | 0.7751 |
| 8 | 42 | 0.8000 | 0.7974 | 0.6799 | 0.6799 | 0.6436 | 0.5884 |
| 9 | 80 | 0.6721 | 0.6791 | 0.5899 | 0.6037 | 0.7349 | 0.7109 |
