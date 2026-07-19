# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.226826 |
| final_old_mf1 | 0.085946 |
| old_aaa | 0.555716 |
| old_aaf1 | 0.515062 |
| old_fr | 0.677095 |
| mean_current_before_acc | 0.500194 |
| mean_current_after_acc | 0.613779 |
| mean_current_acc_gain | 0.113584 |
| mean_current_before_mf1 | 0.393165 |
| mean_current_after_mf1 | 0.555201 |
| mean_current_mf1_gain | 0.162036 |
| final_seen_acc | 0.307429 |
| final_seen_mf1 | 0.144917 |
| bwt_acc | -0.306349 |
| bwt_mf1 | -0.410284 |
| mean_pseudo_acc_diagnostic_only | 0.607445 |
| mean_pseudo_mf1_diagnostic_only | 0.554061 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7035 | 0.7631 | 0.7263 | 0.7000 | 0.7225 |
| 2 | 89 | 0.6700 | 0.6640 | 0.6345 | 0.6237 | 0.6710 | 0.6329 |
| 3 | 1 | 0.1907 | 0.3093 | 0.0769 | 0.2570 | 0.3209 | 0.2719 |
| 4 | 27 | 0.3886 | 0.8580 | 0.2187 | 0.8251 | 0.8511 | 0.8180 |
| 5 | 60 | 0.6159 | 0.7398 | 0.5705 | 0.6861 | 0.7455 | 0.6951 |
| 6 | 5 | 0.4083 | 0.3321 | 0.2127 | 0.1479 | 0.3417 | 0.1659 |
| 7 | 52 | 0.4239 | 0.8193 | 0.2532 | 0.7831 | 0.8125 | 0.7760 |
| 8 | 42 | 0.8013 | 0.7000 | 0.6200 | 0.5999 | 0.6167 | 0.5492 |
| 9 | 80 | 0.5895 | 0.7349 | 0.5153 | 0.7017 | 0.7372 | 0.7045 |
| 10 | 26 | 0.1846 | 0.2769 | 0.0667 | 0.2013 | 0.2779 | 0.2047 |
