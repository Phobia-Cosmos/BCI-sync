# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.704311 |
| final_old_mf1 | 0.685184 |
| old_aaa | 0.705193 |
| old_aaf1 | 0.687278 |
| old_fr | 0.002643 |
| mean_current_before_acc | 0.584236 |
| mean_current_after_acc | 0.585679 |
| mean_current_acc_gain | 0.001443 |
| mean_current_before_mf1 | 0.489493 |
| mean_current_after_mf1 | 0.492963 |
| mean_current_mf1_gain | 0.003469 |
| final_seen_acc | 0.583363 |
| final_seen_mf1 | 0.491008 |
| bwt_acc | -0.002316 |
| bwt_mf1 | -0.001954 |
| mean_pseudo_acc_diagnostic_only | 0.568825 |
| mean_pseudo_mf1_diagnostic_only | 0.510577 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7105 | 0.7631 | 0.7406 | 0.6360 | 0.6590 |
| 2 | 89 | 0.6950 | 0.7010 | 0.6325 | 0.6447 | 0.6150 | 0.5781 |
| 3 | 1 | 0.2012 | 0.1977 | 0.0878 | 0.0812 | 0.3244 | 0.2766 |
| 4 | 27 | 0.8034 | 0.8114 | 0.7051 | 0.7174 | 0.7920 | 0.7410 |
| 5 | 60 | 0.6227 | 0.6307 | 0.5400 | 0.5516 | 0.6955 | 0.6238 |
| 6 | 5 | 0.3179 | 0.3179 | 0.0994 | 0.0993 | 0.3536 | 0.1799 |
| 7 | 52 | 0.8284 | 0.8307 | 0.7660 | 0.7702 | 0.7364 | 0.6816 |
| 8 | 42 | 0.7808 | 0.7885 | 0.6493 | 0.6647 | 0.5949 | 0.5388 |
| 9 | 80 | 0.6640 | 0.6686 | 0.5851 | 0.5932 | 0.6616 | 0.6224 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2788 | 0.2046 |
