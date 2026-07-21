# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.688323 |
| final_old_mf1 | 0.665357 |
| old_aaa | 0.696979 |
| old_aaf1 | 0.678315 |
| old_fr | 0.020118 |
| mean_current_before_acc | 0.583607 |
| mean_current_after_acc | 0.598938 |
| mean_current_acc_gain | 0.015331 |
| mean_current_before_mf1 | 0.493529 |
| mean_current_after_mf1 | 0.515985 |
| mean_current_mf1_gain | 0.022455 |
| final_seen_acc | 0.579596 |
| final_seen_mf1 | 0.493047 |
| bwt_acc | -0.019341 |
| bwt_mf1 | -0.022937 |
| mean_pseudo_acc_diagnostic_only | 0.598178 |
| mean_pseudo_mf1_diagnostic_only | 0.549932 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7279 | 0.7631 | 0.7601 | 0.6907 | 0.7110 |
| 2 | 89 | 0.7440 | 0.7000 | 0.6827 | 0.6514 | 0.6610 | 0.6230 |
| 3 | 1 | 0.1849 | 0.1965 | 0.0762 | 0.0701 | 0.3047 | 0.2547 |
| 4 | 27 | 0.8250 | 0.8670 | 0.7537 | 0.8060 | 0.8614 | 0.8150 |
| 5 | 60 | 0.6034 | 0.7011 | 0.5501 | 0.6425 | 0.7443 | 0.7091 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0962 | 0.3500 | 0.1727 |
| 7 | 52 | 0.7602 | 0.8080 | 0.6893 | 0.7497 | 0.8102 | 0.7700 |
| 8 | 42 | 0.8077 | 0.7885 | 0.6941 | 0.7117 | 0.5423 | 0.5253 |
| 9 | 80 | 0.6651 | 0.6849 | 0.5618 | 0.6054 | 0.7384 | 0.7119 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2788 | 0.2065 |
