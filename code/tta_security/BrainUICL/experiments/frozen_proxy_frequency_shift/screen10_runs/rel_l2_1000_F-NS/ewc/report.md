# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.684371 |
| final_old_mf1 | 0.656937 |
| old_aaa | 0.693517 |
| old_aaf1 | 0.672045 |
| old_fr | 0.025744 |
| mean_current_before_acc | 0.574917 |
| mean_current_after_acc | 0.589804 |
| mean_current_acc_gain | 0.014887 |
| mean_current_before_mf1 | 0.483841 |
| mean_current_after_mf1 | 0.505584 |
| mean_current_mf1_gain | 0.021744 |
| final_seen_acc | 0.570485 |
| final_seen_mf1 | 0.480640 |
| bwt_acc | -0.019319 |
| bwt_mf1 | -0.024944 |
| mean_pseudo_acc_diagnostic_only | 0.586605 |
| mean_pseudo_mf1_diagnostic_only | 0.539580 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6942 | 0.7631 | 0.7271 | 0.6535 | 0.6625 |
| 2 | 89 | 0.6800 | 0.6940 | 0.6260 | 0.6428 | 0.6610 | 0.6221 |
| 3 | 1 | 0.1860 | 0.1930 | 0.0740 | 0.0692 | 0.3070 | 0.2606 |
| 4 | 27 | 0.8227 | 0.8659 | 0.7489 | 0.7992 | 0.8659 | 0.8229 |
| 5 | 60 | 0.5875 | 0.6966 | 0.5372 | 0.6378 | 0.7420 | 0.7038 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0962 | 0.3548 | 0.1886 |
| 7 | 52 | 0.7580 | 0.8011 | 0.6815 | 0.7396 | 0.8148 | 0.7757 |
| 8 | 42 | 0.8064 | 0.7679 | 0.6894 | 0.7031 | 0.4487 | 0.4402 |
| 9 | 80 | 0.6628 | 0.6698 | 0.5540 | 0.5741 | 0.7395 | 0.7134 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2788 | 0.2060 |
