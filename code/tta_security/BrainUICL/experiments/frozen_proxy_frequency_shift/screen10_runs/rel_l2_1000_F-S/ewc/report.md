# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.682695 |
| final_old_mf1 | 0.653312 |
| old_aaa | 0.691159 |
| old_aaf1 | 0.667654 |
| old_fr | 0.028131 |
| mean_current_before_acc | 0.572870 |
| mean_current_after_acc | 0.587352 |
| mean_current_acc_gain | 0.014482 |
| mean_current_before_mf1 | 0.480828 |
| mean_current_after_mf1 | 0.501827 |
| mean_current_mf1_gain | 0.020999 |
| final_seen_acc | 0.567762 |
| final_seen_mf1 | 0.477813 |
| bwt_acc | -0.019590 |
| bwt_mf1 | -0.024014 |
| mean_pseudo_acc_diagnostic_only | 0.577327 |
| mean_pseudo_mf1_diagnostic_only | 0.529109 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6802 | 0.7631 | 0.7127 | 0.6291 | 0.6303 |
| 2 | 89 | 0.6630 | 0.6950 | 0.6058 | 0.6437 | 0.6610 | 0.6214 |
| 3 | 1 | 0.1884 | 0.1977 | 0.0724 | 0.0700 | 0.3058 | 0.2584 |
| 4 | 27 | 0.8239 | 0.8648 | 0.7522 | 0.7979 | 0.8636 | 0.8187 |
| 5 | 60 | 0.5920 | 0.7034 | 0.5409 | 0.6447 | 0.7409 | 0.7016 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0966 | 0.3500 | 0.1794 |
| 7 | 52 | 0.7511 | 0.7966 | 0.6731 | 0.7318 | 0.8170 | 0.7778 |
| 8 | 42 | 0.8064 | 0.7564 | 0.6863 | 0.6896 | 0.3859 | 0.3820 |
| 9 | 80 | 0.6581 | 0.6640 | 0.5501 | 0.5646 | 0.7372 | 0.7088 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2827 | 0.2127 |
