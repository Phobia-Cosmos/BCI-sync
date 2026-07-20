# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.649341 |
| final_old_mf1 | 0.605254 |
| old_aaa | 0.659842 |
| old_aaf1 | 0.621582 |
| old_fr | 0.075612 |
| mean_current_before_acc | 0.517520 |
| mean_current_after_acc | 0.499976 |
| mean_current_acc_gain | -0.017543 |
| mean_current_before_mf1 | 0.430864 |
| mean_current_after_mf1 | 0.413445 |
| mean_current_mf1_gain | -0.017419 |
| final_seen_acc | 0.492255 |
| final_seen_mf1 | 0.404506 |
| bwt_acc | -0.007721 |
| bwt_mf1 | -0.008939 |
| mean_pseudo_acc_diagnostic_only | 0.282653 |
| mean_pseudo_mf1_diagnostic_only | 0.242468 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.6186 | 0.7631 | 0.6465 | 0.3279 | 0.3039 |
| 2 | 89 | 0.5530 | 0.5140 | 0.4959 | 0.4590 | 0.2920 | 0.2822 |
| 3 | 1 | 0.1802 | 0.1826 | 0.0721 | 0.0761 | 0.2337 | 0.2138 |
| 4 | 27 | 0.7352 | 0.7341 | 0.6499 | 0.6499 | 0.2727 | 0.2358 |
| 5 | 60 | 0.6318 | 0.6261 | 0.5515 | 0.5441 | 0.2830 | 0.2510 |
| 6 | 5 | 0.3167 | 0.3167 | 0.0976 | 0.0978 | 0.3357 | 0.1966 |
| 7 | 52 | 0.7273 | 0.7239 | 0.6115 | 0.6043 | 0.3261 | 0.3003 |
| 8 | 42 | 0.4705 | 0.4513 | 0.5140 | 0.5025 | 0.2321 | 0.2133 |
| 9 | 80 | 0.6314 | 0.6326 | 0.4863 | 0.4876 | 0.2570 | 0.2330 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2663 | 0.1947 |
