# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.694731 |
| final_old_mf1 | 0.672682 |
| old_aaa | 0.700212 |
| old_aaf1 | 0.681917 |
| old_fr | 0.010997 |
| mean_current_before_acc | 0.579360 |
| mean_current_after_acc | 0.598305 |
| mean_current_acc_gain | 0.018945 |
| mean_current_before_mf1 | 0.488965 |
| mean_current_after_mf1 | 0.513744 |
| mean_current_mf1_gain | 0.024779 |
| final_seen_acc | 0.585650 |
| final_seen_mf1 | 0.499656 |
| bwt_acc | -0.012655 |
| bwt_mf1 | -0.014087 |
| mean_pseudo_acc_diagnostic_only | 0.569310 |
| mean_pseudo_mf1_diagnostic_only | 0.512452 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7105 | 0.7631 | 0.7406 | 0.6360 | 0.6590 |
| 2 | 89 | 0.6950 | 0.6970 | 0.6325 | 0.6496 | 0.6150 | 0.5781 |
| 3 | 1 | 0.1895 | 0.1942 | 0.0780 | 0.0699 | 0.3221 | 0.2751 |
| 4 | 27 | 0.8182 | 0.8591 | 0.7446 | 0.7969 | 0.7898 | 0.7402 |
| 5 | 60 | 0.6045 | 0.6943 | 0.5503 | 0.6364 | 0.7011 | 0.6393 |
| 6 | 5 | 0.3202 | 0.3155 | 0.1024 | 0.0964 | 0.3524 | 0.1774 |
| 7 | 52 | 0.7682 | 0.8170 | 0.6947 | 0.7584 | 0.7318 | 0.6835 |
| 8 | 42 | 0.8026 | 0.8013 | 0.6860 | 0.7042 | 0.5949 | 0.5383 |
| 9 | 80 | 0.6663 | 0.6942 | 0.5714 | 0.6183 | 0.6721 | 0.6312 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2779 | 0.2025 |
