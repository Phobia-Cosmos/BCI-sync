# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.708503 |
| final_old_mf1 | 0.687378 |
| old_aaa | 0.708274 |
| old_aaf1 | 0.689588 |
| old_fr | 0.008610 |
| mean_current_before_acc | 0.584594 |
| mean_current_after_acc | 0.587346 |
| mean_current_acc_gain | 0.002752 |
| mean_current_before_mf1 | 0.491326 |
| mean_current_after_mf1 | 0.496531 |
| mean_current_mf1_gain | 0.005206 |
| final_seen_acc | 0.584962 |
| final_seen_mf1 | 0.493905 |
| bwt_acc | -0.002384 |
| bwt_mf1 | -0.002626 |
| mean_pseudo_acc_diagnostic_only | 0.568825 |
| mean_pseudo_mf1_diagnostic_only | 0.510827 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7105 | 0.7631 | 0.7406 | 0.6360 | 0.6590 |
| 2 | 89 | 0.6950 | 0.7140 | 0.6325 | 0.6625 | 0.6150 | 0.5781 |
| 3 | 1 | 0.1953 | 0.1942 | 0.0806 | 0.0734 | 0.3233 | 0.2751 |
| 4 | 27 | 0.8091 | 0.8182 | 0.7180 | 0.7372 | 0.7932 | 0.7434 |
| 5 | 60 | 0.6295 | 0.6500 | 0.5551 | 0.5720 | 0.6955 | 0.6231 |
| 6 | 5 | 0.3179 | 0.3179 | 0.0994 | 0.0992 | 0.3512 | 0.1762 |
| 7 | 52 | 0.8080 | 0.8068 | 0.7391 | 0.7384 | 0.7398 | 0.6868 |
| 8 | 42 | 0.7923 | 0.7910 | 0.6692 | 0.6800 | 0.5923 | 0.5374 |
| 9 | 80 | 0.6698 | 0.6709 | 0.5896 | 0.5954 | 0.6651 | 0.6271 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2769 | 0.2020 |
