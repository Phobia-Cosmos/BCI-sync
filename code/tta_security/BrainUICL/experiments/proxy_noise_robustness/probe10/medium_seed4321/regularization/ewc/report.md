# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.595928 |
| final_old_mf1 | 0.536191 |
| old_aaa | 0.617529 |
| old_aaf1 | 0.568000 |
| old_fr | 0.151649 |
| mean_current_before_acc | 0.477301 |
| mean_current_after_acc | 0.457989 |
| mean_current_acc_gain | -0.019312 |
| mean_current_before_mf1 | 0.387201 |
| mean_current_after_mf1 | 0.365697 |
| mean_current_mf1_gain | -0.021504 |
| final_seen_acc | 0.437248 |
| final_seen_mf1 | 0.340090 |
| bwt_acc | -0.020740 |
| bwt_mf1 | -0.025607 |
| mean_pseudo_acc_diagnostic_only | 0.393345 |
| mean_pseudo_mf1_diagnostic_only | 0.334702 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5698 | 0.7631 | 0.6025 | 0.3593 | 0.3468 |
| 2 | 89 | 0.5000 | 0.4370 | 0.4515 | 0.3749 | 0.4060 | 0.3720 |
| 3 | 1 | 0.1837 | 0.1872 | 0.0808 | 0.0784 | 0.3047 | 0.2832 |
| 4 | 27 | 0.6898 | 0.7068 | 0.5986 | 0.6085 | 0.5125 | 0.4417 |
| 5 | 60 | 0.5318 | 0.5989 | 0.4472 | 0.4971 | 0.4852 | 0.4185 |
| 6 | 5 | 0.3155 | 0.3155 | 0.0962 | 0.0974 | 0.3548 | 0.2022 |
| 7 | 52 | 0.6841 | 0.6977 | 0.5325 | 0.5459 | 0.4602 | 0.4001 |
| 8 | 42 | 0.3449 | 0.3205 | 0.3998 | 0.3818 | 0.3526 | 0.3042 |
| 9 | 80 | 0.5942 | 0.5465 | 0.4356 | 0.4038 | 0.4174 | 0.3615 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2808 | 0.2167 |
