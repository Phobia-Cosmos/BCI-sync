# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.697425 |
| final_old_mf1 | 0.676662 |
| old_aaa | 0.700185 |
| old_aaf1 | 0.681846 |
| old_fr | 0.007161 |
| mean_current_before_acc | 0.578699 |
| mean_current_after_acc | 0.597968 |
| mean_current_acc_gain | 0.019269 |
| mean_current_before_mf1 | 0.488258 |
| mean_current_after_mf1 | 0.513051 |
| mean_current_mf1_gain | 0.024793 |
| final_seen_acc | 0.587053 |
| final_seen_mf1 | 0.501738 |
| bwt_acc | -0.010916 |
| bwt_mf1 | -0.011313 |
| mean_pseudo_acc_diagnostic_only | 0.569219 |
| mean_pseudo_mf1_diagnostic_only | 0.512425 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7105 | 0.7631 | 0.7406 | 0.6360 | 0.6590 |
| 2 | 89 | 0.6950 | 0.6970 | 0.6325 | 0.6502 | 0.6150 | 0.5781 |
| 3 | 1 | 0.1872 | 0.1953 | 0.0772 | 0.0702 | 0.3221 | 0.2754 |
| 4 | 27 | 0.8159 | 0.8580 | 0.7428 | 0.7961 | 0.7909 | 0.7411 |
| 5 | 60 | 0.6057 | 0.6955 | 0.5489 | 0.6372 | 0.7011 | 0.6393 |
| 6 | 5 | 0.3202 | 0.3167 | 0.1024 | 0.0977 | 0.3536 | 0.1801 |
| 7 | 52 | 0.7636 | 0.8182 | 0.6919 | 0.7616 | 0.7295 | 0.6801 |
| 8 | 42 | 0.8051 | 0.8026 | 0.6902 | 0.7064 | 0.5949 | 0.5383 |
| 9 | 80 | 0.6651 | 0.6860 | 0.5668 | 0.6038 | 0.6721 | 0.6312 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2769 | 0.2016 |
