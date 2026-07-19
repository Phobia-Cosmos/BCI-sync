# Regularization-only EEG CL: si

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.698503 |
| final_old_mf1 | 0.681841 |
| old_aaa | 0.700858 |
| old_aaf1 | 0.684297 |
| old_fr | 0.005626 |
| mean_current_before_acc | 0.750035 |
| mean_current_after_acc | 0.733558 |
| mean_current_acc_gain | -0.016477 |
| mean_current_before_mf1 | 0.738975 |
| mean_current_after_mf1 | 0.723539 |
| mean_current_mf1_gain | -0.015436 |
| final_seen_acc | 0.746349 |
| final_seen_mf1 | 0.736851 |
| bwt_acc | 0.012791 |
| bwt_mf1 | 0.013312 |
| mean_pseudo_acc_diagnostic_only | 0.739884 |
| mean_pseudo_mf1_diagnostic_only | 0.728772 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7151 | 0.7631 | 0.7464 | 0.7198 | 0.7503 |
| 2 | 89 | 0.7710 | 0.7520 | 0.7148 | 0.7007 | 0.7600 | 0.7072 |
