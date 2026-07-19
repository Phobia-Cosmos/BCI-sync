# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.550060 |
| final_old_mf1 | 0.536148 |
| old_aaa | 0.662900 |
| old_aaf1 | 0.644389 |
| old_fr | 0.216947 |
| mean_current_before_acc | 0.574757 |
| mean_current_after_acc | 0.595252 |
| mean_current_acc_gain | 0.020495 |
| mean_current_before_mf1 | 0.494822 |
| mean_current_after_mf1 | 0.518294 |
| mean_current_mf1_gain | 0.023472 |
| final_seen_acc | 0.520787 |
| final_seen_mf1 | 0.446622 |
| bwt_acc | -0.074465 |
| bwt_mf1 | -0.071672 |
| mean_pseudo_acc_diagnostic_only | 0.604876 |
| mean_pseudo_mf1_diagnostic_only | 0.550055 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7100 | 0.6939 | 0.6598 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1860 | 0.2012 | 0.0766 | 0.0702 | 0.3093 | 0.2597 |
| 4 | 27 | 0.8375 | 0.8761 | 0.7666 | 0.8290 | 0.8636 | 0.8188 |
| 5 | 60 | 0.6125 | 0.7443 | 0.5648 | 0.7013 | 0.7364 | 0.6936 |
| 6 | 5 | 0.3214 | 0.3155 | 0.1040 | 0.0980 | 0.3560 | 0.1797 |
