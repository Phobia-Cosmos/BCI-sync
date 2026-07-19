# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.710778 |
| final_old_mf1 | 0.693550 |
| old_aaa | 0.712130 |
| old_aaf1 | 0.697011 |
| old_fr | 0.011849 |
| mean_current_before_acc | 0.570324 |
| mean_current_after_acc | 0.577368 |
| mean_current_acc_gain | 0.007044 |
| mean_current_before_mf1 | 0.489688 |
| mean_current_after_mf1 | 0.492648 |
| mean_current_mf1_gain | 0.002960 |
| final_seen_acc | 0.571860 |
| final_seen_mf1 | 0.486607 |
| bwt_acc | -0.005508 |
| bwt_mf1 | -0.006041 |
| mean_pseudo_acc_diagnostic_only | 0.602734 |
| mean_pseudo_mf1_diagnostic_only | 0.547722 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7130 | 0.6939 | 0.6635 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1965 | 0.1953 | 0.0916 | 0.0778 | 0.3058 | 0.2557 |
| 4 | 27 | 0.8114 | 0.8364 | 0.7354 | 0.7616 | 0.8614 | 0.8164 |
| 5 | 60 | 0.6170 | 0.6784 | 0.5497 | 0.6037 | 0.7352 | 0.6914 |
| 6 | 5 | 0.3060 | 0.3167 | 0.1045 | 0.0978 | 0.3500 | 0.1743 |
