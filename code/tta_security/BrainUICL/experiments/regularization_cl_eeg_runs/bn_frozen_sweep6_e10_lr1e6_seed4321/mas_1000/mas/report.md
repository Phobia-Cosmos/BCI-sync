# Regularization-only EEG CL: mas

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.679341 |
| final_old_mf1 | 0.652894 |
| old_aaa | 0.699367 |
| old_aaf1 | 0.678647 |
| old_fr | 0.032904 |
| mean_current_before_acc | 0.576664 |
| mean_current_after_acc | 0.582946 |
| mean_current_acc_gain | 0.006281 |
| mean_current_before_mf1 | 0.493342 |
| mean_current_after_mf1 | 0.499534 |
| mean_current_mf1_gain | 0.006192 |
| final_seen_acc | 0.575397 |
| final_seen_mf1 | 0.491262 |
| bwt_acc | -0.007548 |
| bwt_mf1 | -0.008272 |
| mean_pseudo_acc_diagnostic_only | 0.604259 |
| mean_pseudo_mf1_diagnostic_only | 0.548531 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7244 | 0.7631 | 0.7514 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7620 | 0.7170 | 0.6939 | 0.6646 | 0.6640 | 0.6261 |
| 3 | 1 | 0.1919 | 0.1919 | 0.0781 | 0.0697 | 0.3012 | 0.2497 |
| 4 | 27 | 0.8193 | 0.8534 | 0.7474 | 0.7815 | 0.8659 | 0.8223 |
| 5 | 60 | 0.6375 | 0.6943 | 0.5751 | 0.6324 | 0.7409 | 0.6915 |
| 6 | 5 | 0.3202 | 0.3167 | 0.1025 | 0.0976 | 0.3536 | 0.1792 |
