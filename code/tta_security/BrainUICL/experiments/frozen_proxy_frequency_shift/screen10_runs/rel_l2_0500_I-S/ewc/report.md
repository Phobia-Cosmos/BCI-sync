# Regularization-only EEG CL: ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.
ICML 2026 defense mode: `none`.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.685150 |
| final_old_mf1 | 0.659007 |
| old_aaa | 0.696549 |
| old_aaf1 | 0.676710 |
| old_fr | 0.024636 |
| mean_current_before_acc | 0.584823 |
| mean_current_after_acc | 0.595514 |
| mean_current_acc_gain | 0.010691 |
| mean_current_before_mf1 | 0.493679 |
| mean_current_after_mf1 | 0.511254 |
| mean_current_mf1_gain | 0.017575 |
| final_seen_acc | 0.571825 |
| final_seen_mf1 | 0.483956 |
| bwt_acc | -0.023689 |
| bwt_mf1 | -0.027298 |
| mean_pseudo_acc_diagnostic_only | 0.587839 |
| mean_pseudo_mf1_diagnostic_only | 0.540941 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.7221 | 0.7631 | 0.7507 | 0.7000 | 0.7225 |
| 2 | 89 | 0.7610 | 0.7050 | 0.6972 | 0.6556 | 0.6650 | 0.6268 |
| 3 | 1 | 0.1872 | 0.1988 | 0.0769 | 0.0705 | 0.3058 | 0.2591 |
| 4 | 27 | 0.8239 | 0.8648 | 0.7520 | 0.8025 | 0.8625 | 0.8172 |
| 5 | 60 | 0.6023 | 0.6966 | 0.5487 | 0.6382 | 0.7409 | 0.7039 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0976 | 0.0962 | 0.3512 | 0.1788 |
| 7 | 52 | 0.7602 | 0.8057 | 0.6895 | 0.7466 | 0.8080 | 0.7658 |
| 8 | 42 | 0.8051 | 0.7769 | 0.6891 | 0.7089 | 0.4321 | 0.4259 |
| 9 | 80 | 0.6628 | 0.6698 | 0.5559 | 0.5767 | 0.7360 | 0.7057 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2769 | 0.2039 |
