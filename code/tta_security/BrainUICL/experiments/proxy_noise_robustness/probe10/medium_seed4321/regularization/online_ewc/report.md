# Regularization-only EEG CL: online_ewc

Guiding-model hard pseudo labels are used for every target epoch. No confidence filtering, replay, DCB, or CEA is used.

## Final summary

| metric | value |
|---|---:|
| final_old_acc | 0.585689 |
| final_old_mf1 | 0.525785 |
| old_aaa | 0.616434 |
| old_aaf1 | 0.566689 |
| old_fr | 0.166226 |
| mean_current_before_acc | 0.477690 |
| mean_current_after_acc | 0.461133 |
| mean_current_acc_gain | -0.016557 |
| mean_current_before_mf1 | 0.386152 |
| mean_current_after_mf1 | 0.367796 |
| mean_current_mf1_gain | -0.018355 |
| final_seen_acc | 0.428105 |
| final_seen_mf1 | 0.331803 |
| bwt_acc | -0.033028 |
| bwt_mf1 | -0.035993 |
| mean_pseudo_acc_diagnostic_only | 0.393461 |
| mean_pseudo_mf1_diagnostic_only | 0.334836 |
| pseudo_label_coverage | 1.000000 |

## Per-subject adaptation

| task | subject | before ACC | after ACC | before MF1 | after MF1 | pseudo ACC | pseudo MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 0.7291 | 0.5698 | 0.7631 | 0.6025 | 0.3593 | 0.3468 |
| 2 | 89 | 0.5000 | 0.4380 | 0.4515 | 0.3757 | 0.4060 | 0.3720 |
| 3 | 1 | 0.1837 | 0.1849 | 0.0808 | 0.0767 | 0.3023 | 0.2809 |
| 4 | 27 | 0.6841 | 0.7148 | 0.5943 | 0.6157 | 0.5136 | 0.4425 |
| 5 | 60 | 0.5466 | 0.6080 | 0.4580 | 0.5035 | 0.4841 | 0.4178 |
| 6 | 5 | 0.3167 | 0.3155 | 0.0977 | 0.0972 | 0.3571 | 0.2071 |
| 7 | 52 | 0.6830 | 0.6920 | 0.5302 | 0.5393 | 0.4591 | 0.3982 |
| 8 | 42 | 0.3385 | 0.3256 | 0.3840 | 0.3864 | 0.3538 | 0.3042 |
| 9 | 80 | 0.5953 | 0.5628 | 0.4351 | 0.4144 | 0.4174 | 0.3613 |
| 10 | 26 | 0.2000 | 0.2000 | 0.0667 | 0.0667 | 0.2817 | 0.2176 |
