# Population-PERSIST EEG Order Matrix

All values are paired against the existing clean run with the same dataset, method, seed and individual order. Values are percentage points in old ACC / old MF1 / seen-new ACC / seen-new MF1 order.

| Dataset | Schedule | Method | Paired delta mean +/- sample std | All four negative |
|---|---|---|---:|---:|
| ISRUC | late_random | ewc | `-0.006 +/- 0.000 / -1.305 +/- 0.000 / +1.531 +/- 0.000 / +0.434 +/- 0.000` | 0/1 |
| ISRUC | late_random | plain_er | `-2.000 +/- 0.000 / -2.878 +/- 0.000 / -2.012 +/- 0.000 / -2.751 +/- 0.000` | 1/1 |
| ISRUC | stratified_random | ewc | `-0.114 +/- 0.000 / -0.141 +/- 0.000 / -0.092 +/- 0.000 / -0.051 +/- 0.000` | 1/1 |
| ISRUC | stratified_random | plain_er | `-0.066 +/- 0.000 / -0.116 +/- 0.000 / +0.144 +/- 0.000 / +0.149 +/- 0.000` | 0/1 |
| ISRUC | uniform_random | ewc | `+0.347 +/- 0.000 / +0.205 +/- 0.000 / +2.579 +/- 0.000 / +1.405 +/- 0.000` | 0/1 |
| ISRUC | uniform_random | plain_er | `-0.180 +/- 0.000 / -0.598 +/- 0.000 / +0.366 +/- 0.000 / +0.058 +/- 0.000` | 0/1 |

Audit: `{"runs": 6, "all_population_mode": true, "all_without_fixed_subject": true, "all_victim_parameters_hidden": true, "all_have_five_proxy_tasks": true, "all_sequences_modified": true, "all_tasks_class_balanced": true, "all_classes_observed_across_proxy_tasks": true, "minimum_population_subjects": 7, "max_cumulative_relative_l2": 0.10316851735115051}`
