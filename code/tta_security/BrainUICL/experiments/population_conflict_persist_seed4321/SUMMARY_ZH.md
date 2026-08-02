# Population-PERSIST EEG Order Matrix

All values are paired against the existing clean run with the same dataset, method, seed and individual order. Values are percentage points in old ACC / old MF1 / seen-new ACC / seen-new MF1 order.

| Dataset | Schedule | Method | Paired delta mean +/- sample std | All four negative |
|---|---|---|---:|---:|
| FACED | late_random | ewc | `-0.130 +/- 0.000 / -0.221 +/- 0.000 / -0.615 +/- 0.000 / -0.448 +/- 0.000` | 1/1 |
| FACED | late_random | plain_er | `-0.208 +/- 0.000 / -0.204 +/- 0.000 / +0.000 +/- 0.000 / +0.014 +/- 0.000` | 0/1 |
| FACED | stratified_random | ewc | `-0.052 +/- 0.000 / -0.100 +/- 0.000 / -0.123 +/- 0.000 / -0.150 +/- 0.000` | 1/1 |
| FACED | stratified_random | plain_er | `+0.156 +/- 0.000 / +0.125 +/- 0.000 / +0.051 +/- 0.000 / +0.057 +/- 0.000` | 0/1 |
| FACED | uniform_random | ewc | `-0.052 +/- 0.000 / -0.074 +/- 0.000 / -0.072 +/- 0.000 / -0.061 +/- 0.000` | 1/1 |
| FACED | uniform_random | plain_er | `+0.000 +/- 0.000 / -0.038 +/- 0.000 / +0.051 +/- 0.000 / +0.062 +/- 0.000` | 0/1 |
| ISRUC | late_random | ewc | `+2.251 +/- 0.000 / +1.538 +/- 0.000 / +4.402 +/- 0.000 / +2.954 +/- 0.000` | 0/1 |
| ISRUC | late_random | plain_er | `-0.042 +/- 0.000 / -1.072 +/- 0.000 / -0.009 +/- 0.000 / -0.804 +/- 0.000` | 1/1 |
| ISRUC | stratified_random | ewc | `-0.090 +/- 0.000 / -0.130 +/- 0.000 / -0.104 +/- 0.000 / -0.089 +/- 0.000` | 1/1 |
| ISRUC | stratified_random | plain_er | `+0.066 +/- 0.000 / +0.070 +/- 0.000 / +0.376 +/- 0.000 / +0.407 +/- 0.000` | 0/1 |
| ISRUC | uniform_random | ewc | `+1.180 +/- 0.000 / +1.078 +/- 0.000 / +3.000 +/- 0.000 / +1.617 +/- 0.000` | 0/1 |
| ISRUC | uniform_random | plain_er | `+1.485 +/- 0.000 / +0.536 +/- 0.000 / +2.278 +/- 0.000 / +1.441 +/- 0.000` | 0/1 |

Audit: `{"runs": 12, "all_population_mode": true, "all_without_fixed_subject": true, "all_victim_parameters_hidden": true, "all_have_five_proxy_tasks": true, "all_sequences_modified": true, "all_tasks_class_balanced": true, "all_classes_observed_across_proxy_tasks": true, "minimum_population_subjects": 4, "max_cumulative_relative_l2": 0.09604056179523468}`
