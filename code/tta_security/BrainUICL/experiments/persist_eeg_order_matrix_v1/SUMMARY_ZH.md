# PERSIST-EEG K=5 Order Matrix

Values are mean +/- sample standard deviation across three paired seeds, in percentage points. Order: old ACC / old MF1 / seen-new ACC / seen-new MF1.

| Dataset | Schedule | Method | Paired delta (pp) | All four negative |
|---|---|---|---:|---:|
| ISRUC | uniform_random | ewc | `-11.061 +/- 18.953 / -13.236 +/- 22.680 / -7.051 +/- 11.933 / -9.039 +/- 15.533` | 2/3 |
| ISRUC | uniform_random | plain_er | `-1.392 +/- 1.668 / -1.236 +/- 1.865 / -1.030 +/- 0.605 / -1.617 +/- 1.086` | 3/3 |
| ISRUC | stratified_random | ewc | `-0.527 +/- 0.683 / +0.390 +/- 0.875 / -0.549 +/- 0.653 / +0.162 +/- 0.649` | 2/3 |
| ISRUC | stratified_random | plain_er | `-0.306 +/- 0.208 / +0.270 +/- 0.741 / -0.561 +/- 1.113 / -0.434 +/- 1.967` | 1/3 |
| ISRUC | late_random | ewc | `-15.509 +/- 25.251 / -18.568 +/- 32.292 / -9.197 +/- 16.112 / -11.419 +/- 21.261` | 2/3 |
| ISRUC | late_random | plain_er | `-0.107 +/- 2.973 / +0.322 +/- 3.841 / +0.225 +/- 2.388 / +0.050 +/- 3.659` | 2/3 |
| FACED | uniform_random | ewc | `-0.095 +/- 0.060 / -0.115 +/- 0.075 / +0.055 +/- 0.108 / +0.030 +/- 0.082` | 1/3 |
| FACED | uniform_random | plain_er | `+0.069 +/- 0.131 / +0.007 +/- 0.099 / -0.048 +/- 0.114 / -0.060 +/- 0.185` | 1/3 |
| FACED | stratified_random | ewc | `-0.373 +/- 0.478 / -0.550 +/- 0.721 / -0.167 +/- 0.361 / -0.193 +/- 0.399` | 1/3 |
| FACED | stratified_random | plain_er | `+0.061 +/- 0.015 / +0.077 +/- 0.118 / -0.102 +/- 0.125 / -0.184 +/- 0.148` | 0/3 |
| FACED | late_random | ewc | `-2.457 +/- 1.006 / -2.851 +/- 1.270 / -2.018 +/- 1.648 / -2.391 +/- 1.439` | 3/3 |
| FACED | late_random | plain_er | `-0.625 +/- 1.066 / -0.648 +/- 0.760 / -0.181 +/- 0.675 / -0.329 +/- 0.470` | 1/3 |

Audit: `{"clean_runs": 12, "persist_runs": 36, "all_proxy_tasks_equal_5": true, "all_sequences_modified": true, "all_cumulative_relative_l2_within_0_20": true, "all_direction_banks_full": true, "victim_parameters_never_visible": true, "all_pretrain_seed_4321": true, "max_observed_cumulative_relative_l2": 0.13744790852069855}`
