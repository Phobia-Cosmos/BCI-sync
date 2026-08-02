# Population-PERSIST EEG Order Matrix

All values are paired against the existing clean run with the same dataset, method, seed and individual order. Values are percentage points in old ACC / old MF1 / seen-new ACC / seen-new MF1 order.

| Dataset | Schedule | Method | Paired delta mean +/- sample std | All four negative |
|---|---|---|---:|---:|
| ISRUC | uniform_random | ewc | `+1.180 +/- 0.000 / +1.078 +/- 0.000 / +3.000 +/- 0.000 / +1.617 +/- 0.000` | 0/1 |

Audit: `{"runs": 1, "all_population_mode": true, "all_without_fixed_subject": true, "all_victim_parameters_hidden": true, "all_have_five_proxy_tasks": true, "all_sequences_modified": true, "all_tasks_class_balanced": true, "all_classes_observed_across_proxy_tasks": true, "minimum_population_subjects": 11, "max_cumulative_relative_l2": 0.08540631085634232}`
