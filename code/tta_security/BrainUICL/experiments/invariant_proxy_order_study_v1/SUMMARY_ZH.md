# Invariant-preserving Population Proxy Across Fixed Orders

Deltas are Proxy minus the paired clean run with the identical partition and subject order, in percentage points.

| Order | Old ACC/MF1 delta | Seen-new ACC/MF1 delta | All four negative |
|---|---:|---:|---:|
| diversity_greedy | +0.024/+0.038 | +0.079/+0.112 | no |
| easy_to_hard | -0.036/-0.010 | +0.031/+0.069 | no |
| hard_to_easy | -0.042/-0.068 | -0.007/-0.017 | yes |
| seed_random | -0.078/-0.068 | -0.144/-0.122 | yes |
| smooth_nearest | +0.132/+0.154 | +0.127/+0.208 | no |
| source_near_to_far | +0.036/+0.016 | -0.109/-0.043 | no |

Audit: `{"runs": 6, "all_have_five_proxy_tasks": true, "all_sequences_modified": true, "max_invariant_drift": 0.014439011167343706, "max_step_relative_l2": 0.050000011920928955, "max_cumulative_relative_l2": 0.10206633806228638, "all_victim_parameters_hidden": true}`
