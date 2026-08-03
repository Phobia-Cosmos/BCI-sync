# Cross-partition Validation of Selected CL Orders

Deltas are selected order minus the same-partition seed-random clean run, in percentage points.

| Seed | Method | Selected order | Final old ACC/MF1 delta | Final seen-new ACC/MF1 delta | AAA/AAF1 delta |
|---:|---|---|---:|---:|---:|
| 4322 | ewc | hard_to_easy | +35.791/+37.720 | +18.505/+21.718 | -1.844/-2.830 |
| 4323 | ewc | hard_to_easy | +29.647/+38.230 | +24.831/+29.045 | -2.164/-2.574 |
| 4322 | plain_er | source_near_to_far | +11.016/+13.117 | +8.515/+11.626 | +4.801/+6.779 |
| 4323 | plain_er | source_near_to_far | +11.150/+13.083 | +11.528/+15.275 | +3.368/+4.036 |
