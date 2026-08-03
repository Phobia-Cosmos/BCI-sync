# Fixed-partition CL Individual-order Study

All orders use the same partition and initial checkpoint. Balanced rank is the mean within-group z-score of final old ACC/MF1 and final seen-new ACC/MF1; Pareto status is reported separately.

| Dataset | Method | Rank | Order | Old ACC/MF1 | Seen-new ACC/MF1 | AAA/AAF1 | Pareto | Path length |
|---|---|---:|---|---:|---:|---:|---:|---:|
| FACED | ewc | 1 | diversity_greedy | 0.247/0.243 | 0.263/0.217 | 0.232/0.219 | yes | 314.08 |
| FACED | ewc | 2 | seed_random | 0.245/0.241 | 0.265/0.216 | 0.231/0.219 | yes | 294.83 |
| FACED | ewc | 3 | source_near_to_far | 0.256/0.228 | 0.266/0.202 | 0.233/0.221 | yes | 290.18 |
| FACED | ewc | 4 | smooth_nearest | 0.248/0.221 | 0.262/0.196 | 0.230/0.216 | no | 183.39 |
| FACED | ewc | 5 | easy_to_hard | 0.201/0.190 | 0.215/0.165 | 0.232/0.219 | no | 302.30 |
| FACED | ewc | 6 | hard_to_easy | 0.173/0.148 | 0.166/0.117 | 0.233/0.220 | no | 302.30 |
| FACED | plain_er | 1 | smooth_nearest | 0.263/0.245 | 0.274/0.213 | 0.235/0.227 | yes | 183.39 |
| FACED | plain_er | 2 | source_near_to_far | 0.260/0.242 | 0.271/0.213 | 0.237/0.228 | no | 290.18 |
| FACED | plain_er | 3 | seed_random | 0.242/0.240 | 0.266/0.219 | 0.239/0.230 | yes | 294.83 |
| FACED | plain_er | 4 | diversity_greedy | 0.241/0.238 | 0.267/0.221 | 0.235/0.225 | yes | 314.08 |
| FACED | plain_er | 5 | easy_to_hard | 0.216/0.209 | 0.238/0.188 | 0.233/0.223 | no | 302.30 |
| FACED | plain_er | 6 | hard_to_easy | 0.187/0.170 | 0.188/0.141 | 0.236/0.229 | no | 302.30 |
| ISRUC | ewc | 1 | hard_to_easy | 0.697/0.689 | 0.650/0.562 | 0.586/0.547 | yes | 226.79 |
| ISRUC | ewc | 2 | smooth_nearest | 0.680/0.661 | 0.631/0.564 | 0.593/0.559 | yes | 153.05 |
| ISRUC | ewc | 3 | source_near_to_far | 0.679/0.660 | 0.629/0.561 | 0.595/0.565 | no | 222.49 |
| ISRUC | ewc | 4 | diversity_greedy | 0.685/0.660 | 0.619/0.526 | 0.610/0.580 | no | 254.20 |
| ISRUC | ewc | 5 | seed_random | 0.661/0.646 | 0.588/0.508 | 0.604/0.571 | no | 249.28 |
| ISRUC | ewc | 6 | easy_to_hard | 0.229/0.086 | 0.308/0.143 | 0.586/0.548 | no | 226.79 |
| ISRUC | plain_er | 1 | source_near_to_far | 0.733/0.707 | 0.633/0.559 | 0.661/0.632 | yes | 222.49 |
| ISRUC | plain_er | 2 | smooth_nearest | 0.699/0.680 | 0.627/0.548 | 0.668/0.641 | no | 153.05 |
| ISRUC | plain_er | 3 | hard_to_easy | 0.689/0.667 | 0.628/0.536 | 0.574/0.526 | no | 226.79 |
| ISRUC | plain_er | 4 | diversity_greedy | 0.659/0.629 | 0.607/0.515 | 0.625/0.589 | no | 254.20 |
| ISRUC | plain_er | 5 | seed_random | 0.653/0.630 | 0.584/0.507 | 0.645/0.613 | no | 249.28 |
| ISRUC | plain_er | 6 | easy_to_hard | 0.613/0.550 | 0.575/0.477 | 0.655/0.627 | no | 226.79 |
