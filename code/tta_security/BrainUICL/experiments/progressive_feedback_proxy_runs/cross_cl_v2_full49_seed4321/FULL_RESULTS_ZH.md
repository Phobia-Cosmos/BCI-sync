## 跨正则化与Replay的Full-pool Proxy v2验证

协议固定为seed 4321和原始49-task顺序。奇数25个位置每次完整上传subject 18的48条sequence，偶数24个位置保留原clean上传；Fixed-clean48 control在奇数位置上传未变化的同一48条clean，因此v2与control的任务数、位置和数据量完全一致。pretrain训练集1030条clean及hard label是本地唯一hard-label监督；后续clean和proxy都只保存victim返回的五类概率，不能读取增量阶段annotation。

v2每次从上一版48条继续变化，实际单步relative-L2不超过5%、累计不超过20%，输入方向锥余弦不低于约0.98；本地proxy每次使用source CE和历史clean/proxy概率KL更新，并在最新proxy参数下重算历史gradient。只有任务级proxy梯度与64条source样本聚合梯度的余弦不大于0.10时才允许上传，用低冲突或近正交更新排除旧版约+0.79的正常同向更新。EWC与Plain ER使用完全相同的生成规则和超参数。

| 方法/条件 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 |
|---|---:|---:|---:|---:|
| EWC / Fixed-clean48 | 65.11%/60.70% | +0.00 pp/+0.00 pp | 60.29%/51.30% | +0.00 pp/+0.00 pp |
| EWC / Proxy v2 5%/20% | 61.92%/57.76% | -3.19 pp/-2.93 pp | 58.91%/50.73% | -1.38 pp/-0.57 pp |
| Plain ER / Fixed-clean48 | 70.89%/67.74% | +0.00 pp/+0.00 pp | 64.82%/56.77% | +0.00 pp/+0.00 pp |
| Plain ER / Proxy v2 5%/20% | 69.85%/66.30% | -1.04 pp/-1.44 pp | 64.12%/56.10% | -0.70 pp/-0.68 pp |

| 方法/强度 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 | 主判定 |
|---|---:|---:|---:|---:|---:|
| EWC / Proxy v2 5%/20% | 61.92%/57.76% | -3.19 pp/-2.93 pp | 58.91%/50.73% | -1.38 pp/-0.57 pp | 通过 |
| EWC / Proxy v2 10%/40% | 60.88%/56.43% | -4.23 pp/-4.27 pp | 55.54%/48.60% | -4.75 pp/-2.70 pp | 通过 |
| EWC / Proxy v2 15%/60% | 60.83%/55.96% | -4.28 pp/-4.74 pp | 57.73%/49.87% | -2.56 pp/-1.43 pp | 通过 |
| Plain ER / Proxy v2 5%/20% | 69.85%/66.30% | -1.04 pp/-1.44 pp | 64.12%/56.10% | -0.70 pp/-0.68 pp | 通过 |
| Plain ER / Proxy v2 10%/40% | 68.68%/64.75% | -2.21 pp/-2.99 pp | 63.75%/55.21% | -1.07 pp/-1.56 pp | 通过 |
| Plain ER / Proxy v2 15%/60% | 68.91%/65.15% | -1.98 pp/-2.59 pp | 62.64%/54.26% | -2.18 pp/-2.51 pp | 通过 |

| 方法/强度 | Δold ACC/MF1 vs原始Clean49 | Δnew ACC/MF1 vs原始Clean49 | source-gradient cosine mean/max/gate | 最终平均/最大累计relative-L2 | KL before→after/mean Δ | ER memory/replay proxy |
|---|---:|---:|---:|---:|---:|---:|
| EWC / Proxy v2 5%/20% | -7.07 pp/-8.66 pp | -3.59 pp/-2.74 pp | -0.529/-0.236/0.100 | 19.50%/20.00% | 4.739→4.351/-0.388 (35/49) | - |
| Plain ER / Proxy v2 5%/20% | +2.66 pp/+1.55 pp | +1.43 pp/+1.50 pp | -0.557/-0.295/0.100 | 19.48%/20.00% | 7.853→7.420/-0.433 (35/49) | 52.00%/53.65% |

结果满足主成功条件：Proxy v2 5%/20% 是同时使EWC和Plain ER的old/new ACC与MF1均低于各自同量Fixed-clean48 control的最低强度。

负差值表示退化。Fixed-clean48是判断proxy数据变化本身是否有效的主对照；原始Clean49的数据身份和奇数任务数据量不同，只作为额外部署参照，不参与主成功判定。结果是单seed点估计，尚不代表跨seed稳定性。
