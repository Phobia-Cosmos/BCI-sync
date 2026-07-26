## 输出反馈驱动的渐进式 Proxy：EWC 与 Plain ER 完整验证

固定 seed 4321 和原始 49-task subject 顺序；奇数 25 个位置由 pretrain-train subject 18 的固定 48-sequence pool 按当前 task 原数量替换，偶数 24 个位置仍上传原 clean subject。偶数 clean 上传不是额外 probe、不会增加任务或数据量，但其 EEG/EOG、hard label 和 victim 返回的五类概率可被本地 proxy 使用。victim 仅返回本用户上传后的概率，不暴露参数、梯度、optimizer、正则器或 replay memory；victim 学习率、训练轮数和算法逻辑均未修改。

Progressive 条件中，proxy 与 victim 从同一公开 pretrain checkpoint 初始化；本地模型用 pretrain train clean+hard label 保持 source 能力，并用此前偶数 clean 与奇数 proxy 上传的返回概率蒸馏。完整流仍为 2148 条 sequence，其中 proxy 为 1072 条，原 clean 为 1076 条；最终本地 labeled clean buffer 为 2106 条，即 1030 条 pretrain train clean 加 1076 条后续 clean，概率反馈 buffer 为全部 2148 条 clean/proxy 上传。clean 部分不计入 proxy 预算。每个奇数版本从上一版固定 pool 继续更新，单步 relative-L2≤1%、累计≤20%、单步 L∞/std≤0.025、累计≤0.50，并约束输入方向锥和历史 classifier-gradient 方向。Static 条件在相同奇数位置重复使用未变化的 subject 18 pool，不使用反馈更新或渐进偏移。

| 方法/条件 | old ACC/MF1 | Δold vs Clean | new ACC/MF1 | Δnew vs Clean |
|---|---:|---:|---:|---:|
| EWC / Clean | 68.99%/66.43% | +0.00 pp/+0.00 pp | 62.51%/53.47% | +0.00 pp/+0.00 pp |
| EWC / Static | 64.50%/60.05% | -4.49 pp/-6.37 pp | 59.95%/50.91% | -2.56 pp/-2.56 pp |
| EWC / Progressive | 62.21%/57.47% | -6.78 pp/-8.95 pp | 59.15%/50.44% | -3.35 pp/-3.03 pp |
| Plain ER / Clean | 67.19%/64.75% | +0.00 pp/+0.00 pp | 62.69%/54.60% | +0.00 pp/+0.00 pp |
| Plain ER / Static | 70.52%/67.35% | +3.34 pp/+2.60 pp | 64.23%/56.46% | +1.54 pp/+1.86 pp |
| Plain ER / Progressive | 70.02%/66.65% | +2.83 pp/+1.90 pp | 64.44%/56.64% | +1.74 pp/+2.04 pp |

| 方法 | Progressive − Static old ACC/MF1 | Progressive − Static new ACC/MF1 | 反馈 KL 改善事件 | 最大单步/累计 relative-L2 | ER memory/replay proxy |
|---|---:|---:|---:|---:|---:|
| EWC | -2.29 pp/-2.58 pp | -0.80 pp/-0.47 pp | 33/49 | 1.00%/7.45% | — |
| Plain ER | -0.50 pp/-0.70 pp | +0.21 pp/+0.18 pp | 37/49 | 1.00%/7.49% | 50.10%/50.65% |

判断标准分两层：`Progressive − Clean` 回答完整部署轨迹是否退化，`Progressive − Static` 才回答概率反馈、历史 buffer 与同向渐进是否比固定 subject 替换更有效。负差值表示退化。

EWC 上方法成立：Progressive 相对 Clean 的 old/new ACC 分别下降 6.78 pp/3.35 pp，相对 Static 又下降 2.29 pp/0.80 pp。这说明概率反馈、历史 buffer 和渐进同向更新在该 EWC 轨迹上确实增加了总退化。

Plain ER 上方法未成立：Progressive 相对 Clean 的 old/new ACC 仍提高 2.83 pp/1.74 pp；相对 Static 虽使 old ACC 下降 0.50 pp，new ACC 却提高 0.21 pp。因此不能把它认定为 Plain ER 上有效的退化方法，更不能声称当前方法已经跨两种 CL 算法成立。ER 中约一半 memory/replay 都来自 proxy，仍未形成最终退化，说明仅提高进入 replay 的覆盖率和 surrogate 跟踪精度并不足够。

概率跟踪机制本身运行正常：EWC 有 33/49 次、Plain ER 有 37/49 次反馈后 KL 下降。但 KL 下降只证明本地 proxy 更接近 victim 输出，不等价于生成的数据一定让 victim 最终退化；Plain ER 结果正是这一边界。

既有同 seed `k25_q50` frozen N→N 结果为：EWC Δold/new ACC -2.38 pp/-3.89 pp，Plain ER 为 -2.80 pp/-3.55 pp。`k25_q100` 为：EWC -1.06 pp/-1.98 pp，Plain ER +2.02 pp/+0.10 pp。这些旧条件的数据来源和生成轨迹不同，只作为幅度参照，不作为严格配对消融。

所有结果均为 seed 4321 的单次严格方法内比较。动态反馈为 victim-specific，EWC 和 Plain ER 的 Progressive payload 不相同，不能用两者绝对分数作算法间优劣结论。
