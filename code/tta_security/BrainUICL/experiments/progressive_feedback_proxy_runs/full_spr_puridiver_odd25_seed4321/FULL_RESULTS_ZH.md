## Full SPR与PuriDivER的Proxy v2 Full49验证

固定seed 4321、原始49-task顺序、奇数25个proxy任务、偶数24个clean反馈任务，以及Proxy v2单步5%/累计20%预算。每个奇数任务完整上传subject 18的48条sequence；每种方法分别运行同位置Fixed-clean48和反馈驱动v2。没有叠加额外置信度过滤或外部净化，但Full SPR仍保留其作为CL算法组成部分的Delayed Buffer、Expert/Base NT-Xent、SCF与Purified Memory完整流程，Full PuriDivER仍保留逐mini-batch动态purity-diversity memory以及逐replay epoch重算的C/R/U流程。增量hard label、victim参数和method memory均不返回给proxy。

| 方法/条件 | old ACC/MF1 | Δold vs Fixed-clean48 | new ACC/MF1 | Δnew vs Fixed-clean48 | 四终点均下降 |
|---|---:|---:|---:|---:|---:|
| Full SPR-EEG adapted / Fixed-clean48 | 65.65%/63.28% | +0.00 pp/+0.00 pp | 58.67%/50.92% | +0.00 pp/+0.00 pp | - |
| Full SPR-EEG adapted / Proxy v2 | 68.03%/64.35% | +2.38 pp/+1.07 pp | 61.93%/53.60% | +3.26 pp/+2.69 pp | 否 |
| Full PuriDivER-EEG adapted / Fixed-clean48 | 54.75%/54.86% | +0.00 pp/+0.00 pp | 51.83%/47.26% | +0.00 pp/+0.00 pp | - |
| Full PuriDivER-EEG adapted / Proxy v2 | 35.79%/37.76% | -18.96 pp/-17.10 pp | 40.75%/34.55% | -11.08 pp/-12.71 pp | 是 |
| EWC / 既有Proxy v2参照 | 61.92%/57.76% | -3.19 pp/-2.93 pp | 58.91%/50.73% | -1.38 pp/-0.57 pp | 是 |
| Plain ER / 既有Proxy v2参照 | 69.85%/66.30% | -1.04 pp/-1.44 pp | 64.12%/56.10% | -0.70 pp/-0.68 pp | 是 |

| 方法 | Fixed memory/replay位置记录 | v2 memory/replay proxy | v2累计L2均值/最大 | pseudo-label保持/目标命中 | source cosine mean/max/达标任务 | KL改善 |
|---|---:|---:|---:|---:|---:|---:|
| Full SPR-EEG adapted | 4.00%/27.00% | 14.00%/32.27% | 19.80%/20.00% | 61.19%/27.05% | 0.137/0.364/9/25 | 35/49 |
| Full PuriDivER-EEG adapted | 0.40%/2.29% | 20.00%/14.12% | 19.74%/20.00% | 80.51%/14.34% | 0.502/0.804/0/25 | 43/49 |

在当前single-seed配对中，四个终点全部下降的方法为：Full PuriDivER-EEG adapted。
连续前置比连续后置更容易出现退化，原因是时序放大而不是过滤差异：前置数据先改变表示、伪标签和后续参数轨迹，随后还会参与更多次更新；Replay方法会多次抽到早期记录。既有Plain ER位置实验中，K=5前置/后置的全程proxy replay占比为35.25%/0.57%，K=25为87.00%/17.13%。EWC虽没有replay，但早期偏移会进入后续参数锚定与重要度估计。后置区块缺少这种累计窗口，单次轻微变化反而常表现为末端个体适配。
EWC与Plain ER在无额外过滤时退化仍较弱并不矛盾。当前预算下多数guide pseudo-label未改变；proxy只优化surrogate classifier的一步梯度，而victim还包含guide适配、完整encoder更新和不同CL状态，方向不会严格一致。EWC参数锚定、ER中的clean replay、后续正常个体以及CPC自监督都会稀释或修复偏移；full49终点又是在修复窗口之后测量。例如果前置K=25的Plain ER在区块末old ACC/MF1已下降5.44/3.25 pp，后续clean阶段又修复6.69/4.03 pp，所以最终不再退化。
Full SPR在task 1曾相对Fixed-clean48出现old ACC下降7.20 pp、当前个体ACC下降11.05 pp，但后续恢复，full49四终点最终反而提高。其v2最终memory/replay proxy比例为14.00%/32.27%，source-conflict仅9/25个任务达标，说明当前payload没有持续提供相反更新；SCF、purified memory、自监督与后续clean共同把它转化成了适配或正则化信号。因此当前方法不能使SPR稳定退化。
Full PuriDivER的退化在中后段随memory累积扩大：old ACC配对差值在task 20/30/40约为-2.66/-6.67/-19.23 pp，最终为-18.96 pp。Fixed-clean48最终只保留0.40%位置记录，而v2保留20.00%，全程replay比例从2.29%升到14.12%。渐变数据改变了伪标签类别构成并持续填入class-balanced memory；其低loss/低uncertainty会被后续C/R/U视为模型一致数据，形成自增强漂移。这里25个任务的source cosine均未达到0.10，说明大幅退化主要来自PuriDivER特有的memory选择与伪标签反馈，而不是一个已经跨算法稳定成立的classifier梯度冲突。
source-gradient cosine <= 0.10仅作为跨方法生成质量诊断，不作为阻塞条件；否则同一Proxy v2在某种CL结构上无法形成足够冲突时，实验会在进入victim更新前终止，无法测量该方法的真实响应。未达标任务本身说明当前生成目标没有稳定迁移到该方法。
memory中的位置记录比例用于观察SCF或purity-diversity pruning的最终保留结果；replay比例还受到记录到达时间影响，不能单独解释为净化成功率。不同方法的proxy根据各自概率反馈独立生成，因此只比较每种方法内部的v2−Fixed差值，不用绝对分数或不同payload作方法间排名。

负差值表示退化。结果为seed 4321单次完整轨迹，不代表跨seed统计显著性。
