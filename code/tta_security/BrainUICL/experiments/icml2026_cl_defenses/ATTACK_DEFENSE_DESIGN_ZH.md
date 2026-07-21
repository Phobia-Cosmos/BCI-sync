# EEG 持续学习攻击入口与 ICML 2026 防御验证设计

本文档解释当前 EEG 持续学习中的攻击应当作用在哪里、replay 与正则化方法如何受到同一输入攻击、PACOL/BrainWash/Proxy Noise/ICML 2026 理论攻击之间有什么区别，以及 T2T 和 Robust Feature Defense 应当如何验证。文档末尾记录了首轮 BrainWash 对 T2T 的实际结果。

## 一、共同攻击应该放在哪里

如果目标是公平比较 BrainUICL、SPR-EEG、PuriDivER-EEG、Finetune、EWC、Online EWC、SI 和 MAS，最合理的共同攻击入口是“当前个体上传的 EEG/EOG 数据进入学习器之前”。设第 (t) 个个体的干净数据为 (D_t={(x_i,y_i)})，共同攻击只把当前输入替换为 (x_i+delta_i)，然后让每种 CL 算法按照自己的正常流程处理这份上传数据。

同一个输入攻击进入不同 CL 算法后会沿不同路径持续传播：

| 方法类型 | 当前被污染 EEG 首先影响什么 | 污染如何持续 |
|---|---|---|
| Replay 方法 | guiding pseudo label、当前梯度、置信度/过滤分数、memory admission | 如果样本进入 buffer，以后会被多次 replay，形成时间放大 |
| 正则化方法 | guiding pseudo label、当前梯度、任务结束参数、Fisher/SI/MAS importance | 不保存 EEG，但错误被固化到参数锚点和重要性矩阵中 |
| Finetune | guiding pseudo label和当前梯度 | 只通过更新后的参数继续传播，没有 replay 或历史 importance |

因此，“攻击同时作用于 replay 和正则化”不应理解为攻击者一边直接篡改 replay buffer、一边篡改 EWC 矩阵。共同公平攻击只污染当前上传数据。Replay 是否把污染写进 buffer、正则化方法是否把污染写进 importance，都是被测算法自身产生的后果。

直接修改历史 replay buffer 属于更强的 storage-tampering threat model。它只能攻击有 buffer 的方法，对 EWC/SI/MAS 没有对应入口，必须单独作为 replay-specific attack 报告，不能混入共同主表。

## 二、攻击目标应该针对什么

共同输入攻击至少需要同时记录四层目标，不能只报告 EEG 上加了多少噪声：

1. 输入层：限制每个 sequence 的 (ell_infty)、相对 (ell_2)、被污染 sequence 比例和被攻击任务比例。
2. 伪标签层：记录 guiding model 的伪标签翻转率、置信度变化和类别分布变化。在无标签 EEG 中，攻击 (x) 可能间接改变训练用的伪标签，这与监督图像攻击“标签保持不变”不同。
3. 更新层：优化或测量当前梯度与历史目标梯度的冲突、一次近似更新后的历史 proxy loss、参数更新范数和任务 Hessian/Fisher 变化。
4. 持久化层：Replay 方法报告污染 sequence 进入 buffer 的比例及后续 replay 次数；正则化方法报告 importance/anchor 的变化和最终 BWT。

建议保留两条互补攻击线：

| 攻击线 | 输入是否跨方法完全相同 | 回答的问题 |
|---|---:|---|
| Fixed shared upload | 是 | 在相同攻击数据和预算下，哪类 CL 机制更容易受损 |
| Adaptive white-box | 否，每个 victim 重新优化 | 知道 victim 状态的攻击者最多可以造成多大破坏 |

Fixed shared upload 适合主横向表；adaptive white-box 适合最坏情况安全评估。二者不能放在同一列而不注明攻击是否按 victim 重生成。

## 三、现有攻击与论文攻击的区别

| 攻击 | 原始场景和目标 | 当前 EEG 实现 | 关键差异 |
|---|---|---|---|
| PACOL | 对未来任务做 clean-label、范数有界的输入投毒，使当前梯度模仿针对旧任务的错误标签梯度；原文测试 EWC、Online EWC、SI 和 DGR | 使用 source EEG proxy 和 victim 硬预测构造旧任务错误梯度，优化当前 EEG 输入 | EEG 的训练标签是 guiding model 伪标签，输入改变后伪标签也可能改变，因此不再是严格监督语义下的 clean-label；当前只用分类器或有限参数范围近似 |
| BrainWash | 攻击最新任务，通过模型反演恢复旧数据，再做双层优化，使一次当前任务更新增大旧任务损失；覆盖 regularization 和 memory-based CL | 直接使用 source EEG proxy，不执行原论文模型反演；只展开一步分类器更新 | 当前攻击者比原论文在旧数据方面更强，但优化范围更窄；EEG 是单头 domain/subject incremental，而原文主要是多头 task incremental |
| Proxy Noise | 本仓库为跨算法比较构造的固定代理输入流 | BrainUICL proxy 为所有算法生成同一批上传 EEG 扰动 | 便于 replay/正则化公平比较，但不自动满足 shifted 或条件零均值定义，也不等价于 PACOL/BrainWash |
| ICML 2026 理论攻击 | 在广义二次正则化 CL 中，对当前 (x) 和 (y) 的扰动进行 online minimax 分析 | T2T 与 Robust Feature 的理论参照 | 它是一类按攻击频率、预算增长和均值漂移划分的理论攻击，不是名为“正则化攻击”的具体生成器，也没有声称任意 BrainWash 都会被 T2T 检出 |

这里的 ICML 2026 论文是 `Theory of Continual Learning Against Data Poisoning Attacks`。论文把攻击分为：少量但可以非常强且带均值漂移的 shifted attack，以及频繁、幅度有界、条件零均值的 non-shifted attack。T2T 针对前者，Robust Feature 针对后者。用不匹配的攻击验证防御，只能作为额外压力测试，不能验证对应定理。

## 四、T2T 的 2.5 倍阈值是什么

T2T 在任务 (t) 完成一次临时更新后计算动态分数 (d_t)。它不是准确率，也不是 loss，而是由最近两个参数更新、最近两个任务 Hessian 以及正则矩阵共同构造的“局部更新动力学异常程度”。实验启发式规则是：

$$
d_t \ge 2.5\times \frac{1}{m}\sum_{s=t-m}^{t-1}d_s,
\qquad m=\min(5,\text{此前可用分数数}).
$$

例如此前 5 个有效分数是 (2,3,4,5,6)，均值是 4，则当前分数至少达到 10 才触发。这里的“有效检测分数”表示已经具备连续三个模型状态，并且对角化后的 (A_t) 和 (B_t) 存在共同非零参数方向，因此能够实际计算 (d_t) 的任务数。它不是“正确检测到攻击”的数量。

“触发次数”表示有多少个有效分数超过阈值。启用论文 Algorithm 1 的 rollback 动作后，每次触发会把模型、BatchNorm buffers 和正则器状态恢复到两次更新之前。

## 五、为什么会回滚两个任务，以及这里的任务是什么

论文只根据最近两个更新判断“任务 (t-1) 或任务 (t) 中至少有一个异常”，但无法确定究竟是哪一个。它采用保守策略，同时丢弃这两个更新并恢复到 (w_{t-2})。

在我们的 subject-CL 中，一个任务就是一个新 EEG 个体。例如第 5、6 个任务对应两个具体受试者。回滚 6–12 个任务意味着最终模型中有 6–12 个正常个体的增量更新被撤销，不是删除硬盘上的 EEG 文件，也不是把这些个体从最终测试集合中删除。最终仍然在全部 49 个体上评价，因此被撤销个体通常会表现出没有适配或适配不足。

## 六、为什么 clean EEG 也会触发 T2T

T2T 理论假设相邻任务的干净最优参数处于相对稳定的邻域，论文的严格阈值还建立在线性回归、共享真实参数和可控噪声假设上。EEG 个体持续学习明显偏离这些条件：

- 不同个体的通道幅值、睡眠结构、类别比例和生理特征本来就会发生合法 domain shift。
- guiding model 的伪标签质量随个体显著变化，现有 full49 中一些个体伪标签 ACC 低至约 22%，另一些超过 85%。
- 深度 EEG 网络的任务 Hessian 不会严格交换，也无法共享一个精确特征基。
- 当前 EWC 的累计加权中心不总是论文公式中的上一轮参数点。

因此，大的 (d_t) 既可能来自攻击，也可能来自正常个体漂移或低质量伪标签。启用 rollback 后，正常漂移确实会被直接过滤。现有 clean full49 的 6–12 个回滚应被解释为“固定论文启发式阈值不能直接迁移到 EEG”，而不是防御成功。

一个可部署的版本不能在没有攻击时大量撤销正常个体。当前 runner 已增加 `--t2t-action monitor`：校准阶段只记录阈值穿越但不回滚。正确流程应在与攻击评估个体完全分离的 clean calibration stream 上确定阈值，使 clean false-positive rate 低于预先设定值，然后冻结阈值和所有超参数再进入攻击实验。如果 clean 与 attack 分数分布无法分开，则应得出“T2T 不适合当前 EEG 协议”的结论，而不是继续调阈值直到结果好看。

## 七、为什么使用对角 Hessian 近似

论文式 (6) 原本需要保存和运算 (p\times p) Hessian 与正则矩阵。当前 BrainUICL backbone 约有 662 万个可训练参数，完整 float32 Hessian 需要约 (6.6\text{M}^2\times4) bytes，即约 176 TB，无法实际保存。

因此当前实现用每个参数的 empirical Fisher 作为 Hessian 对角近似，把存储从 (O(p^2)) 降到 (O(p))。这不是我们自行加入的隐藏改动：ICML 2026 论文 Appendix E 明确说明其 CIFAR-10 CNN 的 T2T 实验也使用 diagonal Hessian approximation；同一附录给出了 2.5 倍阈值和长度 5 的窗口。论文理论部分还给出了线性回归下依赖噪声方差的概率阈值 (	heta_t)，但那些量在当前非线性、无标签 EEG 中无法可靠估计，所以实验实现使用启发式阈值。

## 八、为什么 Finetune 的历史正则矩阵是零

正则化 CL 的任务目标可以写成：

$$
L_t(w)+\frac{1}{2}(w-w_{t-1})^\top H_t(w-w_{t-1}).
$$

Finetune 只最小化当前任务损失 (L_t(w))，没有以 (w_{t-1}) 为中心、按历史参数重要性加权的第二项，所以在这个定义下 (H_t=0)。普通 weight decay 以零点为中心，不保存“哪些历史参数重要”，不能当成 T2T 所需的历史 (H_t)。当 (H_{t-1}=0) 时，T2T 的 (A_t) 退化，和 (B_t) 没有可用共同投影方向，因此无法原样计算论文分数。给 Finetune 增加非零历史 (H_t) 后，它已经变成某种正则化 CL，而不再是 Finetune 基线。

## 九、Robust Feature Defense 和 protected set

Robust Feature 不检测或删除任务。它假定攻击频繁出现、每次预算有界并且条件均值为零，然后主动设计正则矩阵 (H_t)，降低数据扰动到模型参数更新的最大敏感度。

把当前任务特征曲率分解成若干方向后，不同方向对攻击的脆弱程度不同。论文 minimax 解会找出攻击者最可能集中预算的方向，并把这些方向组成 protected set；系统在这些方向上重新分配更强或更合适的正则特征值，使最坏方向不再单独支配总风险。Protected set 是参数/特征方向集合，不是“被保护的数据样本集合”。

Robust Feature 理论上并非只能用于最后一层。我们选择最后线性层，是因为论文真实 CIFAR-100 实验冻结 ViT，只训练线性分类器，其闭式推导依赖线性平方损失和可共同对角化的任务 Hessian。BrainUICL 全网络是非线性、使用交叉熵且 Hessian 不交换。把完整公式直接放到全部 662 万参数上既不可计算，也没有原定理保证。当前 EEG 实现使用最后分类器前的 128 维特征协方差，把防御作为额外二次项叠加到五种 CL 方法；这是有明确边界的近似迁移，不是“只能接最后一层”的数学限制。

## 十、Robust Feature 的维度归一化是什么

论文 CIFAR-100 使用攻击总预算 (M=2000)，线性头维度是 (768\times100=76800)。如果把同一个总预算 2000 直接用于只有 (128\times5=640) 个权重的 EEG 分类器，每个参数承担的假设攻击预算会被放大 120 倍。

当前实现保持每参数预算一致：

$$
\frac{M_{\text{paper}}}{p_{\text{paper}}}
=\frac{2000}{768\times100},
\qquad
M_{\text{EEG}}
=\frac{2000}{768\times100}(128\times5)
\approx16.67.
$$

这叫“按参数维度归一化攻击预算”。它不是对 EEG 波形做归一化，也不是证明两个数据集完全等价，只是避免因为分类器大小不同而使用明显不成比例的 (M)。

## 十一、clean 实验不能证明防护，应该如何证明

对每个 victim、seed、攻击预算和攻击任务集合，至少需要四个条件：

| 条件 | 作用 |
|---|---|
| Clean | 无攻击、无防御基线 |
| Clean + Defense | 测量防御本身的 clean 代价和误报 |
| Attack | 测量攻击真实造成的退化 |
| Attack + Defense | 测量防御后还剩多少攻击退化 |

防御效果不能只比较 `Attack+Defense` 和 `Attack` 的绝对 ACC，因为防御本身可能改变 clean 轨迹。应分别计算：

$$
\Delta_{\text{attack}}
=R(\text{Attack})-R(\text{Clean}),
\qquad
\Delta_{\text{attack|defense}}
=R(\text{Attack+Defense})-R(\text{Clean+Defense}).
$$

如果使用 ACC，则符号方向相反，但仍应做同样的差分。只有当攻击确实造成稳定退化、加入防御后退化显著缩小，并且 clean 代价和误报可接受，才能说防御有证据支持。还需要至少 3 个 seeds、攻击预算/覆盖率 sweep、固定超参数和置信区间。

T2T 额外必须报告 clean FPR、attack TPR、检测延迟、触发任务和回滚任务数。Robust Feature 没有检测动作，应报告 attack-induced ACC/MF1/BWT degradation 是否缩小，以及防御梯度相对原训练梯度是否足够大。

## 十二、首轮 BrainWash 对 T2T 的实际结果

实验目录：`experiments/icml2026_cl_defenses/brainwash_probe9/`。Victim 是 EWC，使用前 9 个 seed-4321 个体，仅攻击第 9 个体 subject 80。四条件均使用 10+10 epochs、`ssl_lr=cl_lr=1e-6`、EWC 5000 和冻结学生 BN。

低预算 BrainWash 使用 3/43 个 sequence、`eps=0.005×modality_std`、5 步分类器一步双层近似。它没有形成有效攻击：无防御时旧个体 ACC 相对 clean 变化 `+0.04 pp`，最终新个体 ACC 变化 `+0.04 pp`。在相同 T2T 轨迹内，攻击任务分数从 clean 的 `1.727e6` 变为 `1.691e6`，阈值为 `1.233e7`，没有触发。

Stress BrainWash 使用 9/43 个 sequence、`eps=0.1×modality_std`、10 步；平均输入相对 (ell_2) 约 6.1%–6.6%，伪标签保持率下降到约 40%–46%。

| 条件内攻击效应 | 旧个体 ACC 变化 | 最终新个体 ACC 变化 | BWT ACC 变化 |
|---|---:|---:|---:|
| BrainWash stress 相对 Clean | -0.86 pp | -0.78 pp | -0.65 pp |
| BrainWash stress + T2T 相对 Clean + T2T | -0.77 pp | -1.17 pp | -0.89 pp |

Stress 攻击任务的 T2T 分数从 clean 的 `1.727e6` 增加到 `1.963e6`，只增加约 13.7%，仍远低于 `1.233e7` 阈值，没有触发攻击回滚。两条 T2T 轨迹都在第 6 个正常个体发生一次相同触发并回滚第 5、6 个体；这是 clean 误报，与第 9 个体 BrainWash 无关。

因此当前实验不能支持“T2T 能防 BrainWash”。低预算攻击本身无效；stress 攻击产生了可测退化，但 T2T 没有检测到，加入 T2T 后最终新个体和 BWT 的攻击退化反而更大。这个负结果与攻击条件不匹配有关：BrainWash 是范数有界的一次输入双层攻击，而论文 T2T 主要针对少数但可非常强的 shifted attack。条件不匹配可以解释结果，但不能把失败改写为成功。

## 十三、下一轮正式协议

### T2T 轨道

1. 先用 `--t2t-action monitor` 在独立 clean calibration subjects 上收集分数，不允许回滚。
2. 预先规定 clean FPR 上限，例如 1%，据此确定阈值；阈值、窗口、参数范围随后全部冻结。
3. 先测试论文对齐的少量 shifted feature attack：攻击 1/49、3/49、5/49 个体，做 1%、5%、10%、20% 相对 (ell_2) sweep。
4. 再测试 PACOL 和 BrainWash，明确标为 bounded out-of-regime stress test，并报告它们是否真的造成旧知识退化。
5. 只有攻击任务被检测、clean FPR 可接受且回滚后性能恢复，才能支持 T2T 有效。

### Robust Feature 轨道

1. 构造 frequent、bounded、conditional-zero-mean 的固定共享攻击流，不能直接把任意 proxy noise 称为 non-shifted。
2. 使用成对正负扰动或任务内中心化保证经验均值接近零，同时把协方差预算放到分类器最敏感特征方向。
3. 对 49 个任务中的 20%、50%、100% 任务攻击，并对预算做 sweep。
4. 比较五种方法的四条件结果，同时记录 protected set、额外正则梯度和最坏方向敏感度。

### Replay 与正则化共同轨道

1. 使用完全相同的 fixed shared upload 攻击所有八种方法。
2. Replay 方法额外记录污染进入 buffer 的比例、停留时间和 replay 次数；正则化方法记录 importance/anchor 污染。
3. 不直接篡改历史 buffer；若要测试 storage tampering，另开独立 replay-only 实验。
4. 主结论基于固定共享输入；每方法 adaptive white-box 结果作为单独的最坏情况附表。
