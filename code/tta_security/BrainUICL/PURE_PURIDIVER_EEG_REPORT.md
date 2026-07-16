# Labeled Pure PuriDivER-EEG 协议、实现和 49-subject 实验

本实验是独立的 labeled EEG continual learning，不使用 BrainUICL 的 guiding/teacher model、pseudo label、CPC、CEA、预训练 checkpoint、source replay 保护或 new/replay loss。代码只读取 ISRUC 预处理数据，并使用一个随机初始化的单 epoch EEG classifier。

**适用范围更正：这是 labeled upper-bound，不是 BrainUICL 的 unlabeled subject CL。** 训练流直接读取 ISRUC 人工 sleep-stage annotation 作为 observed label，因此它只能回答“原始 PuriDivER 在有标签 EEG 流上能否工作”，不能回答“无标签新个体上是否优于 BrainUICL”。原始 PuriDivER 本身要求每个流样本有一个可能被污染的 observed label；完全无标签时，第一层交叉熵和 loss-GMM 没有定义。

**观测标签从哪里来。** `true label` 是 ISRUC 数据集中的人工 sleep-stage annotation，不是 guiding model 的输出。guiding model 生成的是 pseudo label，最多只能作为 observed label，不能称为真实标签。

observed label 在 clean 和 noisy 实验中都存在：

$$
\tilde y_i=
\begin{cases}
y_i, & \text{clean stream}\\
\operatorname{Corrupt}(y_i), & \text{noisy stream}
\end{cases}
$$

模型训练和 memory 管理只能读取 observed label。true label 只用于 held-out evaluation、memory purity 和 GMM precision 统计。

三套实验的标签语义必须分开：

| 实验 | 训练时的 observed label | true label | Guiding model | 方法性质 |
|---|---|---|---|---|
| 原始 PuriDivER/CIFAR | 数据集给出的 noisy/clean label | clean test label，只用于评估 | 无 | 原始有标签 noisy-label CL |
| 当前 Labeled Pure PuriDivER-EEG | ISRUC 人工 annotation；noisy 分支再人工扰动 | ISRUC annotation，只用于 held-out 评估和诊断 | 无 | 有标签 EEG upper-bound |
| 之前 BrainUICL + PuriDivER-style hybrid | guiding/teacher 产生的 pseudo label | ISRUC annotation，只用于评估 | 有 | 无标签 EEG hybrid |

任何 guiding model 输出都只能称为 pseudo label 或 observed pseudo label，不能称为 true label。若训练代码把它放进名为 `label` 的变量，也不会改变它的统计语义。

**ER、CE 和 Puri memory 分别是什么。** CE 是 Cross Entropy，只是一种分类损失，不是一种持续学习方法。ER 是 Experience Replay：模型训练当前 online batch 时复习 replay memory，task 后再用 memory 的 observed labels 做普通 CE；本实验的 ER 使用 reservoir sampling 管理 memory，没有 GMM、relabel 或 consistency。

`Puri memory + CE replay` 是消融方法：memory 使用 PuriDivER 的 purity-diversity score 和逐次动态删除，但 task 后仍只做普通 observed-label CE replay。它没有两层 GMM，也没有 C/R/U 三分支。完整 `PuriDivER` 则同时包含 Puri memory construction 和两层 GMM robust replay。

**mini-batch 有多少数据。** PuriDivER 官方 CIFAR-10 正式配置的 batch size 是 16。每个 task 有 10,000 个样本，因此每个 task 有 625 个 online mini-batches。

本 EEG 实验将一个 30 秒 epoch 定义为一个样本，online batch size 为 128。每个 subject 约有 600 至 700 个 adaptation epochs，因此每个 subject 有 5 或 6 个 online mini-batches。49 subjects 总计执行 252 个 online mini-batches。

**模型如何初始化。** 原始 PuriDivER 的 ResNet 在 Task 0 前随机初始化一次，正式配置设置 `init_model=False`，因此模型参数跨 task 保留；每个 task 开始时重新创建 SGD optimizer。它不是每个 task 都重新随机初始化。

本 EEG 实验同样只随机初始化一次模型，使用 seed 4321，并在每个 subject 开始时重置 SGD optimizer。没有加载任何 BrainUICL checkpoint，结果 JSON 中记录：

```text
initialization.type = random_pytorch_initialization
initialization.pretrained_checkpoint = null
initialization.teacher_model = null
```

**toy GMM 的 0.13 和 0.84 如何得到。** 假设 loss 是：

$$
[0.08,0.12,0.15,0.18,0.76,0.91]
$$

GMM 使用 EM 算法。令第 `i` 个 loss 属于第 `k` 个 Gaussian 的 posterior responsibility 为：

$$
\gamma_{ik}=P(z_i=k\mid\ell_i)
$$

Gaussian 均值在 M-step 中更新为：

$$
\mu_k
=
\frac{\sum_i\gamma_{ik}\ell_i}
{\sum_i\gamma_{ik}}
$$

当两组分离明显时，posterior 接近 0 或 1，所以近似为两组的算术平均：

$$
\mu_{low}
\approx
\frac{0.08+0.12+0.15+0.18}{4}
=0.1325
$$

$$
\mu_{high}
\approx
\frac{0.76+0.91}{2}
=0.835
$$

四舍五入后就是 0.13 和 0.84。真实 GMM 还会同时迭代估计 variance 和 mixture prior，因此结果不一定严格等于算术平均。

**两层 GMM 是否是同一个 GMM。** 不是。它们只是都使用二成分 GaussianMixture，但每个 replay epoch 会独立拟合两个模型：

1. `loss-GMM` 在全部 memory 样本的逐样本交叉熵上拟合，得到 clean/noisy。
2. `uncertainty-GMM` 只在第一层判为 noisy 的样本上拟合，得到 relabel/unlabeled。

第二层的输入不是 loss，而是：

$$
U(x_i)=1-\max_c p_m(c\mid x_i)
$$

**为什么先做一次普通 SGD。** Online CL 规定流数据通常只能看一次。普通 SGD 让模型立即吸收当前 batch 的信息，同时产生有意义的当前 loss 和 feature，供 memory score 使用。没有 online update 时，随机模型的 loss 和表示不能可靠反映 purity 或 diversity。

online SGD 会立即更新模型参数：

$$
\Theta
\leftarrow
\Theta-lambda\nabla_\Theta
\frac{1}{|B|}
\sum_{i\in B}
\operatorname{CE}(p_i,\tilde y_i)
$$

从第二个 task 开始，官方代码会从 memory 采样与当前 batch 数量相近的 replay 样本，与当前 batch 拼接后做 online update。本实现跟随该代码行为；Task 0 的 online update 不使用 memory。

**交叉熵是什么，为什么使用它。** 对 one-hot 分类标签，交叉熵为：

$$
\operatorname{CE}(p_i,\tilde y_i)
=
-\sum_c
\operatorname{onehot}(\tilde y_i)_c
\log p_i(c)
=
-\log p_i(\tilde y_i)
$$

它等价于 categorical likelihood 的负对数。模型给观测标签的概率越小，loss 越大；给观测标签的概率越接近 1，loss 越接近 0。它既可微、可用于训练，也使逐样本 loss 成为“模型是否认同该标签”的代理量，所以 PuriDivER 用它同时做 SGD 和 purity 判断。

**batch 难易表示什么。** online batch average loss 为：

$$
\overline{\ell}_k
=
\frac{1}{|B_k|}
\sum_{i\in B_k}
-\log p_i(\tilde y_i)
$$

高 average loss 表示模型当前很难解释这批观测标签。原因可能是标签污染、N1 等类别本身难分类、subject domain shift、信号伪迹、类别边界模糊，或者模型尚未学会该模式。高 loss 不等于一定有噪声。

PuriDivER 使用：

$$
\alpha_k
=
0.5\min
\left(
\frac{1}{\overline{\ell}_k},1
\right)
$$

batch 越难，average loss 越高，系数越小，memory score 越强调 small-loss purity；batch 较容易时，系数最高为 0.5，允许更多考虑 diversity。

**mini-batch 后谁进入 replay buffer。** 候选集合不是“buffer 剩余空间”，而是：

$$
\mathcal M_{candidate}
=
\mathcal M_{current}\cup B_{new}
$$

如果候选数没有超过容量，全部保留。超过容量后，算法先选择 observed-label 数量最多的类别，再在该类别中删除 PuriDivER score 最大的样本，直到 memory 回到容量上限。新样本没有天然优先权，旧样本和新样本都可能被删除。

memory 物理上是一个池，但每条记录保存 observed label。选择时会按 observed label 统计和比较，因此逻辑上是 class-aware memory，不是五个完全独立的 buffer。

**warm-up 是什么。** 模型训练初期的 loss 和 confidence 不可靠。warm-up 期间只对 memory 做普通 observed-label CE，不执行两层 GMM。原始 CIFAR-10 正式配置是 255 个 replay epochs，其中前 10 个为 warm-up。本 EEG 实验为了优先覆盖 49 subjects，使用 10 个 replay epochs，其中前 2 个 warm-up；这是计算预算调整，不是论文原超参数。

**为什么对 memory 前向计算。** 每个 robust replay epoch 开始时，需要用当前模型重新得到每个 memory epoch 的 loss、softmax confidence 和 feature。该阶段使用 evaluation mode 和 no-gradient，不更新参数；它只构造 C/R/U 划分。随后使用三种目标反向传播，才更新模型参数。

三种目标解决三个不同风险：

- Clean：模型和观测标签一致，直接保留监督信息。
- Relabel：模型不认同观测标签但对另一类别很确定，用 soft target 渐进修正。
- Unlabeled：模型和标签冲突且模型也不确定，避免错误监督，只学习增强不变性。

最终目标用于直接优化同一个模型参数：

$$
\mathcal L
=
\mathcal L_{clean}
+
\mathcal L_{relabel}
+
\eta\mathcal L_{consistency}
$$

**EEG 是否可以每个 subject 后评价 old 和 new。** 可以，而且比只评价固定 old-generalization subjects 更完整。本实验将每个 subject 的 sequence 按时间顺序划分为前 70% online adaptation、后 30% held-out test，保证 sequence 不重叠。

在学习 subject `t` 前后评价其 held-out test，得到 new-subject plasticity；学习完成后重新评价所有已经学过的 subjects，形成 subject-level matrix：

$$
R_{t,j}
=
\text{学习 subject }t\text{ 后在 subject }j\text{ test 上的性能}
$$

由此计算当前 new subject 的适配增益、old-subject mean、final seen-subject mean、average seen-subject curve 和 forgetting。ACC 与 macro-F1 同时报告。

## 原论文和当前 EEG 的完整实验流程

原始 PuriDivER/CIFAR 流程：

```text
CIFAR clean train labels
        ↓ 注入 SYM/ASYM noise
observed noisy-label stream，划分为 5 个 blurry class tasks
        ↓
随机初始化一个 ResNet（后续 task 不重置模型）
        ↓
每个 task 的每个 mini-batch：online CE 更新
        ↓ 从第二个 task 起，官方代码拼接等量 memory replay
M ∪ current batch 形成 memory candidates
        ↓
按 observed class 找当前多数类
        ↓
动态计算 purity-diversity score，删除最高分样本，重复到 K
        ↓
task online stream 完成后做 memory replay
        ↓ 前 10 epochs 为普通 CE warm-up
每个后续 replay epoch：loss-GMM → C/N
        ↓
只在 N 上做 uncertainty-GMM → R/U
        ↓
Clean CE + Soft Relabel CE + Consistency
        ↓
在 clean CIFAR test set 上评价全类别 accuracy
        ↓
报告 A_last、A_avg、memory purity/diversity
```

当前 Labeled Pure PuriDivER-EEG 流程：

```text
ISRUC 人工 annotation
        ↓ clean: observed=true；noise20: 扰动 observed label
49 个 incoming subjects，每个 subject 是一个 task
        ↓
每个 subject 按 sequence 划分 70% labeled online / 30% held-out test
        ↓
随机初始化一个 compact EEG CNN，不加载 BrainUICL checkpoint
        ↓
每个 30 秒 epoch 作为一个 PuriDivER sample
        ↓
online CE + replay、动态 Puri memory、两层 GMM robust replay
        ↓
每学完一个 subject，评价当前 new subject 和全部已学习 old subjects
        ↓
报告 subject-level ACC/MF1 matrix、plasticity、curve、forgetting
```

两者在 PuriDivER 算法主体上对应，但监督前提不同于 BrainUICL：当前 EEG online 70% 部分有人工标签。

真正的 Unlabeled EEG CL 不能直接运行原始 PuriDivER。最低限度的可执行改造是：

```text
无标签 incoming subject
        ↓
guiding model 仅生成 observed pseudo labels
        ↓ ISRUC annotation 完全隐藏，只留给 held-out evaluation
单 student 执行 online CE、Puri memory、loss-GMM、uncertainty-GMM 和 C/R/U replay
        ↓
评价 old/new subject matrix
```

这应命名为 `Pseudo-label PuriDivER-EEG`，不是 pure PuriDivER。为了尽量少使用 BrainUICL，可以只保留 guiding model 的一次性 pseudo-label 生成，删除 teacher entropy、teacher/student JS、CEA、CPC continual objective、source-label protection 和 BrainUICL new/replay loss。若连 guiding model 也删除，则没有 observed label，第一层 loss-GMM 无法计算，研究方法必须改成无监督聚类或纯 consistency 方法，不再是 PuriDivER。

## 实现与运行配置

独立代码：

```text
experiments/pure_puridiver_eeg.py
tests/test_pure_puridiver_eeg.py
```

共同配置：

```text
subjects: 49
seed: 4321
model: randomly initialized compact 1-D EEG CNN
sample: one 30-second epoch
train/test: 70%/30% disjoint sequences per subject
online batch: 128 epochs
memory: 1000 epochs
replay epochs: 10
warm-up epochs: 2
optimizer: SGD, lr=0.01
teacher/CPC/CEA/checkpoint: none
```

49 subjects 的 online mini-batches 为每 subject 5 至 6 个，共 252 个。

## 49-subject clean-stream 结果

| 方法 | Final ACC | Final MF1 | Curve ACC | Curve MF1 | ACC Forgetting | MF1 Forgetting | New ACC gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| ER | **0.7133** | **0.5952** | **0.7298** | **0.6241** | **0.0643** | **0.0648** | **0.0712** |
| Labeled PuriDivER | 0.6700 | 0.5420 | 0.6262 | 0.5116 | 0.0813 | 0.0725 | 0.0617 |

clean stream 中所有 memory labels 都正确，二者 memory purity 均为 1.0。但 Pure PuriDivER 的两层 GMM 平均仍将 memory 划为：

```text
Clean:     91.89%
Relabel:    3.72%
Unlabeled:  4.39%
```

这意味着约 8.1% 标签正确但困难的 EEG epoch 被当成可疑样本。Pure PuriDivER 的 final ACC 比 ER 低 4.33 个百分点，MF1 低 5.31 个百分点，ACC forgetting 高 1.70 个百分点。结论是：它在 clean subject-shift EEG 上保留了较好的持续学习能力，但存在明确的净化和强制类别均衡成本，仍弱于普通 ER。

## 49-subject、20% observed-label noise 结果

| 方法 | Final ACC | Final MF1 | Curve ACC | Curve MF1 | ACC Forgetting | Final memory purity |
|---|---:|---:|---:|---:|---:|---:|
| ER + reservoir | 0.6115 | 0.4982 | 0.6023 | 0.5002 | **0.0526** | 0.792 |
| Puri memory + CE replay | 0.6358 | 0.5485 | **0.6448** | **0.5330** | 0.0973 | **0.976** |
| Labeled PuriDivER | **0.6733** | **0.5698** | 0.6262 | 0.5218 | 0.0748 | 0.971 |

20% 标签污染确实破坏 ER：final ACC 从 clean 的 0.7133 降到 0.6115。PuriDivER memory construction 将最终 purity 提高到 97% 以上，比 reservoir 高约 18 个百分点。

更高 memory purity 转化成了更高 CL accuracy。完整 PuriDivER final ACC 比 noisy ER 高 6.19 个百分点，MF1 高 7.16 个百分点；相对 clean PuriDivER 的 final ACC 还高 0.33 个百分点，说明在该 seed 下基本抵消了 20% 标签污染。代价是 ACC forgetting 比 noisy ER 高 2.22 个百分点。

memory-only 消融的 curve ACC/MF1 最高，说明精确的 purity-diversity memory construction 本身贡献很大；加入 C/R/U robust replay 后，final ACC/MF1 和 forgetting 更好，但全程平均 curve 略低。这是 final stability 与中间适应性能之间的权衡。

20% noisy run 中，GMM 平均划分为：

```text
Clean:     93.81%
Relabel:    2.91%
Unlabeled:  3.28%
```

clean set 的平均真实 label precision 为约 97.86%，说明第一层 GMM 与精确 memory construction 形成了有效的净化闭环。relabel set 中 observed label 本来就正确的比例约 42.26%，即约 57.74% 的 relabel 样本确实带有错误 observed label，说明第二层 GMM 富集了需要修正的样本。

## 当前结论

本次运行已经完成 49 个 subjects，而不是只做 smoke/probe。它回答了“尽量不用 BrainUICL、只使用 PuriDivER 后 EEG CL 是否仍好”的问题：在当前忠实但计算缩减的迁移中，答案是可以保持较好性能，但 clean stream 比同预算 ER 略差；在 20% 标签污染下明显优于 ER。

PuriDivER 对 noisy replay memory 的净化非常明显，并提高了 subject-incremental EEG 的 final ACC/MF1。但 clean-stream 成本也说明 high loss 在 EEG 中同时表示 label noise、N1 难度、个体域偏移和信号质量，GMM 会误处理一部分正确困难样本；class-balanced memory 也会改变自然睡眠阶段分布。

实现必须在每删除一个候选样本后重新计算当前多数类别内的 similarity、loss z-score 和 score。曾测试过每个 incoming batch 只静态计算一次 score 的近似写法，它会显著降低 clean/noisy 性能和 memory purity，因此不属于可接受的 PuriDivER 实现。

当前结论基于一个随机 seed 和 10 replay epochs。下一步合理改进应作为 EEG-specific PuriDivER 变体明确报告，例如 class-conditional loss normalization、subject-aware GMM、保留自然类别先验、减少 clean stream 上的一致性权重；不能把这些修改称为原始 PuriDivER。

正式结果目录：

```text
experiments/rttdp_brainuicl_runs/full49_pure_er_clean_lr001_e10_seed4321/
experiments/rttdp_brainuicl_runs/full49_pure_puridiver_clean_exactscore_lr001_e10_seed4321/
experiments/rttdp_brainuicl_runs/full49_pure_er_noise20_lr001_e10_seed4321/
experiments/rttdp_brainuicl_runs/full49_pure_memory_erreplay_noise20_exactscore_lr001_e10_seed4321/
experiments/rttdp_brainuicl_runs/full49_pure_puridiver_noise20_exactscore_lr001_e10_seed4321/
```
