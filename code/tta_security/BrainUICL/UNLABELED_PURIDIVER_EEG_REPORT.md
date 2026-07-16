# Unlabeled Pseudo-label PuriDivER-EEG

本实验对应真实部署前提：incoming EEG subjects 没有人工 annotation。它不是 BrainUICL，也不是 labeled pure PuriDivER，而是原始 PuriDivER 在无标签 target stream 上的最小必要改造。报告先给出 frozen guide + guide-copy student 主实验，随后单独报告 CPC-dynamic guide + random student 扩展实验；两种设置不能混为同一个控制变量。

## 方法边界

原始 PuriDivER 必须有 observed label，才能计算：

$$
\ell_i=-\log p_s(\tilde y_i\mid x_i)
$$

完全无标签时该 loss 没有定义。因此本实验保留一个 source guiding model，只用它生成 observed pseudo label：

$$
\tilde y_i
=
\arg\max_c p_g(c\mid x_i)
$$

guiding output 不是 true label。默认 target adaptation loader 不打开 ISRUC label 文件；annotation 只由 benchmark evaluator 读取。若显式设置 `--benchmark-annotation-diagnostics`，benchmark 还会读取隐藏标签计算 pseudo/memory purity，但这些字段不参与训练、选择或梯度。

guiding model 必须来自历史有标签 source cohort 或外部预训练模型；如果真实系统中连 source/external supervision 都不存在，就无法得到具有 W/N1/N2/N3/REM 语义的 guiding classifier，这不是 PuriDivER 可以解决的问题。

下述 frozen-guide 主实验不包含以下 BrainUICL 部分：

```text
CPC target adaptation
CEA
teacher entropy
teacher/student JS divergence
confidence threshold
confident-epoch count filter
source replay protection
BrainUICL new/replay alpha loss
BrainUICL pretrained checkpoint
```

guide confidence 只写入结果作为诊断，不参与候选接受、loss 权重或 memory selection。每个 target adaptation epoch 都获得一个 argmax pseudo label，接受率固定为 100%。

后文的随机 student 扩展实验会有意加入一种独立实现的 CPC-style target guide adaptation，但仍不使用 CEA、confidence gate、source replay protection 或 BrainUICL joint-update objective。

## 完整流程（frozen guide + guide-copy student）

```text
24 labeled source subjects
        ↓
离线训练 compact source guiding model
        ↓ 在 6 个 source validation subjects 上选择最佳 MF1
冻结 guiding model
        ↓
49 个无标签 incoming target subjects
        ↓ 每个 subject 70% sequences 用于 adaptation
guide 为每个 30 秒 epoch 生成 argmax observed pseudo label
        ↓ 不做 confidence gate
student online CE update
        ↓ 从第二个 subject 起拼接等量 target memory replay
PuriDivER 动态 purity-diversity memory construction
        ↓
每个 subject 后进行 10 epochs target memory replay
        ↓ 前 2 epochs 是普通 CE warm-up
loss-GMM 将 memory 分为 C/N
        ↓
N 上的 uncertainty-GMM 将样本分为 R/U
        ↓
Clean CE + Soft Relabel CE + Consistency
        ↓
在从未进入 adaptation/memory 的 30% held-out sequences 上评价
        ↓
更新全部已学习 subjects 的 ACC/MF1 matrix 和 forgetting
```

本节主实验中，guiding model 在 target CL 期间始终冻结。student 从 guiding weights 初始化，但之后只由 PuriDivER 更新。target memory 不包含受保护的 source true-label 样本。动态 guide 与随机 student 的生命周期见后文独立扩展实验。

## 两个模型、三类 loss 与一次 subject 更新的精确顺序（主实验）

整个系统有两个独立模型对象：

```text
guiding model G
    source 阶段训练，target 阶段永久冻结
    唯一职责：为每个 incoming target epoch 生成一次 argmax pseudo label

student model S
    CL 开始时新建模型对象，并复制 G 的全部权重
    target continual learning 中唯一被 SGD 更新的模型
    学完 subject t 后的权重继续作为 subject t+1 的初始权重
```

因此 student 不是随机开始，也不是直接修改冻结 guide。它是“从 source-pretrained guide 权重初始化的可训练副本”。每个 subject 会重新创建 SGD optimizer，但不会重置 student 参数。

假设当前 subject 是 `t`，执行顺序如下。

第一步，guide 对该 subject 的 70% adaptation epochs 前向一次：

$$
\tilde y_i
=
\arg\max_c p_g(c\mid x_i)
$$

所有 pseudo-labeled epochs 都保留，没有 confidence threshold。之后 guide 不再参与该 subject 的 student loss，也不会更新。

第二步，student 做 online update。Task 0 只使用当前 batch；从第二个 subject 开始，当前 batch 与从 target memory 随机采样的等量 replay epochs 拼接：

$$
\mathcal L_{online}
=
\frac{1}{|B_t\cup B_M|}
\sum_{i\in B_t\cup B_M}
-\log p_s(\tilde y_i\mid x_i;\theta_s)
$$

然后 SGD 更新 student：

$$
\theta_s
\leftarrow
\theta_s
-
\eta\nabla_{\theta_s}\mathcal L_{online}
$$

这个 batch average loss 只用于 online gradient 和计算 adaptive diversity coefficient：

$$
\alpha_k
=
0.5\min
\left(
\frac{1}{\mathcal L_{online}},1
\right)
$$

第三步，更新 Puri memory。候选集合为当前 memory 与刚到达的 current batch：

$$
\mathcal M_{cand}
=
\mathcal M\cup B_t
$$

如果超过 1000 个 epochs，使用刚完成 online SGD 的 student 重新对全部 candidates 前向，重新计算每个候选的逐样本 CE 和 feature：

$$
\ell_i^{mem}
=
-\log p_s(\tilde y_i\mid x_i;\theta_s)
$$

它与 online CE 使用相同公式，但不是直接复用 online batch 的平均 loss。online loss 是一个 batch scalar；memory loss 是对所有 candidates 新计算的一组逐样本值。算法在当前 observed-pseudo-label 多数类中计算：

$$
S_i
=
(1-\alpha_k)z(\ell_i^{mem})
+
\alpha_k z(\operatorname{similarity}_i)
$$

删除 score 最大的样本，然后重新计算当前多数类的 z-score、similarity 和 score，直到 memory 回到 1000。该步骤只做 forward 和选择，不做 backward，不更新 student。

完成该 subject 的全部 online mini-batches 后，进入 10 个 memory replay epochs。前 2 个 warm-up epochs 对 memory pseudo labels 做普通 CE，并用 SGD 更新 student。

从第 3 个 replay epoch 开始，先用当前 student 对同一个 memory 做 evaluation/no-gradient forward。这里第三次计算逐样本 CE：

$$
\ell_i^{gmm}
=
-\log p_s(\tilde y_i\mid x_i;\theta_s)
$$

它不是 online loss，也不是 memory insertion 时缓存的 loss。因为 student 已经变化，必须在每个 replay epoch 重新计算。

第一层 loss-GMM 在全部 memory epochs 上拟合，低均值 component 的 posterior 为：

$$
w_i
=
P(g_{low}\mid\ell_i^{gmm})
$$

得到：

$$
\begin{aligned}
\mathcal C
&=
\{i:w_i\ge0.5\}\\
\mathcal N
&=
\{i:w_i<0.5\}
\end{aligned}
$$

只对 noisy set 计算 student uncertainty：

$$
u_i
=
1-
\max_c p_s(c\mid x_i)
$$

第二个、独立的二成分 GMM 在 noisy-set uncertainty 上拟合。令 low-uncertainty component posterior 为：

$$
q_i
=
P(u_{low}\mid u_i)
$$

得到：

$$
\begin{aligned}
\mathcal R
&=
\{i\in\mathcal N:q_i\ge0.5\}\\
\mathcal U
&=
\{i\in\mathcal N:q_i<0.5\}
\end{aligned}
$$

C、R、U 不是三个新的持久 buffer，也不会把样本从 memory 中移动出去。它们只是当前 replay epoch 对同一 memory 的三个临时 mask；student 更新后，下一个 epoch 会重新计算，某个 epoch 可以从 U 变为 R 或 C。

三个集合分别产生训练信号。Clean set 暂时相信 guide pseudo label：

$$
\mathcal L_C
=
\sum_{i\in\mathcal C}
\operatorname{CE}
\left(
p_s(x_i),\tilde y_i
\right)
$$

Relabel set 的 student 很确定但不认同 observed pseudo label，因此构造 soft target：

$$
\hat y_i
=
q_i\operatorname{stopgrad}(p_s(x_i))
+
(1-q_i)\operatorname{onehot}(\tilde y_i)
$$

并使用 soft cross-entropy：

$$
\mathcal L_R
=
-\sum_{i\in\mathcal R}
\sum_c
\hat y_i(c)
\log p_s(c\mid x_i)
$$

Unlabeled set 既不相信 pseudo label，也不相信当前 hard prediction，只约束同一 EEG epoch 的 weak/strong augmentation 输出接近：

$$
\mathcal L_U
=
\sum_{i\in\mathcal U}
\left\|
p_s(a_{strong}(x_i))
-
\operatorname{stopgrad}
\left(
p_s(a_{weak}(x_i))
\right)
\right\|_2^2
$$

实现按 C/R/U 的实际样本数合并并归一化：

$$
\mathcal L_{replay}
=
\frac{
\mathcal L_C+\mathcal L_R+\eta\mathcal L_U
}{
|\mathcal C|+|\mathcal R|+|\mathcal U|
}
$$

对该 loss 做 backward 和 SGD，更新的仍然只有 student。guide 不更新，C/R/U mask 本身也不是参数。完成 10 个 replay epochs 后，用更新后的 student 在当前和全部旧 subjects 的 held-out test sequences 上评价，然后进入下一个 subject。

默认训练 pool 用 `-1` 表示 annotation unavailable，可以在没有 target label 文件的训练侧运行。正式研究表为了计算 pseudo-memory purity，显式开启 `--benchmark-annotation-diagnostics`，使 benchmark pool 携带隐藏 annotation；单元测试已经验证，改变这些隐藏 annotation 不会改变 guide pseudo labels、额外噪声位置、Puri memory selection 或 C/R/U。真实部署应关闭该开关。

## 与 BrainUICL 的流程对比（主实验）

| 环节 | BrainUICL | 当前 Pseudo-label PuriDivER-EEG |
|---|---|---|
| 初始模型 | 加载 BrainUICL source-pretrained FeatureExtractor、Transformer、MLP checkpoint | 离线训练 compact source guide；student 复制其权重 |
| Target guide | 每个 subject 从当前模型克隆 teacher，并用无标签 CPC 在当前 subject 上更新 | source guide 在全部 target subjects 上永久冻结 |
| Target pseudo label | subject-adapted teacher 预测 | frozen source guide argmax |
| 额外置信度过滤 | confidence threshold 和 confident-epoch count 决定 buffer 写入 | 无；所有 pseudo-labeled epochs 进入候选流 |
| Replay 数据 | labeled source replay 加 accepted target pseudo replay | 只有 target pseudo replay，不保护 source labels |
| CL 主目标 | new/replay pseudo-label finetune 加 CEA feature alignment | Online CE 加 Puri C/R/U robust replay |
| 污染判断 | 原始 BrainUICL 没有 loss-GMM/uncertainty-GMM | 两层 PuriDivER GMM |
| 被更新模型 | 主 BrainUICL blocks；subject teacher 还执行 CPC 更新 | 只有 student；frozen guide 永不更新 |
| 旧任务评价 | 固定的 never-adapted old-generalization subject set | 所有已经按顺序学过的 old-subject held-out tests |
| 新任务评价 | 当前 subject 全量/原协议评价 | 当前 subject 独立 30% held-out sequences |

BrainUICL 的 guiding teacher 是每个 subject 都进行 CPC 自监督适配的动态模型；本节主实验的 guide 是固定 source model。这是为了满足“不使用 BrainUICL，只保留必要 pseudo-label source”的要求，但也使 target pseudo-label quality 取决于 source model 的跨 subject 泛化。

本节主实验的 student 与 BrainUICL 一样都从 source knowledge 开始，而不是在 target CL 开始时随机初始化。区别是 BrainUICL student 使用其预训练多模块 checkpoint；当前 student 使用 compact source guide 的权重副本。后文扩展实验则将 student 独立随机初始化。

此前 BrainUICL 报告的 final ACC/MF1 是固定 old-generalization subjects 上的指标，而本报告的 Final ACC/MF1 是 49 个已经学习 subjects 各自 held-out test 的最终平均值。训练/测试划分、backbone 和评价集合均不同，因此 `0.6781` 不能直接与此前 BrainUICL 的 `0.6569/0.6231` 做胜负比较。公平比较需要让 BrainUICL 也使用同一 70% adaptation、30% held-out subject matrix，并报告同样的 final/curve/forgetting。

## 对比方法

| 方法 | Target 更新 | Memory | Task 后训练 |
|---|---|---|---|
| Frozen guide | 无 | 无 | 无 |
| Pseudo-ER | 全量 guide pseudo labels | Reservoir，1000 epochs | 普通 pseudo-label CE |
| Pseudo-Puri memory + CE | 全量 guide pseudo labels | 动态 Puri purity-diversity，1000 epochs | 普通 pseudo-label CE |
| Pseudo-PuriDivER | 全量 guide pseudo labels | 动态 Puri purity-diversity，1000 epochs | 两层 GMM 与 C/R/U replay |

`CE` 是 Cross Entropy loss；`ER` 是 Experience Replay 方法；`Puri memory + CE` 是用于分离 memory construction 贡献的消融，不是完整 PuriDivER。

## 配置（frozen guide + guide-copy student）

```text
seed: 4321
source train/validation subjects: 24/6
target subjects: 49
guide: compact 1-D EEG CNN, source-supervised, frozen on target
guide training: 15 epochs, Adam lr=0.001
best source validation ACC/MF1: 0.7048/0.6836
student initialization: guide weights
target split: 70% unlabeled adaptation / 30% held-out test by sequence
target online batch: 128 epochs
target memory: 1000 epochs
target replay: 10 epochs, first 2 warm-up
target optimizer: SGD lr=0.01
confidence filter: none
target annotation used in training/selection: false
```

guide checkpoint 缓存在：

```text
/home/undefined/Disk/ai-storage/BrainUICL/model_parameter/PseudoPuriDivER/compact_guide_seed4321.pt
```

## 49-subject 结果

guide 在 49 个 target adaptation splits 上的平均 pseudo-label ACC/MF1 为：

$$
71.89\% / 65.74\%
$$

| 方法 | Final ACC | Final MF1 | Curve ACC | Curve MF1 | ACC Forgetting | New ACC gain | Final memory purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen guide | **0.7023** | **0.5891** | **0.6978** | **0.5915** | **0.0000** | 0.0000 | N/A |
| Pseudo-ER | 0.7004 | 0.5823 | 0.6965 | 0.5881 | 0.0242 | -0.0061 | 0.725 |
| Pseudo-Puri memory + CE | 0.6712 | 0.5735 | 0.6893 | 0.5773 | 0.0568 | -0.0090 | **0.817** |
| Pseudo-PuriDivER | 0.6781 | 0.5784 | 0.6826 | 0.5575 | 0.0708 | -0.0162 | 0.807 |

Pseudo-PuriDivER 将最终 target memory pseudo-label purity 从 Pseudo-ER 的 72.5% 提高到 80.7%，说明 Puri filtering 确实过滤了部分 guide 错误。

但净化没有转化为更高 CL 性能：

```text
Pseudo-PuriDivER vs Pseudo-ER:
Final ACC:  -2.23 percentage points
Final MF1:  -0.39 percentage points
Forgetting: +4.66 percentage points

Pseudo-PuriDivER vs Frozen guide:
Final ACC:  -2.42 percentage points
Final MF1:  -1.06 percentage points
```

四种方法的平均 new-subject gain 均不为正；Pseudo-PuriDivER 为 `-1.62` ACC 百分点。这说明 source guide 已有较强跨 subject 能力，而基于其自身错误 pseudo labels 的 target self-training 发生 confirmation bias，总体没有产生正适配。

## GMM 诊断

Pseudo-PuriDivER 在所有非 warm-up replay epochs 上的平均划分为：

```text
Clean:     93.14%
Relabel:    6.15%
Unlabeled:  0.71%
```

Clean set 的平均 pseudo-label precision 为 87.88%，说明 loss-GMM 对 memory 中的错误标签有一定过滤能力。Relabel set 中 observed pseudo label 正确的比例仍为 59.24%，即只有约 40.76% 确实是错误伪标签。第二层 GMM 没有把 relabel 集合充分富集为错误标签，soft relabel 会修改不少原本正确但困难的 epoch。

Subject 25 的 guide pseudo-label accuracy 只有 37.74%。当错误是 guide 高度自洽的系统性错误时，student 对该 pseudo label 的 loss 可能同样较低，small-loss GMM 无法识别。这与此前高置信输入投毒下 PuriDivER 失败的机制一致。

## BrainUICL-compatible old/new 评测

为进行近似直接比较，新增 `--evaluation-protocol brainuicl`。该协议与前面的严格 70%/30% held-out subject matrix 不同：

```text
old stability:
    固定 seed4321 划分出的 19 个 old_generalization subjects
    在 CL 前和每个 new subject 后重新评价

new plasticity:
    当前 new subject 的全部 sequences 用于无标签适配
    同一批 sequences 的 annotation 只在评价时读取
    记录 initial model / adaptation 前 / adaptation 后

metrics:
    ACC、MF1、AAA、AAF1、FR
    公式与 BrainUICL utils/util.py 一致
```

其中：

$$
\operatorname{AAA}
=
\frac{1}{T+1}
\sum_{t=0}^{T}A_t^{old}
$$

$$
\operatorname{FR}
=
\frac{|A_0^{old}-A_T^{old}|}{A_0^{old}}
$$

固定 old 集结果：

| 方法 | Initial old ACC | Final old ACC | Final old MF1 | AAA | AAF1 | FR |
|---|---:|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.7025 | 0.6569 | 0.6231 | 0.6934 | 0.6685 | 0.0649 |
| Frozen compact guide | **0.7292** | **0.7292** | **0.7206** | **0.7292** | **0.7206** | **0.0000** |
| Compact Pseudo-ER | 0.7292 | 0.7301 | 0.7174 | 0.7275 | 0.7157 | 0.0012 |
| Compact Pseudo-PuriDivER | 0.7292 | 0.6950 | 0.7014 | 0.7059 | 0.6784 | 0.0469 |

49 个 new subjects 的平均 plasticity：

| 方法 | Initial ACC | Before ACC | After ACC | Initial MF1 | Before MF1 | After MF1 |
|---|---:|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.6464 | 0.6304 | 0.6182 | 0.5568 | 0.5549 | 0.5548 |
| Frozen compact guide | **0.7138** | **0.7138** | **0.7138** | **0.6692** | **0.6692** | **0.6692** |
| Compact Pseudo-ER | 0.7138 | 0.7111 | 0.7080 | 0.6692 | 0.6627 | 0.6595 |
| Compact Pseudo-PuriDivER | 0.7138 | 0.6807 | 0.6771 | 0.6692 | 0.6207 | 0.6062 |

按绝对值看，Pseudo-PuriDivER 相比 BrainUICL：

```text
Final old ACC: +3.81 percentage points
Final old MF1: +7.83 percentage points
AAA:           +1.25 percentage points
AAF1:          +0.99 percentage points
FR:            -1.80 percentage points
New after ACC: +5.88 percentage points
New after MF1: +5.14 percentage points
```

但这些差异不能全部归因于 PuriDivER。Compact source guide 的初始能力已经高于 BrainUICL：old initial ACC 高 2.68 个百分点，new initial ACC 高 6.74 个百分点，new initial MF1 高 11.24 个百分点。两者 backbone、source training 和 guide policy 也不同。

看相对变化更合理：BrainUICL 的 fixed-old ACC 从 0.7025 降到 0.6569，Pseudo-PuriDivER 从 0.7292 降到 0.6950；Puri 的相对 FR 更低。另一方面，Pseudo-PuriDivER 的 new MF1 从 before 0.6207 降到 after 0.6062，当前 subject 适配仍有负迁移；Pseudo-ER 和 frozen guide 更稳定。

因此该结果支持“当前 Pseudo-PuriDivER 的 old-task 保持能力与 BrainUICL 近似且绝对指标更高”，但不支持“PuriDivER 算法本身已经公平击败 BrainUICL”。严格算法比较需要相同 backbone、相同 source checkpoint 和相同 guide 更新策略，只替换 CL objective；这与本实验“不使用 BrainUICL 模型和 CL 组件”的约束存在控制变量冲突。

BrainUICL-compatible 正式结果：

```text
experiments/rttdp_brainuicl_runs/full49_unlabeled_frozen_guide_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_er_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_brainuicl_eval_seed4321/
```

## 动态 CPC guide + 随机 student 扩展实验

### 设计动机与模型生命周期

原始有标签 PuriDivER 的分类模型在实验开始时随机初始化一次，随后跨 task 保留参数。无标签 EEG 仍需要一个具备 W/N1/N2/N3/REM 语义的 source guide 来产生 pseudo label，但负责持续学习的 student 不必复制 guide。因此本扩展将两个模型显式解耦：

```text
source guide
    在 24 个有标签 source subjects 上离线训练
    每个 new subject 到达时，只用该 subject 的无标签 EEG 做 3 epochs CPC
    CPC 累积更新 guide encoder；sleep-stage classifier 始终冻结
    CPC 完成后，为该 subject 的全部 epochs 生成 argmax pseudo labels

student
    seed 4321 + 17 独立随机初始化一次
    不复制 guide 权重，也不在 task boundary 重置
    只由 online CE、Puri memory 和 C/R/U replay 持续更新
```

这组实验仍然不使用 target annotation、confidence gate、CEA、source replay protection 或 BrainUICL joint-update objective。人工标签只用于 full-subject old/new 评价；正式报告另显式开启 benchmark diagnostics 计算 purity。`brainuicl` 评价协议固定 19 个 never-adapted old subjects，并按相同顺序处理 49 个 new subjects。

正式配置为：

```text
seed: 4321
guide policy: cumulative CPC dynamic
guide CPC: 3 epochs, Adam lr=0.0001, sequence batch=8
CPC prediction steps / temperature: 3 / 0.1
student initialization: random once, then continual
target candidates: all guide argmax epochs
memory: 1000 epochs
replay: 10 epochs, first 2 CE warm-up
evaluation: BrainUICL-compatible old/new, 49 new subjects
```

### Fixed-old 稳定性与长期学习

表中每个单元格均为 `ACC / MF1`。`Signed endpoint change` 是 final 减 initial，正值代表学习，负值才代表下降。

| 方法 | Initial old | Final old | AAA / AAF1 | Last-10 old | Signed endpoint change |
|---|---:|---:|---:|---:|---:|
| BrainUICL clean | 0.7025 / 0.6880 | 0.6569 / 0.6231 | 0.6934 / 0.6685 | 0.6969 / 0.6714 | -4.56 / -6.49 points |
| Frozen guide + guide-copy Pseudo-PuriDivER | **0.7292 / 0.7206** | **0.6950 / 0.7014** | **0.7059 / 0.6784** | **0.7015 / 0.7004** | -3.42 / -1.92 points |
| CPC-dynamic guide + random student | 0.3204 / 0.1024 | 0.6940 / 0.6645 | 0.6224 / 0.5791 | 0.6432 / 0.5857 | **+37.36 / +56.21 points** |

随机 student 的 fixed-old ACC 从 `0.3204` 上升到 `0.6940`，最终 ACC 与 guide-copy Pseudo-PuriDivER 只差 `0.10` 个百分点，说明它能够从 guide pseudo labels 和持续 replay 中学到有效的睡眠阶段分类能力。但随机冷启动显著拉低了整个轨迹：相对 guide-copy 版本，AAA/AAF1 低 `8.35/9.93` 点，最终 MF1 低 `3.69` 点。

代码为兼容 BrainUICL 仍输出：

```text
FR = |initial old ACC - final old ACC| / initial old ACC = 1.16595
```

这里 final 高于 initial，所以 `1.16595` 表示相对初值提高 `116.60%`，不是“遗忘 116.60%”。正式结果同时记录 signed `old_acc_change=+0.37359`；本组不把 FR 当作遗忘指标。

相对 BrainUICL，随机 student 的 final old ACC/MF1 高 `3.71/4.14` 点，但 AAA/AAF1 低 `7.10/8.94` 点，Last-10 old ACC/MF1 也低 `5.37/8.58` 点。因此最终单点接近或更高，不代表完整持续学习轨迹优于 BrainUICL。

### New-subject plasticity

| 方法 | Initial ACC / MF1 | Before ACC / MF1 | After ACC / MF1 | After - Before |
|---|---:|---:|---:|---:|
| BrainUICL clean | 0.6464 / 0.5568 | 0.6304 / 0.5549 | 0.6182 / 0.5548 | -1.22 / -0.01 points |
| Frozen guide + guide-copy Pseudo-PuriDivER | **0.7138 / 0.6692** | **0.6807 / 0.6207** | **0.6771 / 0.6062** | -0.36 / -1.45 points |
| CPC-dynamic guide + random student | 0.3019 / 0.0944 | 0.6166 / 0.5445 | 0.6629 / 0.5834 | **+4.63 / +3.88 points** |

这里的 `Initial` 是同一个尚未接触 target stream 的随机 student 在每名 new subject 上的结果；`Before` 则会随前面 subjects 的训练逐步提高。不能把随机 Initial 与 source-pretrained BrainUICL Initial 当作同起点比较。

对新方法自身的 49 个 `After - Before` 配对差：

| Metric | Mean gain | 95% paired bootstrap CI | Positive / ties | Wilcoxon p |
|---|---:|---:|---:|---:|
| ACC | +4.63 points | [+1.70, +7.62] | 31 / 2 of 49 | 0.0106 |
| MF1 | +3.88 points | [+0.80, +7.05] | 28 / 0 of 49 | 0.0449 |

因此最稳妥的正面结论是：在本 seed 的 subject-level 配对分析中，随机 student 的当前个体适配恢复为正 plasticity。相比之下，frozen-guide + guide-copy Pseudo-PuriDivER 的平均 After-Before 为 `-0.36/-1.45` 点。

所有 paired bootstrap CI 都对 49 个 subject-level 配对差独立使用 `default_rng(4321)` 做 20,000 次有放回抽样，并取 percentile 2.5/97.5；Wilcoxon 为双侧检验。

这些 subjects 共用同一条顺序学习轨迹，并不是 49 个独立随机 seed，因此 CI 和 p 值用于描述该流内的一致性，不能替代多 seed 复现。

新方法的 New-after 均值比 BrainUICL 高 `4.46` ACC 和 `2.86` MF1 点，但 subject-wise 差异高度异质：

| Metric | New method - BrainUICL | 95% paired bootstrap CI | Wins | Wilcoxon p |
|---|---:|---:|---:|---:|
| ACC | +4.46 points | [-1.95, +11.31] | 21/49 | 0.7374 |
| MF1 | +2.86 points | [-3.96, +10.13] | 20/49 | 0.7599 |

两个置信区间都跨 0，Wilcoxon 也不显著，所以不能声称 new-subject 性能已经显著超过 BrainUICL。

### Dynamic guide、memory 与 GMM 诊断

在完全相同的 49-subject 顺序上，dynamic CPC guide 与 frozen source guide 的 paired pseudo-label 对比如下：

| Metric | Frozen guide | CPC-dynamic guide | Paired gain | 95% CI | Wins / ties | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Pseudo-label ACC | 0.7138 | **0.7539** | +4.01 points | [+2.46, +5.59] | 39 / 1 | 0.000034 |
| Pseudo-label MF1 | 0.6693 | **0.6946** | +2.54 points | [+0.89, +4.16] | 34 / 0 | 0.0030 |

CPC 是整体平均改善，不是每名 subject 都改善。例如 Subject 25 的 pseudo-label ACC 从 `39.11%` 升到 `61.22%`，而 Subject 17 下降约 `9.88` 点。它也只更新 encoder，classifier 参数在每个 subject 后均通过运行时断言验证为未改变。

完整流处理 `42,960` 个 target epochs，最终 memory 为 `1000`，五类各 `200` 个 epoch，事后 pseudo-label purity 为 `86.60%`。非 warm-up 的 392 次 GMM replay 平均分区为：

| Clean | Relabel | Unlabeled | Clean precision | Relabel 中 observed-label precision |
|---:|---:|---:|---:|---:|
| 93.58% | 3.35% | 3.07% | 88.81% | 48.87% |

Relabel observed-label precision 为 `48.87%`，即约 `51.13%` 的 relabel 样本确实是错伪标签；相比 frozen-guide 主实验，它对错误标签的富集更明显。不过 memory purity 同时受 guide 质量、随机 student features 和 Puri selection 影响，不能把 `86.60%` 的变化单独归因于 CPC。

### 短程两因素 probe 与结论边界

正式 49-subject 主运行同时改变了 guide policy 和 student initialization，不能仅凭它分离两者贡献。已有 5-subject probe 提供短程诊断：

| Guide / student | Final old ACC / MF1 | AAA / AAF1 |
|---|---:|---:|
| Frozen / random | 0.5932 / 0.5406 | 0.4619 / 0.3888 |
| CPC-dynamic / random | 0.6249 / 0.6128 | 0.5298 / 0.4350 |
| CPC-dynamic / guide-copy | **0.7177 / 0.6192** | **0.7100 / 0.6690** |

这些 probe 表明 CPC 对随机 student 的前 5 个 subjects 有帮助，而 guide-copy 消除了早期冷启动；但它们只有 5 subjects，不能替代完整两因素 49-subject 消融。

本扩展支持以下结论：

1. 无标签 EEG 中可以把 PuriDivER student 随机初始化，并通过 source guide pseudo labels 持续学起来；随机的是 student，不是负责输出语义类别的 guide classifier。
2. guide 可以不冻结。累积 CPC 更新 guide encoder 在本实验中显著提高平均伪标签质量，同时分类头必须冻结以保留类别语义。
3. random + CPC 恢复了显著正的当前个体 plasticity，最终 old ACC 接近 guide-copy PuriDivER；代价是冷启动造成较低 AAA/AAF1、Last-10 和最终 MF1。
4. 与 BrainUICL 的 New-after 均值差异没有通过 paired significance test，因此目前不能宣称显著优于 BrainUICL。

### 复现命令与产物

```bash
/home/undefined/Disk/ai-storage/BrainUICL/envs/brainuicl/bin/python \
  experiments/unlabeled_puridiver_eeg.py \
  --method puridiver \
  --guide-policy cpc_dynamic \
  --student-initialization random \
  --evaluation-protocol brainuicl \
  --max-subjects 0 \
  --guide-cpc-epochs 3 \
  --guide-cpc-learning-rate 0.0001 \
  --benchmark-annotation-diagnostics \
  --memory-size 1000 \
  --replay-epochs 10 \
  --warmup-epochs 2 \
  --output-root experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_brainuicl_eval_seed4321
```

RTX 4070 SUPER 的验证重跑用时约 `102` 秒，49/49 tasks 和 50 个 old checkpoints 均完整写入：

```text
experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_brainuicl_eval_seed4321/metrics.json
```

## 额外伪标签噪声防护实验

为回答无标签 EEG 中 PuriDivER 是否真的能防护噪声，新增：

```text
guide argmax pseudo label
        ↓
以固定 seed 选择约 20% epochs
        ↓
将 pseudo label 均匀翻转为另外四类之一
        ↓
再进入 student online update、memory 与 replay
```

该翻转不读取 target annotation。实际 flip rate 为 `19.832%`。由于 guide 原本已有错误 pseudo labels，额外翻转偶尔会碰巧改回 true class；frozen guide 全流 observed-label purity 从 `71.35%` 变为 `58.67%`，有效总错误率为 `41.33%`。

控制变量固定为 frozen guide、guide-copy student、BrainUICL old/new protocol、memory 1000、49 subjects、seed 4321。两种方法收到完全相同的 pseudo labels 和额外 flip mask：

| 方法 | Extra flip | Input purity | Final old ACC/MF1 | AAA/AAF1 | New-after ACC/MF1 | Final memory purity |
|---|---:|---:|---:|---:|---:|---:|
| Pseudo-ER | 0% | 0.7135 | **0.7301 / 0.7174** | **0.7275 / 0.7157** | **0.7080 / 0.6595** | 0.706 |
| Pseudo-PuriDivER | 0% | 0.7135 | 0.6950 / 0.7014 | 0.7059 / 0.6784 | 0.6771 / 0.6062 | 0.829 |
| Pseudo-ER | 19.832% | 0.5867 | 0.6318 / 0.6092 | 0.6264 / 0.6095 | 0.6340 / 0.5822 | 0.580 |
| Pseudo-PuriDivER | 19.832% | 0.5867 | **0.6919 / 0.6962** | **0.6991 / 0.6842** | **0.6841 / 0.6252** | **0.841** |

| 对比 | Final old ACC | Final old MF1 |
|---|---:|---:|
| ER 的 0→20% 变化 | -9.83 points | -10.82 points |
| PuriDivER 的 0→20% 变化 | -0.31 points | -0.52 points |
| 加噪后 PuriDivER − ER | +6.01 points | +8.70 points |
| Difference-in-differences | +9.52 points | +10.30 points |

因此，在本 seed 的独立对称伪标签翻转下，PuriDivER 显著降低了普通 replay 的噪声敏感性，并将加噪后的 final memory purity 从 `58.0%` 提高到 `84.1%`。这支持“对独立标签噪声有防护”的结论。

边界也很明确：无额外噪声时 PuriDivER 的 final old ACC/MF1 比 ER 低 `3.51/1.60` 点，说明 robust filtering 会误处理正确但困难的 EEG epochs；本实验尚不能代表系统性 guide bias、coherent class confusion 或 adversarial poisoning。结果仍需多 seed 复现。

动态 guide + 随机 student 分支的额外 20% stress test，Final old ACC/MF1 从 `0.6940/0.6645` 下降到 `0.6395/0.5992`，但 memory purity 从 `86.6%` 保持在 `86.5%`。这说明 memory 过滤仍强，但随机 student 的 endpoint 对额外噪声更敏感。

正式加噪结果：

```text
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_er_extra_pseudo_noise20_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_extra_pseudo_noise20_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_extra_pseudo_noise20_brainuicl_eval_seed4321/
```

## 结论

该实现满足本次要求：target 数据无标签、必须使用 guiding model、不使用 CEA、不使用额外 confidence filtering，student 使用 PuriDivER 的 memory 和两层 GMM 过滤。frozen 主实验没有 target CPC；动态扩展只在 guide encoder 上加入 CPC-style 无标签更新。

在 seed 4321 的自然伪标签流中，PuriDivER 提高了 pseudo-memory purity，但没有超过 Frozen guide 或 Pseudo-ER，因此不能声称它改善了 clean/natural 无标签 EEG 持续学习。额外 20% 对称翻转下，它却显著优于 Pseudo-ER，并几乎保持 endpoint，说明其优势主要体现在独立标签噪声防护，而不是无噪声性能。

动态扩展说明随机 student 可以学到有效模型，且恢复了正 plasticity，但其长期平均轨迹和 MF1 仍弱于 guide-copy/frozen-guide 对照。结果基于一个 source guide 和一个 seed。由于用户要求不增加额外置信度过滤，当前主方法不应通过 confidence threshold 调参来改善结果。后续若修改 class-conditional GMM、时序一致性或 subject-aware loss normalization，应明确命名为 EEG-specific PuriDivER，而不是原始方法。

## 代码和结果

```text
experiments/unlabeled_puridiver_eeg.py
tests/test_unlabeled_puridiver_eeg.py

experiments/rttdp_brainuicl_runs/full49_unlabeled_frozen_guide_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_er_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puri_memory_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_seed4321/

experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/probe5_unlabeled_puridiver_frozenguide_randomstudent_seed4321/
experiments/rttdp_brainuicl_runs/probe5_unlabeled_puridiver_cpcguide_randomstudent_seed4321/
experiments/rttdp_brainuicl_runs/probe5_unlabeled_puridiver_cpcguide_guidestudent_seed4321/

experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_er_extra_pseudo_noise20_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_extra_pseudo_noise20_brainuicl_eval_seed4321/
experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_extra_pseudo_noise20_brainuicl_eval_seed4321/
```
