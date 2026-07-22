# EWC Frozen Proxy 攻击强度与覆盖率实验

> 本实验只使用 EWC，所有攻击流来自同一 frozen proxy 方向和 nested sequence mask，victim 不使用 replay、BrainUICL 或额外防御。结果与同一协议的 clean EWC 配对。

## 1. 为什么 clean 不使用 replay 仍有 60%–70% ACC

这些方法不是从随机参数开始学习。source-pretrained 模型在任何新个体增量更新之前，old-generalization ACC 已经是 `70.25%`，在全部 49 个新个体上的初始模型 subject 平均 ACC 是 `64.64%`。EWC clean 完成 49 个任务后旧个体 ACC 为 `69.02%`，说明高准确率主要来自 source pretraining 和跨个体共享的睡眠分期表示，而不是 replay 重新记住了历史样本。

当前是 subject/domain-incremental，而不是增加新类别的 class-incremental：每个个体都使用相同 5 类睡眠阶段和同一个分类头。EWC 的平均当前个体 ACC 从适配前 `64.33%` 变为适配后 `65.24%`，平均只提高 `0.91 pp`。因此新个体 60% 以上的表现多数已经存在于预训练模型，CL 更新只是小幅适配。

无 replay 也不等于没有历史状态。EWC 保存上一阶段参数锚点和对角 Fisher/importance，Online EWC、SI、MAS 也都把历史压缩进参数与重要性向量；它们不保存原始 EEG sequence，但仍以参数形式保留旧知识。本协议还使用 `cl_lr=1e-6`、EWC strength 5000 和冻结 BN running statistics，使每次更新非常保守，进一步减少遗忘。

EWC 是四种正则化 CL 中最直接的代表，因此本 sweep 先用它回答“输入污染需要多大、覆盖多少任务/sequence 才能穿过保守正则化更新的累积阈值”，避免把算法差异混入第一轮强度曲线。

## 2. 预先定义的退化标准

- **可见退化**：最终旧个体 ACC 和最终已见新个体 ACC 都至少比 clean 低 1 个百分点。
- **明显强退化**：上述两个 ACC 都至少比 clean 低 2 个百分点。
- 单个指标下降而另一个指标不下降，只记为局部退化，不称为同时 old/new 明显退化。

## 3. Clean 参照

EWC clean：旧个体 ACC `69.02%`，已见新个体 ACC `62.85%`，旧个体 MF1 `66.14%`，已见新个体 MF1 `53.21%`，BWT ACC `-2.40%`。

## 4. 每 sequence 强度 sweep

固定攻击 25/49 个任务、每个被攻击任务约 20% sequence，使用 shifted `F-S`（所有修改方向取正号）。

| 条件 | 相对 L2 目标 | L∞/std 上限 | 修改 sequence/全流 | proxy pseudo ACC 变化 | 旧 ACC | 新 ACC | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `strength_s005_k25_q20` | 0.5% | 0.02 | 224/2148 | -0.04 pp | 68.86% | 62.52% | -0.16 pp | -0.32 pp | -0.39 pp | 否 / 否 |
| `strength_s010_k25_q20` | 1.0% | 0.04 | 224/2148 | -0.38 pp | 68.92% | 62.50% | -0.10 pp | -0.35 pp | -0.30 pp | 否 / 否 |
| `strength_s025_k25_q20` | 2.5% | 0.10 | 224/2148 | -1.53 pp | 68.63% | 62.41% | -0.38 pp | -0.44 pp | -0.28 pp | 否 / 否 |
| `strength_s050_k25_q20` | 5.0% | 0.20 | 224/2148 | -3.78 pp | 67.92% | 61.76% | -1.10 pp | -1.09 pp | -0.29 pp | 是 / 否 |
| `strength_s100_k25_q20` | 10.0% | 0.40 | 224/2148 | -5.68 pp | 66.90% | 59.90% | -2.12 pp | -2.95 pp | -0.59 pp | 是 / 是 |

攻击强度确实改变了训练输入的 guiding pseudo-label 质量：攻击任务上的 proxy pseudo-label diagnostic ACC 变化从约 0 pp（0.5%）扩大到约 -5.7 pp（10%）。但最终 old/new ACC 只下降约 2.1/3.0 pp，说明 EWC 锚点、同任务未污染 sequence 和后续任务共同削弱了输入层扰动向最终参数的传递。

## 5. 攻击 subject/task 数量 sweep

固定每个被攻击任务约 20% sequence、每 sequence 5% 相对 L2 和 `0.20 × std` 上限；只改变被攻击任务数量。

| 条件 | 攻击任务数 | 修改 sequence/全流 | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |
|---|---:|---:|---:|---:|---:|---|
| `subjects_s050_k01_q20` | 1 | 9/2148 | -0.19 pp | -0.35 pp | -0.18 pp | 否 / 否 |
| `subjects_s050_k03_q20` | 3 | 28/2148 | -0.57 pp | -0.52 pp | -0.38 pp | 否 / 否 |
| `subjects_s050_k10_q20` | 10 | 88/2148 | -0.75 pp | -0.55 pp | -0.30 pp | 否 / 否 |
| `subjects_s050_k25_q20` | 25 | 224/2148 | -1.10 pp | -1.09 pp | -0.29 pp | 是 / 否 |

## 6. 每任务 sequence 覆盖率 sweep

固定攻击 25 个任务、每 sequence 5% 相对 L2；只改变每个被攻击任务内的 sequence 比例。

| 条件 | 任务内 sequence 比例 | 修改 sequence/全流 | 旧 ACC 变化 | 新 ACC 变化 | BWT 变化 | 1 pp / 2 pp |
|---|---:|---:|---:|---:|---:|---|
| `sequences_s050_k25_q05` | 5% | 71/2148 | -0.40 pp | -0.73 pp | -0.33 pp | 否 / 否 |
| `sequences_s050_k25_q10` | 10% | 121/2148 | -0.53 pp | -0.70 pp | -0.17 pp | 否 / 否 |
| `sequences_s050_k25_q20` | 20% | 224/2148 | -1.10 pp | -1.09 pp | -0.29 pp | 是 / 否 |

## 7. 本轮首次明显退化点

- 固定 25 个攻击任务和 20% 任务内覆盖率时，首次达到 old/new 同时下降至少 1 pp 的条件是 `strength_s050_k25_q20`；首次同时下降至少 2 pp 的条件是 `strength_s100_k25_q20`。
- 固定 5% L2 和 20% 任务内覆盖率时，subject/task 数量 sweep 首次达到 1 pp 的条件是 `subjects_s050_k25_q20`；1、3、10 个攻击任务均未达到。
- 固定 25 个攻击任务和 5% L2 时，sequence 覆盖率 sweep 首次达到 1 pp 的条件是 `sequences_s050_k25_q20`；5% 和 10% 覆盖率均未达到。

在当前单 seed 协议中，可把 `25 个攻击任务 + 每任务 20% sequence + 每 sequence 5% 相对 L2` 视为出现约 1 pp 可见退化的首个工程工作点；把 L2 提高到 10% 后才出现 old/new 同时超过 2 pp 的强退化。这个阈值只对当前 frozen proxy、EWC 超参数和 ISRUC split 有效。

## 8. 为什么 BrainWash 原论文下降更大

| 维度 | BrainWash 原论文 | 当前 EEG EWC sweep |
|---|---|---|
| CL 场景 | 10-split CIFAR-100 等，类别互斥、多头 task-incremental | ISRUC 跨个体、共享 5 类、单头 subject/domain-incremental |
| 初始模型 | 按任务顺序训练 ResNet-18 | 已有 source-pretrained EEG 模型，初始 old/new ACC 已约 70%/65% |
| 攻击者 | 读取 victim 当前参数，模型反演旧任务，并对最后任务做白盒双层优化 | 冻结 surrogate，不读取 EWC 轨迹，只复用一步 classifier proxy 方向 |
| 污染范围 | 主表通常污染最后任务全部样本 | 每个攻击任务最多污染约 20% sequence；5% 档全流仅 224/2148 |
| 输入预算 | 图像先归一化到 [0,1]，L∞ ε=0.1 或 0.3 | EEG/EOG 相对 L2 0.5%–10%，并同时受 0.02–0.40 × std 的 L∞ 上限约束 |
| Victim 更新 | SGD learning rate 1e-2 | `cl_lr=1e-6`、EWC 5000、冻结 BN，更新更保守 |
| 主指标 | BWT 与最后一个被攻击任务 ACC | 19 个 old 个体和 49 个 new 个体的最终平均 ACC/MF1、BWT |

BrainWash Table 1 中 CIFAR-100 EWC 的 clean BWT 为 -5.2，ε=0.1 reckless 后为 -12.6，即 BWT 额外下降 7.4 pp；最后任务 ACC 从 68.3% 变为 51.0%。这不是“给 10% EEG sequence 加 5% L2 后总体 ACC 必须下降 10 pp”，而是更强白盒攻击、全任务样本污染、不同任务协议和不同指标共同作用的结果。

## 9. 解释和限制

本 sweep 的强度是输入级、有限、系统同向的 proxy 扰动；它不是原始 BrainWash 的双层优化复现。此前本地无界 proxy degradation 探针在两个精心挑选的 subject 上累计造成约 32.38 个百分点 old ACC 下降，而受限 BrainWash stress 在一个 subject 上约 0.86 个百分点；两者的攻击目标、预算和选择机制都比本 sweep 更激进或不同，不能直接拿 10% 数字横比。

即使强度增加，正则化 CL 仍可能只受到有限影响，因为 source-pretrained 模型已经提供了强分类表示，任务共享同一 5 类 label space，EWC 只更新小步参数，且最终指标是跨 49 个体的平均。若只有少数 sequence 被改，污染梯度还会被同一任务的 clean sequence、伪标签和历史锚点稀释。相反，若攻击任务多、sequence 覆盖率高、方向同向，污染会在更多任务中重复进入参数和 importance 状态，才可能出现明显累积退化。

本报告中的 1/2 个百分点阈值是工程判据，不是统计显著性；当前仍是单 seed。正式结论需要至少 3 个 paired seeds，并报告 subject-level bootstrap 区间、扰动后的伪标签翻转率和 EEG 合理性指标。

## 10. 复现文件

- 上传流生成器：`experiments/generate_ewc_attack_strength_sweep.py`
- 断点运行脚本：`scripts/run_ewc_attack_strength_sweep.sh`
- 运行结果：`runs_ewc/`
- 上游方向 manifest：`frozen_proxy_F-S/manifest.json`
