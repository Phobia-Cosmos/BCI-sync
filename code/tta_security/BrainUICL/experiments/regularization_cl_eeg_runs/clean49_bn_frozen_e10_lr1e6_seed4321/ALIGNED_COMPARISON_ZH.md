# 六种 EEG 持续学习方法的对齐比较

## 已统一的实验条件

| 维度 | 统一设置 |
|---|---|
| 数据 | 同一份 ISRUC Group 1 物理数据文件 |
| 数据划分 | seed 4321 的同一 train/validation/old/new split |
| 新个体顺序 | 相同的 49 个新个体顺序 |
| 网络 | 相同的 BrainUICL FeatureExtractor、TransformerEncoder 和 SleepMLP |
| 初始化 | 相同的 seed-4321 预训练 checkpoint |
| 训练轮数 | 每个个体 guiding CPC 10 轮，学生更新 10 轮 |
| 优化设置 | batch size 16，学生学习率 `1e-6`，相同 Adam 参数和 weight decay |
| BatchNorm | 所有学生模型均冻结预训练 running mean/variance，affine 参数仍训练 |
| 训练标签 | 当前新个体真实标签不参与训练，只用于诊断和评估 |
| 旧个体指标 | 任务 49 后在同一 19 个 old-generalization 个体上评估 |
| 新个体指标 | 任务 49 后重新评估全部 49 个已见新个体 |
| BWT | 对每个新个体计算“最终性能减去刚学完该个体时的性能”，再取平均 |
| 随机性 | 单随机种子 4321 |

## 保留的算法差异

| 方法组 | Replay | 当前伪标签策略 | 历史监督资源 |
|---|---|---|---|
| Finetune/EWC/Online EWC/SI/MAS | 无 | guiding model 的全部硬伪标签，覆盖率 100% | 无历史 buffer |
| BrainUICL aligned | 有 | 置信度阈值 0.9，并要求序列中至少 15 个 epoch 达标 | 初始有标签历史数据及后续高置信伪标签 |

BrainUICL 中 44.23% 的新序列进入 replay，最终 buffer 为 1,980 个序列，其中包括初始历史训练数据。因此当前比较已经做到数据、训练和指标对齐，但没有做到内存和历史监督资源对齐。

## 同口径结果

| CL 算法 | 最终旧个体 ACC | 最终旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |
|---|---:|---:|---:|---:|---:|
| Finetune | 60.65% | 59.02% | 54.89% | 47.12% | -9.41% |
| EWC | 69.02% | 66.14% | 62.85% | 53.21% | -2.40% |
| Online EWC | 69.51% | 66.69% | 63.67% | 53.91% | -1.03% |
| SI | 71.20% | 69.27% | 64.85% | 55.56% | -0.17% |
| MAS | 70.69% | 68.48% | 64.32% | 55.24% | -0.34% |
| **BrainUICL aligned** | **73.26%** | **69.95%** | **67.62%** | **60.67%** | **+1.39%** |

## 结论边界

在统一指标下，BrainUICL aligned 的旧个体、新个体和 BWT 均优于五个无 replay 方法。此前 BrainUICL 看起来较差，主要是旧运行没有冻结学生 BN，并且缺少任务 49 后的全部新个体重评；当时使用的“新个体刚适配后均值”不能替代最终新个体性能。

这个结果说明 BrainUICL 的 replay 与高置信样本选择在允许额外历史存储时有效，不能据此得出“replay 一定优于正则化”的一般结论。SI 和 MAS 的价值在于不存储历史 EEG 的情况下仍能把 BWT 控制在接近 0。

若要进一步实现严格资源匹配，需要另做两类实验：第一类给 EWC、Online EWC、SI、MAS 同样的 1,980 序列 replay buffer；第二类移除 BrainUICL replay 并统一使用全部硬伪标签。第二类不再是原始 BrainUICL，只能作为组件消融命名。论文主表应使用当前“原生算法、统一指标”的结果，并单独报告 replay 大小和伪标签覆盖率。

## 文件

- 中文图：`aligned_regularization_vs_brainuicl_clean49_zh.png` / `.pdf` / `.svg`
- 英文图：`aligned_regularization_vs_brainuicl_clean49.png` / `.pdf` / `.svg`
- 原始数值：`aligned_regularization_vs_brainuicl_clean49.csv`
- BrainUICL 对齐结果：`../../rttdp_brainuicl_runs/aligned_full49_bn_frozen_lr1e6_seed4321/clean/metrics.json`
