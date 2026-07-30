# FACED 上六种 EEG 持续学习方法的对齐比较

本报告只比较自然 clean 个体流，不包含 Proxy、输入噪声、数据过滤、净化或 Robust Feature 防护。六种方法均已完整运行 61 个增量个体，且从同一个 FACED 预训练 checkpoint 独立重启。

| 对齐维度 | 设置 |
|---|---|
| 数据划分 | FACED 共 123 位被试，seed 4321 下为 30 train、8 validation、24 old/generalization、61 incremental |
| 增量数据 | 61 位新个体的自然 clean 数据，每位 8 条 sequence，每条 20 个 epoch，共 488 条 sequence |
| 网络与初始化 | 相同的 FACED FeatureExtractor、TransformerEncoder、SleepMLP 和预训练 checkpoint |
| 任务内学习 | 每个个体先进行 10 epoch 无标签 CPC Guide 适配，再进行 10 epoch student 更新 |
| 优化设置 | batch 16，student learning rate `1e-6`，相同 Adam 参数和 weight decay |
| BatchNorm | student 的 running mean/variance 冻结，affine 参数仍可训练 |
| 标签可见性 | 增量真实标签不参与训练，只用于离线诊断和最终评估 |
| 评估 | old 为最终模型在独立 24 位 generalization 个体上的 epoch 级结果；seen-new 为最终模型重新评估全部 61 位已见增量个体 |
| 随机性 | 单随机种子 4321 |

| 方法 | 历史 EEG replay | 当前伪标签与持续学习约束 |
|---|---:|---|
| Finetune | 无 | Guide 的全部 hard argmax 伪标签，无参数约束 |
| EWC | 无 | 全部 hard 伪标签；累计 Fisher 二次约束，strength 5000 |
| Online EWC | 无 | 全部 hard 伪标签；在线 Fisher 与单一 anchor，strength 6500、decay 1 |
| SI | 无 | 全部 hard 伪标签；路径积分重要度约束，strength 1,500,000、`xi=1e-6` |
| MAS | 无 | 全部 hard 伪标签；输出敏感度重要度约束，strength 3000、decay 1 |
| BrainUICL | 有 | 置信度至少 0.9，且每条 sequence 至少 15/20 个 epoch 达标；source/target replay |

正则化强度直接沿用此前 ISRUC 对齐迁移设置，没有使用 FACED validation 或 test 重新调参。

| CL 方法 | 最终 old ACC/MF1 | 最终 seen-new ACC/MF1 | BWT ACC | 当前个体 ACC 增益 |
|---|---:|---:|---:|---:|
| Finetune | 22.47% / 21.65% | 25.10% / 20.58% | -1.96 pp | +3.30 pp |
| EWC | 22.97% / 22.32% | 25.47% / 20.73% | -1.49 pp | +2.46 pp |
| Online EWC | 22.32% / 21.55% | 24.83% / 20.34% | -1.74 pp | +2.67 pp |
| SI | 23.67% / 23.15% | 24.10% / 19.34% | +0.01 pp | +0.26 pp |
| MAS | 23.18% / 22.28% | 24.58% / 19.25% | +0.08 pp | +0.45 pp |
| **BrainUICL** | **24.69% / 23.47%** | **26.62% / 22.06%** | **+0.34 pp** | **+0.64 pp** |

| 方法相对 BrainUICL | old ACC/MF1 差值 | seen-new ACC/MF1 差值 |
|---|---:|---:|
| Finetune | -2.21 / -1.82 pp | -1.52 / -1.49 pp |
| EWC | -1.72 / -1.15 pp | -1.15 / -1.33 pp |
| Online EWC | -2.37 / -1.92 pp | -1.79 / -1.73 pp |
| SI | -1.02 / -0.32 pp | -2.52 / -2.73 pp |
| MAS | -1.51 / -1.19 pp | -2.04 / -2.81 pp |

BrainUICL 在四个最终终点上均为本轮最高。无 replay 方法中，SI 的 old 保持最好，EWC 的 seen-new 性能最好。SI 和 MAS 将 BWT ACC 控制在接近 0，但当前个体 ACC 增益仅为 0.26 和 0.45 pp，说明较强约束同时抑制了个体适配；Finetune、EWC 和 Online EWC 的即时适配更明显，但最终 BWT 为负。BrainUICL 在保留与适配之间取得了本轮最好的综合结果。

该比较不是 memory-budget 匹配实验。五种对照不保存历史 EEG，hard 伪标签覆盖率均为 100%；BrainUICL 初始 replay 中有 240 条带标签 source sequence，随后 61 个任务上传的 488 条 clean sequence 中有 88 条通过高置信条件，覆盖率为 18.03%，涉及 38/61 个任务，最终 buffer 为 328 条 sequence。按准入 sequence 内全部 epoch 和真实标签做离线加权诊断，其伪标签 ACC 为 29.77%；真实标签没有参与准入或训练。

预训练 checkpoint 在 old/generalization 集上的初始 ACC/MF1 为 24.09%/23.92%，在 61 位增量个体上的初始均值为 23.88%/19.53%。因此 FACED 的 20% 至 27% 绝对结果首先反映该九分类 checkpoint 的跨个体泛化起点较低，并不表示迁移代码只学到了随机输出。BrainUICL 最终 old ACC 相对初始提高 0.60 pp，old MF1 下降 0.45 pp。

| 方法 | ISRUC old/new ACC | FACED old/new ACC |
|---|---:|---:|
| Finetune | 60.65% / 54.89% | 22.47% / 25.10% |
| EWC | 69.02% / 62.85% | 22.97% / 25.47% |
| Online EWC | 69.51% / 63.67% | 22.32% / 24.83% |
| SI | 71.20% / 64.85% | 23.67% / 24.10% |
| MAS | 70.69% / 64.32% | 23.18% / 24.58% |
| BrainUICL | 73.26% / 67.62% | 24.69% / 26.62% |

ISRUC 与 FACED 的绝对值不能作为同一任务上的直接胜负比较，因为类别数、通道、epoch 时长、每位个体的数据量和预训练可泛化性均不同。当前可以支持的结论是：在两个数据集各自的同初始化、同划分和同评估协议内，BrainUICL 都优于这五个无 replay 对照；FACED 上算法间差距更小，并且所有方法都受到较低预训练泛化起点限制。

本轮仅运行 seed 4321，不能给出显著性结论。下一阶段加入 Robust Feature 前，应至少保留当前结果作为无防护 clean 基线；若要比较 replay 与 regularization 本身，还需要 memory-matched 组件消融。

原始结果位于本目录的 `regularization/summary.json`、各方法 `metrics.json` 和 `brainuicl/clean/metrics.json`。统一运行入口为 `../run_faced_clean_regularization_brainuicl.sh`，运行时启用了 `--no-save-checkpoints`，没有为 61 个任务逐个保存 checkpoint。
