# BrainUICL 三数据集配置审计、ISRUC/FACED 对齐复现与 EWC Robust Feature 结果

## 直接结论

BrainUICL 论文没有为 ISRUC、FACED、Physionet-MI 分别设计三套主体网络或三套训练超参数。论文明确说明三项任务使用相同的 `4×CNN + 3×TransformerEncoder + MLP` 主体，仅修改输入层和输出层以适配通道数与类别数；Table 8 的 batch、学习率、epoch、置信度阈值和 CEA 间隔也是三数据集共享设置。因此 FACED 数值较低不能用“论文在 FACED 上使用了另一套更弱参数”解释。

本地已按论文声明的 batch 32、CPC 10 epoch/`1e-6`、持续更新 10 epoch/`1e-7`、BN running statistics 更新、seed 4321 和 full subject stream 重跑 ISRUC 与 FACED。两者的 BrainUICL `M0` 接近论文，但论文报告的持续正迁移没有复现。修正公开 CEA 的 batch 错配后有小幅改善，仍不足以闭合差距。因此当前状态是“论文指标已逐项对应并定位差距”，不是“数值复现成功”。

EWC 上的 Robust Feature 已实现并完成无 Proxy 的严格配对 full 实验。默认理论预算下，ISRUC 四个最终 old/seen-new 指标变化为 `-0.03/-0.06/-0.04/-0.06 pp`，FACED 四项均为 `0.00 pp`；当前没有可测 clean 性能提升或代价。额外防护损失仅约 `6.1e-9` 与 `8.9e-9`，说明默认强度在这两个 EEG 流上几乎不改变 EWC 更新。该结果只说明 clean 兼容性，不证明噪声鲁棒性。

## 三数据集相同与不同之处

| 项目 | ISRUC | FACED | Physionet-MI |
|---|---:|---:|---:|
| 任务 | 五类睡眠分期 | 九类情绪识别 | 四类运动想象 |
| 论文可用被试 | 98 | 123 | 103 |
| Pretraining / Generalization / Incremental | 30 / 19 / 49 | 38 / 24 / 61 | 32 / 20 / 51 |
| 本地 pretraining 内部 train/validation | 24 / 6 | 30 / 8 | 尚未迁移 |
| 通道 | 6 EEG + 2 EOG | 32 EEG | 64 EEG |
| 采样率 | 100 Hz | 250 Hz | 160 Hz |
| 单 epoch 时长与点数 | 30 s / 3000 | 10 s / 2500 | 4 s / 640 |
| 类别数 | 5 | 9 | 4 |
| 本地 full runner | 完整公开 ISRUC 路径 | 依据论文与公开片段重建 | 尚无统一 full runner |

三者共享的论文配置为：监督预训练 100 epoch、`lr=1e-4`；CPC 10 epoch、`lr=1e-6`；持续更新 10 epoch、`lr=1e-7`；batch 32；AdamW `β1=0.5, β2=0.99, weight_decay=3e-4`；置信度阈值 `ξ1=ξ2=0.9`；CEA interval 2；DCB 的 `Strue:Spseudo=8:2`。网络共享四个 CNN block、512 维三层八头 Transformer、dropout 0.1 和 `512→256→128→classes` 分类头。

需要区分论文声明和公开代码事实。公开仓库的可运行路径只覆盖 ISRUC；FACED 没有完整 loader、网络前端、CPC 和训练入口，Physionet 分支也不是当前统一 runner 的可运行等价物。本地 FACED 使用单分支 32 通道 CNN 重建，而 ISRUC 使用 EEG/EOG 双分支，这个实现差异无法由论文或公开源码确认。论文 Table 8 写 AdamW，但公开训练代码实际调用 `torch.optim.Adam`；论文写 ISRUC `0.3–35 Hz` 带通，而公开预处理没有显式执行；论文比较表使用固定 partition 下五种 stream order，本轮只有 seed 4321 单顺序。这些都是不能把本地结果称为严格论文复现的实质边界。

## BrainUICL 与论文逐项对应

| 数据集 / 实现 | ACC `M0/Mi-1/Mi` | MF1 `M0/Mi-1/Mi` | `MNT` AAA/AAF1 |
|---|---:|---:|---:|
| ISRUC 论文 Table 2 | 65.10/72.80/75.10% | 57.60/67.10/70.00% | 74.10/72.10% |
| ISRUC 公开 CEA full49 | 64.64/63.50/62.45% | 55.68/56.14/56.65% | 70.18/67.84% |
| ISRUC 同 batch CEA 修正版 full49 | 64.64/63.81/63.14% | 55.68/56.06/57.01% | 70.28/67.94% |
| FACED 论文 Table 2 | 24.20/38.90/40.30% | 17.60/35.20/37.10% | 36.50/34.50% |
| FACED 公开 CEA 重建 full61 | 23.88/25.69/28.17% | 19.53/20.79/24.21% | 24.11/23.28% |
| FACED 同 batch CEA 修正版 full61 | 23.88/26.14/29.20% | 19.53/21.35/25.37% | 24.22/23.43% |

`M0` 接近论文说明输入尺度、类别数和预训练起点不是主要差距来源。差距集中在 `Mi-1`、`Mi` 和持续稳定性：ISRUC 修正版 `Mi ACC/MF1` 仍低论文 `11.96/12.99 pp`，FACED仍低 `11.10/11.73 pp`。公开 CEA 按 shuffle 后的 `batch_idx` 对齐不同样本，与论文“同一 `XB` 在两个模型时刻”的定义不符；修正为同一 batch 后，ISRUC/FACED 的 `Mi ACC` 分别增加 `0.69/1.03 pp`，但不能解释剩余差距。

## 无 Proxy 的正则化方法迁移

下表全部使用与上述诊断相同的 batch 32、BN 更新、`cl_lr=1e-7` 和 full stream。当前 target 训练使用 CPC Guide 的全部 hard argmax 伪标签，不使用 replay、DCB、CEA、置信度过滤或真实 target label。它们是统一 EEG runner 上的算法迁移，不等同于作者未公开的 Table 3 基线实现。

| 数据集 | 方法 | `Mi` ACC/MF1 | AAA/AAF1 | 最终 old ACC/MF1 | 最终 seen-new ACC/MF1 |
|---|---|---:|---:|---:|---:|
| ISRUC | Finetune | 65.05/57.98% | 60.33/57.11% | 66.02/64.49% | 58.62/50.64% |
| ISRUC | EWC | 65.08/58.01% | 60.36/57.14% | 66.08/64.57% | 58.80/50.77% |
| ISRUC | Online EWC | 65.44/58.50% | 60.43/57.30% | 66.62/65.23% | 59.12/51.31% |
| ISRUC | SI | 65.65/58.79% | 60.45/57.36% | 66.83/65.46% | 59.19/51.41% |
| ISRUC | MAS | 65.57/58.65% | 60.45/57.34% | 66.78/65.40% | 59.16/51.37% |
| FACED | Finetune | 32.72/29.85% | 23.14/21.85% | 24.43/24.01% | 26.51/21.60% |
| FACED | EWC | 32.73/29.85% | 23.14/21.86% | 24.53/24.10% | 26.52/21.62% |
| FACED | Online EWC | 32.73/29.86% | 23.13/21.85% | 24.43/24.01% | 26.51/21.61% |
| FACED | SI | 32.82/30.00% | 23.19/21.90% | 24.74/24.23% | 26.53/21.60% |
| FACED | MAS | 32.84/29.99% | 23.17/21.88% | 24.64/24.15% | 26.57/21.65% |

论文 Table 3 的 EWC 为 ISRUC `70.2/65.2/68.4/66.1%`、FACED `37.5/33.3/33.4/30.5%`，顺序是 `Mi ACC/MF1/AAA/AAF1`。本地 EWC 对应值分别为 ISRUC `65.08/58.01/60.36/57.14%`、FACED `32.73/29.85/23.14/21.86%`，尚未复现论文 EWC。论文没有公开其 EWC strength、importance 估计、伪标签筛选和 FACED 实现，不能通过事后调参把本地数值命名为论文复现。

## EWC 加 Robust Feature

| 数据集 | 条件 | 最终 old ACC/MF1 | 最终 seen-new ACC/MF1 | `Mi` ACC/MF1 | AAA/AAF1 |
|---|---|---:|---:|---:|---:|
| ISRUC | EWC | 66.08/64.57% | 58.80/50.77% | 65.08/58.01% | 60.36/57.14% |
| ISRUC | EWC + Robust Feature | 66.05/64.50% | 58.76/50.71% | 65.06/57.99% | 60.35/57.12% |
| FACED | EWC | 24.53/24.10% | 26.52/21.62% | 32.73/29.85% | 23.14/21.86% |
| FACED | EWC + Robust Feature | 24.53/24.10% | 26.52/21.62% | 32.74/29.86% | 23.14/21.86% |

ISRUC 的 Robust Feature 平均保护 `10.25%` 的分类器输出-特征方向，平均正则特征值为 `3.3218`，最后 epoch 平均额外损失为 `6.11e-9`；FACED 对应为 `8.32%`、`13.8931` 和 `8.93e-9`。当前实现只作用于最终 `128×classes` 线性分类器，并把论文在线性平方损失下的闭式解作为非线性 EEG 网络上的 hybrid approximation；它不继承论文完整理论保证，也不读取 target 真实标签。

结果目录为 `experiments/paper_fidelity_diagnostics/` 和 `experiments/paper_aligned_regularization_clean/`。复现脚本为 `experiments/run_paper_aligned_regularization_robust_feature.sh`，所有 run 使用 `--no-save-checkpoints`。`tests.test_brainuicl_alignment`、`tests.test_icml2026_cl_defenses` 和 `tests.test_regularization_cl_eeg` 共 27 项测试通过。
