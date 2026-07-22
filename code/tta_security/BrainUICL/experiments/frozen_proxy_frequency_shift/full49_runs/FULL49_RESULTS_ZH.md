# Frozen Proxy 频率 × Shift 正则化 CL 实验报告

> 本报告只分析 EWC、Online EWC、SI、MAS 四种无 replay 的正则化持续学习方法；没有把 BrainUICL 放入本实验。所有数值直接由 `metrics.json` 和攻击流 `metadata.json` 自动汇总。

## 1. 实验是否完整

- 数据：ISRUC Group-I，固定 seed `4321`，49 个新个体按同一顺序作为 49 个任务。
- 代理：冻结 source-pretrained proxy；参数 hash 前后相同：`True`。
- 训练：每个方法 10 个 CPC epoch + 10 个增量 epoch，学生 BN running statistics 冻结，guiding model 只提供 hard pseudo-label；不做置信度过滤、不使用 replay。
- 攻击：每个被攻击任务修改 20% sequence；相对 L2 上限 `5%`，逐点上限为 `0.20 × modality std`，扰动保留在 `0.3–35.0 Hz`。
- `I-NS/I-S` 攻击 3 个任务（任务 13、25、37），`F-NS/F-S` 攻击 25 个任务；四个流共用干净底本、sequence mask 和 proxy 方向。
- `NS` 使用任务内平衡的正负方向，`S` 使用全为正方向；这是有限样本下对 non-shifted/shifted 的工程近似，不是对渐近理论条件的证明。
- 结果完整性：无防御 20 组、Robust Feature 20 组、T2T 8 组，共 48 个方法-条件运行；完成标记为 `_EXECUTION_COMPLETE`，22 项测试通过。

## 2. 攻击流验证

| 流 | 攻击任务数 | 修改 sequence | 上传 sequence | 实际覆盖率 | EEG 相对 L2 均值/最大值 | EOG 相对 L2 均值/最大值 | EEG L∞/std 均值 | 符号 (+/-) | proxy 标签保持率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `I-NS` | 3 | 28 | 2148 | 1.30% | 5.00%/5.00% | 5.00%/5.00% | 0.161 | 14/14 | 68.75% |
| `I-S` | 3 | 28 | 2148 | 1.30% | 5.00%/5.00% | 5.00%/5.00% | 0.161 | 28/0 | 64.82% |
| `F-NS` | 25 | 224 | 2148 | 10.43% | 5.00%/5.00% | 5.00%/5.00% | 0.168 | 112/112 | 63.04% |
| `F-S` | 25 | 224 | 2148 | 10.43% | 5.00%/5.00% | 5.00%/5.00% | 0.168 | 224/0 | 56.79% |

补充诊断：四个流的扰动频带外能量约为 `2.15e-14` 量级，说明带通投影生效；扰动后的样本越过各自 clean sequence 振幅范围的比例约为 `3e-6` 到 `1e-5`。这里的 `relative L2` 和振幅范围只验证输入约束，不等价于攻击成功。
最终保存的攻击流上，冻结 proxy 的 hard pseudo-label 保持率约为 56.79%–68.75%。因此攻击者没有直接篡改真实 label，但输入扰动会改变一部分 guiding pseudo-label；这不是严格意义上的 supervised clean-label poisoning，后续论文表述应称为“输入级 proxy pseudo-label 攻击”。

## 3. 无防御 clean 基线

| CL 方法 | 旧个体 ACC | 旧个体 MF1 | 已见新个体 ACC | 已见新个体 MF1 | BWT ACC | BWT MF1 |
|---|---:|---:|---:|---:|---:|---:|
| EWC | 69.02% | 66.14% | 62.85% | 53.21% | -2.40% | -3.07% |
| Online EWC | 69.51% | 66.69% | 63.67% | 53.91% | -1.03% | -1.46% |
| SI | 71.20% | 69.27% | 64.85% | 55.56% | -0.17% | -0.20% |
| MAS | 70.69% | 68.48% | 64.32% | 55.24% | -0.34% | -0.36% |

这里的旧个体指标是在 clean old-generalization 集上测量，已见新个体指标是在所有 49 个 clean 个体上测量；BWT 越接近 0 表示相对各个体刚适配后的遗忘越小。SI 在本协议中 clean 旧/新 ACC 最高，MAS 次之；这只是单 seed 的协议内基线，不足以宣称普遍算法排名。

## 4. 攻击对无防御 CL 的影响

下表是 `攻击流 - clean` 的百分点变化，负值表示退化；每个单元格顺序为“旧 ACC / 已见新 ACC / BWT ACC”。

| CL 方法 | I-NS 旧/新/BWT | I-S 旧/新/BWT | F-NS 旧/新/BWT | F-S 旧/新/BWT |
|---|---:|---:|---:|---:|
| EWC | -0.07 pp / -0.20 pp / -0.17 pp | -0.23 pp / -0.36 pp / -0.35 pp | -0.65 pp / -0.77 pp / -0.32 pp | -0.74 pp / -0.82 pp / -0.12 pp |
| Online EWC | +0.02 pp / -0.08 pp / -0.21 pp | +0.04 pp / -0.00 pp / -0.24 pp | -0.36 pp / -0.48 pp / -0.26 pp | -0.71 pp / -0.94 pp / -0.61 pp |
| SI | -0.11 pp / -0.23 pp / +0.16 pp | -0.22 pp / -0.35 pp / +0.12 pp | -1.15 pp / -1.34 pp / +0.05 pp | -1.29 pp / -1.37 pp / +0.10 pp |
| MAS | -0.06 pp / -0.12 pp / +0.24 pp | -0.25 pp / -0.33 pp / +0.18 pp | -1.02 pp / -0.78 pp / +0.31 pp | -1.49 pp / -1.01 pp / +0.25 pp |

MF1 的对应变化如下，单元格顺序为“旧 MF1 / 已见新 MF1 / BWT MF1”。

| CL 方法 | I-NS 旧/新/BWT | I-S 旧/新/BWT | F-NS 旧/新/BWT | F-S 旧/新/BWT |
|---|---:|---:|---:|---:|
| EWC | +0.10 pp / +0.23 pp / -0.06 pp | -0.12 pp / +0.02 pp / -0.27 pp | -0.78 pp / -0.37 pp / -0.10 pp | -0.86 pp / -0.45 pp / +0.11 pp |
| Online EWC | +0.38 pp / +0.25 pp / -0.26 pp | +0.43 pp / +0.31 pp / -0.29 pp | -0.30 pp / -0.28 pp / -0.21 pp | -0.73 pp / -0.73 pp / -0.58 pp |
| SI | -0.23 pp / -0.26 pp / +0.22 pp | -0.38 pp / -0.38 pp / +0.19 pp | -2.08 pp / -1.26 pp / +0.04 pp | -2.24 pp / -1.29 pp / +0.06 pp |
| MAS | -0.24 pp / -0.19 pp / +0.27 pp | -0.54 pp / -0.40 pp / +0.16 pp | -1.71 pp / -0.84 pp / +0.27 pp | -2.38 pp / -1.08 pp / +0.21 pp |

从配对差值看，攻击频率从 3 个任务增加到 25 个任务后，退化总体更明显；同一频率下，`F-S` 通常不优于 `F-NS`，但差异并非对所有方法单调。由于每个任务的总预算相同，频繁条件同时增加了累计污染能量，因此这里回答的是“现实中长期重复接收污染的累计影响”，不是只改变攻击频率而保持总能量不变的因果实验。

## 5. Robust Feature 配对结果

Robust Feature 的 clean 代价必须单独计算；下面的恢复量定义为：`(RF 攻击 - RF clean) - (无防御攻击 - 无防御 clean)`。恢复量为正表示相对减轻了攻击退化。

| CL 方法 | 攻击 | 无防御旧 ACC 退化 | RF 旧 ACC 退化 | 旧 ACC 恢复量 | 无防御新 ACC 退化 | RF 新 ACC 退化 | 新 ACC 恢复量 |
|---|---|---:|---:|---:|---:|---:|---:|
| EWC | I-NS | -0.07 pp | +0.08 pp | +0.16 pp | -0.20 pp | +0.04 pp | +0.24 pp |
| EWC | I-S | -0.23 pp | +0.04 pp | +0.27 pp | -0.36 pp | -0.09 pp | +0.27 pp |
| EWC | F-NS | -0.65 pp | -0.51 pp | +0.13 pp | -0.77 pp | -0.41 pp | +0.35 pp |
| EWC | F-S | -0.74 pp | -0.84 pp | -0.10 pp | -0.82 pp | -0.63 pp | +0.19 pp |
| Online EWC | I-NS | +0.02 pp | -0.02 pp | -0.04 pp | -0.08 pp | -0.03 pp | +0.04 pp |
| Online EWC | I-S | +0.04 pp | -0.02 pp | -0.06 pp | -0.00 pp | -0.04 pp | -0.04 pp |
| Online EWC | F-NS | -0.36 pp | -0.69 pp | -0.34 pp | -0.48 pp | -0.46 pp | +0.01 pp |
| Online EWC | F-S | -0.71 pp | -0.71 pp | 0.00 pp | -0.94 pp | -0.75 pp | +0.19 pp |
| SI | I-NS | -0.11 pp | +0.05 pp | +0.16 pp | -0.23 pp | +0.04 pp | +0.27 pp |
| SI | I-S | -0.22 pp | -0.10 pp | +0.13 pp | -0.35 pp | -0.04 pp | +0.31 pp |
| SI | F-NS | -1.15 pp | -0.97 pp | +0.18 pp | -1.34 pp | -1.15 pp | +0.19 pp |
| SI | F-S | -1.29 pp | -1.13 pp | +0.16 pp | -1.37 pp | -1.31 pp | +0.07 pp |
| MAS | I-NS | -0.06 pp | -0.17 pp | -0.11 pp | -0.12 pp | -0.16 pp | -0.03 pp |
| MAS | I-S | -0.25 pp | -0.19 pp | +0.06 pp | -0.33 pp | -0.17 pp | +0.16 pp |
| MAS | F-NS | -1.02 pp | -0.87 pp | +0.15 pp | -0.78 pp | -0.86 pp | -0.07 pp |
| MAS | F-S | -1.49 pp | -1.05 pp | +0.44 pp | -1.01 pp | -0.97 pp | +0.04 pp |

本轮结果不能支持 Robust Feature 已经被证明有效。原因是：第一，只有一个 seed；第二，当前 EEG 实现把论文的线性平方损失/共同特征基公式近似到最后线性分类器的特征协方差；第三，`F-NS` 虽然最接近论文适用条件，但应以跨 seed 的正恢复量和置信区间作为证据。当前表格只说明在这条固定 proxy 流和这组超参数下，哪些方法的配对退化变小或变大。

## 6. T2T 结果与误报

T2T 只在 clean 和 `I-S` 上运行，动作是 rollback、参数范围是全部可训练参数。`clean 触发` 在没有攻击时按定义都是 clean false positive；`I-S` 的检测不能因为触发了某个正常任务就称为攻击检测。

| CL 方法 | clean 有效分数 | clean 触发对数 | clean 回滚任务数 | I-S 有效分数 | I-S 触发对数 | I-S 回滚任务数 | 新增检测端点 | 新增回滚攻击任务 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EWC | 41 | 6 | 12 | 41 | 6 | 12 | 无 | 无 |
| Online EWC | 41 | 6 | 12 | 40 | 7 | 14 | 35 | 无 |
| SI | 42 | 5 | 10 | 44 | 3 | 6 | 8 | 无 |
| MAS | 44 | 3 | 6 | 44 | 3 | 6 | 无 | 无 |

T2T 的关键观察是：clean 流已经出现多次触发和回滚，而 `I-S` 没有对攻击任务产生清晰、可归因且超出 clean 基线的新检测。换句话说，固定的 `2.5 × 最近分数均值` 在这个深度 EEG subject-CL 协议中主要响应正常跨个体更新动力学；它同时造成性能轨迹变化，因此不能把 T2T 的最终 ACC 直接解释为防护收益。

## 7. 结论边界与下一步

1. **攻击流成立。** 四类上传文件固定、共享、有限，并满足预设频率/符号/幅值/频带设计；因此可以作为后续防御比较的同一输入基准。
2. **攻击在当前协议下可测但较弱。** `F-S` 对无防御方法的 old ACC 下降约 0.7–1.5 个百分点量级，说明有累计影响，但不是强破坏性攻击；需要在不破坏 EEG 合理性的前提下做预算/覆盖率 sweep。
3. **Robust Feature 目前只能作为探索性结果。** 要声称防护有效，必须扩展至少 3 个 seed，并同时报告 clean 代价、攻击恢复量和不确定性。
4. **T2T 当前不适合作为自动 rollback 防御。** clean 误报已经较高，且 `I-S` 没有清晰的新增攻击检测；应先做 clean calibration + monitor-only，再决定是否保留。
5. **本实验不包含 BrainUICL、memory、PACOL/BrainWash 原始攻击复现，也没有测试总污染能量固定的频率消融。** 因此不能把本报告结论推广到 replay 方法或宣称复现了原论文攻击。
6. **正式主实验建议。** 固定当前协议，加入至少 3 个 model/attack seed；对 `F-NS` 做 Robust Feature 主验证，对 `I-S` 保留 T2T 附加压力测试；同时增加 `K` 攻击任务但固定总平方 L2 能量的对照，分离频率效应与累计能量效应。

## 8. 复现入口

- 总编排脚本：`scripts/run_frozen_proxy_regularization_full49.sh`
- 攻击 manifest：`/home/undefined/Desktop/bci/code/tta_security/BrainUICL/experiments/frozen_proxy_frequency_shift/full49_seed4321/manifest.json`
- 运行结果根目录：`/home/undefined/Desktop/bci/code/tta_security/BrainUICL/experiments/frozen_proxy_frequency_shift/full49_runs`
- 自动汇总脚本：`experiments/summarize_frozen_proxy_frequency_shift.py`
- 完成日志：`full49_runs/orchestrator.log`；测试日志：`full49_runs/final_tests.log`。
