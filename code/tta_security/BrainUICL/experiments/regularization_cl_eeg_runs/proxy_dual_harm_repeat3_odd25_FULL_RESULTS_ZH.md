# 正则化 CL 自适应白盒 Proxy 攻击实验

> 本报告分析 EWC、Online EWC、SI、MAS 的单 seed 强白盒上界实验。该协议只修改输入流，但使用 20% relative-L2 上限、攻击 25/49 个任务并重复上传生成序列，不应表述为低预算或隐蔽攻击。

## 实验协议

- 数据与模型：ISRUC Group-I、seed 4321、BrainUICL source checkpoint、49 个固定顺序的新个体、冻结学生 BN running statistics。
- CL：每个任务 10 个 CPC guide epoch + 10 个学生 epoch，`cl_lr=1e-6`；无 replay、无 confidence filter，目标真实标签只用于离线评估。
- 攻击频率：奇数任务，共 `25/49` 个任务；每个攻击任务替换全部原始上传 sequence。
- 输入预算：逐点 `L∞ <= 0.50 × modality std`，每 sequence/模态 `relative L2 <= 20%`，3 步投影符号梯度。
- 交互放大：每个生成 sequence 额外上传 `3` 份，因此一个攻击任务的 learner 输入为 1 份生成上传 + 3 份重复代理上传。
- 白盒权限：攻击者读取当前 student、CPC-adapted guide 和 EWC/SI/MAS 的 importance/anchor；source-train 输入只作为私有 old proxy 计算外层目标，不进入 learner optimizer。
- 攻击 surrogate：生成器只对 classifier 做一次 `inner_lr=1e-4` 的可微梯度步；真实 learner 则对全模型使用 Adam、`cl_lr=1e-6`、weight decay、gradient clipping 和 10 epochs。该 surrogate 用于寻找有害输入方向，不是对真实优化器的逐步精确复现。

## 最终结果

| 方法 | Clean old ACC/MF1 | Repeat-clean old ACC/MF1 | Attack old ACC/MF1 | Clean new ACC/MF1 | Repeat-clean new ACC/MF1 | Attack new ACC/MF1 |
|---|---:|---:|---:|---:|---:|---:|
| EWC | 69.02%/66.14% | 67.65%/64.95% | 63.04%/56.85% | 62.85%/53.21% | 60.86%/51.66% | 55.38%/43.59% |
| Online EWC | 69.51%/66.69% | 67.48%/64.67% | 60.16%/54.48% | 63.67%/53.91% | 61.86%/52.43% | 53.39%/41.38% |
| SI | 71.20%/69.27% | 70.61%/68.83% | 53.33%/48.39% | 64.85%/55.56% | 64.37%/55.27% | 47.22%/36.71% |
| MAS | 70.69%/68.48% | 70.26%/68.59% | 65.48%/61.64% | 64.32%/55.24% | 64.15%/55.59% | 58.17%/47.64% |

## 配对差值

| 方法 | Repeat-clean − clean old | Attack − clean old | Attack − repeat-clean old | Repeat-clean − clean new | Attack − clean new | Attack − repeat-clean new |
|---|---:|---:|---:|---:|---:|---:|
| EWC | -1.37 pp/-1.18 pp | -5.98 pp/-9.28 pp | -4.60 pp/-8.10 pp | -1.99 pp/-1.55 pp | -7.47 pp/-9.62 pp | -5.48 pp/-8.07 pp |
| Online EWC | -2.03 pp/-2.02 pp | -9.35 pp/-12.21 pp | -7.32 pp/-10.19 pp | -1.80 pp/-1.47 pp | -10.27 pp/-12.53 pp | -8.47 pp/-11.05 pp |
| SI | -0.59 pp/-0.45 pp | -17.87 pp/-20.88 pp | -17.28 pp/-20.44 pp | -0.48 pp/-0.29 pp | -17.62 pp/-18.86 pp | -17.15 pp/-18.56 pp |
| MAS | -0.43 pp/+0.10 pp | -5.21 pp/-6.85 pp | -4.78 pp/-6.95 pp | -0.16 pp/+0.35 pp | -6.15 pp/-7.60 pp | -5.98 pp/-7.95 pp |

## 攻击诊断

| 方法 | 生成 sequence | 额外上传副本 | EEG/EOG relative L2 | 生成输入上的 guide 最大类置信度 | proxy 标签保持率 | 目标标签命中率 | attacked-stream pseudo ACC | poisoned-CPC 后 clean-current pseudo ACC | 相对 repeat-clean 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EWC | 1072 | 3216 | 20.00%/19.99% | 85.69% | 24.31% | 40.86% | 28.35% | 57.12% | -13.74 pp |
| Online EWC | 1072 | 3216 | 20.00%/19.99% | 85.18% | 25.58% | 40.07% | 28.76% | 57.06% | -13.60 pp |
| SI | 1072 | 3216 | 20.00%/19.99% | 88.45% | 24.38% | 38.71% | 27.34% | 59.59% | -11.31 pp |
| MAS | 1072 | 3216 | 20.00%/19.99% | 88.25% | 25.77% | 36.81% | 28.17% | 58.73% | -12.03 pp |

## 组合设计与观测证据

1. **组合目标。** 生成器把 PACOL 式梯度方向项、BrainWash 式一步 surrogate、source/current 双代理损失和正则曲率权重放在同一目标中；设计意图是同时影响旧域稳定性与新域可塑性，并寻找较少受历史正则保护的方向。
2. **两级交互。** 同一上传先进入 CPC guide，再由 guide 生成 hard pseudo-label 训练 student；攻击任务上 pseudo-label ACC 约为 27%–29%，而 poisoned-CPC 后 untouched clean-current pseudo ACC 约为 57%–60%，说明影响不仅停留在单个输入的瞬时预测。
3. **流式累积。** 25 个奇数任务和每 sequence 3 个额外副本让有害上传在极低 `cl_lr` 下获得足够采样次数，并把更新后的模型与 importance/anchor 带入后续任务。
4. **等量控制。** repeat-clean 使用完全相同的任务、索引和 `N -> 4N` 输入量；四方法 `attack - repeat-clean` 的 old/new ACC 仍下降 4.60–17.28/5.48–17.15 pp，因此主要退化不能只由数据量解释。
5. **证据边界。** 当前实验验证的是完整组合相对 clean 与等量 clean-repeat 的效果；没有 random-noise、target-only、no-curvature、no-unroll、repeat 0/1/3 等组件消融，因此不能把下降量分别归因给任何单个设计项。

## 结论边界

- 这是单 seed、强白盒、data-stream upper-bound；证明当前正则化 CL 在足够强的自适应输入与上传频率控制下会发生明显 old/new 双侧退化。
- volume-matched benign-repeat 控制只重复 clean sequence，不改变输入值；因此 `attack - benign-repeat` 是扣除数据量/采样次数后的残余退化，`benign-repeat - clean` 则报告重复上传本身的影响。
- 20% relative L2 较大，尚未证明生理不可察觉。后续应固定本攻击结构，逐步降低预算，并加入频带、幅值、EDF 伪迹和跨 seed 约束。
- source proxy 比 BrainWash 原论文的模型反演权限更强；本实验应称为 adaptive white-box proxy upper bound，而不是 BrainWash 原样复现。
- 历史正式产物中的攻击生成诊断是先对 generation batch 做等权平均、再对攻击任务做等权平均；性能 ACC/MF1 来自完整评估集，不受该诊断聚合口径影响。当前代码已为后续新运行增加 sequence-weighted 诊断。

## 复现文件

- 攻击实现：`experiments/regularization_cl_attacks.py`
- CL runner：`experiments/regularization_cl_eeg.py`
- 汇总器：`experiments/summarize_proxy_dual_harm.py`
- 机器可读结果：`experiments/regularization_cl_eeg_runs/proxy_dual_harm_repeat3_odd25_FULL_RESULTS.json`
