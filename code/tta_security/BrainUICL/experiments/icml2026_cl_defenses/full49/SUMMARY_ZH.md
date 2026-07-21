# ICML 2026 两种持续学习防御在 clean EEG full49 上的迁移结果

本报告只回答两个问题：论文的 Task-to-Task（T2T）验证和 Robust Feature Defense 能否接入当前无 replay 的正则化 CL-EEG；在没有人为噪声或投毒时，加入防御相对原 clean 基线会产生什么代价。这里没有攻击，因此结果不能证明任何攻击鲁棒性。

## 直接结论

T2T 可以接入 EWC、Online EWC、SI 和 MAS，因为这些方法在任务间都有可解释为二次曲率的历史正则项。它不能原样接入 Finetune：Finetune 的历史正则矩阵为零，论文式 (6) 中 A 和 B 的共同投影子空间为空，无法产生有效检测分数。BrainUICL、SPR-EEG 和 PuriDivER-EEG 以 replay/memory 为核心，也不属于论文式 (2) 的无 replay 正则化框架，本轮没有把论文的理论保证外推到它们。

Robust Feature Defense 可以作为额外二次正则项接到五种方法。当前实现将它放在 BrainUICL 睡眠分类器的最后线性层：对每个新个体，仅用输入 EEG 提取 128 维倒数第二层特征，估计特征二阶矩阵，在其特征方向上求论文式 (14)–(15) 的 protected set 和正则特征值。它不读取 target 真实标签、不使用 replay，也不做置信度过滤。

clean full49 的核心结果是：两种防御都没有产生跨算法一致的免费增益。T2T 会把正常个体漂移误判为异常并回滚 6–12 个任务；Robust Feature 不拒绝任务，性能变化大多小于 0.35 个百分点，但额外损失远小于原训练损失。后续攻击实验必须分别按论文适用条件验证，不能仅凭本轮 clean 运行声称防御有效。

## 对齐协议

三组结果使用同一 ISRUC Group-I、seed 4321、49 个体顺序、BrainUICL backbone/checkpoint、10 个 CPC epoch、10 个增量 epoch、`ssl_lr=cl_lr=1e-6`、batch 16、冻结学生 BatchNorm running statistics、全部 guiding-model 硬伪标签和任务 49 后重评全部 49 个体。正则参数仍为 EWC 5000，Online EWC 6500/decay 1，SI 1500000/xi 0.000001，MAS 3000/decay 1。

- 无防御：`experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321`
- T2T：`experiments/regularization_cl_eeg_runs/icml2026_t2t_clean49_bn_frozen_e10_lr1e6_seed4321`
- Robust Feature：`experiments/regularization_cl_eeg_runs/icml2026_robust_feature_clean49_bn_frozen_e10_lr1e6_seed4321`

## T2T clean 结果

括号内是相对同一算法无防御基线的变化；`pp` 表示百分点。

| 算法 | 旧个体 ACC | 旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |
|---|---:|---:|---:|---:|---:|
| EWC + T2T | 68.87% (-0.14 pp) | 66.28% (+0.14 pp) | 62.41% (-0.44 pp) | 52.93% (-0.28 pp) | -2.95% (-0.56 pp) |
| Online EWC + T2T | 70.01% (+0.50 pp) | 67.97% (+1.28 pp) | 63.74% (+0.07 pp) | 54.54% (+0.63 pp) | -1.22% (-0.19 pp) |
| SI + T2T | 71.01% (-0.19 pp) | 69.07% (-0.21 pp) | 64.53% (-0.32 pp) | 55.36% (-0.21 pp) | -0.14% (+0.03 pp) |
| MAS + T2T | 70.65% (-0.04 pp) | 68.58% (+0.10 pp) | 64.47% (+0.16 pp) | 55.33% (+0.09 pp) | -0.43% (-0.09 pp) |

| 算法 | 有效检测分数 | 触发次数 | 被回滚的 clean 任务 | clean 任务回滚率 |
|---|---:|---:|---:|---:|
| EWC | 41 | 6 | 12/49 | 24.49% |
| Online EWC | 41 | 6 | 12/49 | 24.49% |
| SI | 42 | 5 | 10/49 | 20.41% |
| MAS | 44 | 3 | 6/49 | 12.24% |

论文实验采用对角 Hessian 近似和固定启发式阈值：当前分数至少是此前最多 5 个可用分数均值的 2.5 倍时触发。我们没有根据 EEG clean test 结果事后修改该阈值。结果表明 EEG 的合法跨个体 domain shift 足以产生检测峰值：T2T 在所有四种方法上都发生 clean 误报。Online EWC 的 ACC 和多种方法的部分 MF1 有小幅上升，是因为回滚偶然过滤了部分低质量伪标签任务，而不是因为本轮存在攻击；它同时拒绝了 24.49% 的正常任务，不能据此称为 clean 性能改进算法。

## Robust Feature clean 结果

| 算法 | 旧个体 ACC | 旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |
|---|---:|---:|---:|---:|---:|
| Finetune + Robust Feature | 60.63% (-0.01 pp) | 59.14% (+0.12 pp) | 54.71% (-0.18 pp) | 47.10% (-0.03 pp) | -9.44% (-0.03 pp) |
| EWC + Robust Feature | 68.87% (-0.15 pp) | 66.22% (+0.08 pp) | 62.51% (-0.34 pp) | 53.37% (+0.16 pp) | -2.70% (-0.31 pp) |
| Online EWC + Robust Feature | 69.67% (+0.16 pp) | 67.23% (+0.54 pp) | 63.72% (+0.05 pp) | 54.26% (+0.35 pp) | -1.23% (-0.20 pp) |
| SI + Robust Feature | 71.18% (-0.02 pp) | 69.16% (-0.11 pp) | 64.70% (-0.15 pp) | 55.35% (-0.21 pp) | -0.01% (+0.16 pp) |
| MAS + Robust Feature | 70.74% (+0.05 pp) | 68.45% (-0.03 pp) | 64.32% (+0.01 pp) | 55.19% (-0.06 pp) | -0.02% (+0.32 pp) |

| 算法 | 平均 protected directions | 平均正则特征值 | 最后 epoch 平均防御损失 |
|---|---:|---:|---:|
| Finetune | 10.99% | 3.6541 | 2.386e-06 |
| EWC | 10.94% | 3.3980 | 1.814e-06 |
| Online EWC | 10.97% | 3.4577 | 1.361e-06 |
| SI | 11.42% | 3.7821 | 9.529e-09 |
| MAS | 11.24% | 3.8363 | 7.185e-09 |

默认预算按论文 CIFAR-100 设置做维度归一化：论文 `M=2000`、线性头 `768×100`，因此 EEG 使用每参数预算 `2000/(768×100)`，总预算随 `128×5` 分类器维度缩放。平均约 11% 的输出-特征方向进入 protected set。对 Finetune/EWC/Online EWC，最后 epoch 的额外防御损失约为 `1e-6`；对 SI/MAS 约为 `1e-8`，远小于伪标签交叉熵和已有正则项。因此本轮主要证明公式、状态和训练接口已经迁移，尚未证明该默认预算足以抵抗 EEG 投毒。

## 实现与论文保证的边界

1. T2T 使用每个任务的对角 empirical Fisher 近似损失 Hessian，并用现有正则器 penalty 对参数的二阶导数作为 H。触发后严格回滚到两次更新之前，同时恢复模型参数、BatchNorm buffers 和 EWC/SI/MAS 状态。EWC 的累计加权中心并不总等于论文中的上一模型点，因此这是论文非线性实验风格的近似迁移，不继承线性定理。
2. Robust Feature 的论文闭式解要求线性平方损失和任务 Hessian 可同时对角化。EEG 使用非线性 backbone、交叉熵和持续变化的特征。当前实现只在最终线性分类器上使用当前特征协方差的特征基，并把风险协方差旋转到新基底；这是可运行的 hybrid approximation，不是定理 5.3 的严格实例。
3. Robust Feature 作为额外二次项叠加在原 EWC/Online EWC/SI/MAS 上，以保留原算法身份。论文的 H 本身是系统要设计的唯一正则矩阵，因此“叠加版”不能直接称作论文最优 H。
4. 所有 target 真实标签只用于训练后诊断 ACC/MF1，不参与 T2T 分数、特征协方差、protected set、伪标签或梯度更新。
5. 当前只有一个 seed。表中小于约 0.5 pp 的变化不能视为统计显著结论。

## 下一步攻击验证应如何分开

T2T 只应先测试少量任务上的 shifted 强攻击，并在独立 clean calibration subjects 上固定阈值后再锁定攻击任务和预算。评价必须同时报告检测率、clean 误报率、回滚任务数和最终性能。

Robust Feature 只应先测试频繁、幅度有界、条件零均值的 non-shifted 扰动。开始攻击实验前，应做预算 sweep 并检查防御梯度/原损失梯度比；如果额外项始终只有 `1e-8`–`1e-6`，即使 clean 指标不下降也不能期待可测防护。对 shifted proxy noise 使用 Robust Feature 不符合论文定理的适用条件。

## 复现命令

```bash
/home/undefined/Disk/python-envs/brainuicl/bin/python \
  experiments/regularization_cl_eeg.py \
  --methods ewc,online_ewc,si,mas --defense-mode t2t \
  --ssl-epoch 10 --incremental-epoch 10 \
  --ssl-lr 1e-6 --cl-lr 1e-6 --freeze-bn-stats \
  --ewc-strength 5000 --online-ewc-strength 6500 \
  --si-strength 1500000 --si-xi 1e-6 --mas-strength 3000 \
  --no-save-checkpoints \
  --output-root experiments/regularization_cl_eeg_runs/icml2026_t2t_clean49_bn_frozen_e10_lr1e6_seed4321

/home/undefined/Disk/python-envs/brainuicl/bin/python \
  experiments/regularization_cl_eeg.py \
  --methods finetune,ewc,online_ewc,si,mas \
  --defense-mode robust_feature \
  --ssl-epoch 10 --incremental-epoch 10 \
  --ssl-lr 1e-6 --cl-lr 1e-6 --freeze-bn-stats \
  --ewc-strength 5000 --online-ewc-strength 6500 \
  --si-strength 1500000 --si-xi 1e-6 --mas-strength 3000 \
  --no-save-checkpoints \
  --output-root experiments/regularization_cl_eeg_runs/icml2026_robust_feature_clean49_bn_frozen_e10_lr1e6_seed4321

/home/undefined/Disk/python-envs/brainuicl/bin/python \
  experiments/summarize_icml2026_clean_defenses.py
```
