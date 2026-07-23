# 固定共享上传下的 Replay 防御验证

> 所有方法从相同 source-supervised checkpoint 开始，使用相同 49-task ISRUC 顺序、CPC guide hard pseudo-label、10+10 epoch、冻结 BN、1000-sequence memory 和 current:replay=1:1。所有 target 真实标签只在评估和本报告的离线 memory-purity 诊断中读取，不进入 CPC、训练损失、过滤、memory admission 或攻击文件选择。

## 公平性检查

- 每种方法都完成 Clean、等量 Repeat-clean、Fixed-upload Attack 三条件；攻击与 Repeat-clean 在相同奇数 25 个任务执行 N->4N 上传。
- 每个 Fixed-upload Attack 在加载前记录所有攻击 sequence 的 SHA-256 聚合摘要；四种方法在每一个攻击任务的摘要和源目录完全相同。攻击文件不会在被测方法内重新生成。
- 固定文件来自早期 Plain ER source run 的已保存 upload，生成预算为每模态 relative L2 不超过 20%。因此这是固定 stream 比较，不是每个防御各自的 adaptive white-box 上界；文件最初由 Plain ER 状态生成的来源不对称性仍须在解释中保留。

## 绝对结果

| 方法 | Clean old ACC/MF1 | Clean new ACC/MF1 | Attack old ACC/MF1 | Attack new ACC/MF1 | Attack memory 伪标签纯度 |
|---|---:|---:|---:|---:|---:|
| Plain ER | 67.19%/64.75% | 62.69%/54.60% | 62.95%/60.35% | 57.60%/50.21% | 34.00% |
| SPR-style ER | 68.40%/65.67% | 62.86%/54.43% | 65.69%/62.70% | 60.45%/53.35% | 34.22% |
| PuriDivER memory + CE | 64.74%/62.84% | 60.19%/52.78% | 63.25%/60.03% | 57.96%/49.53% | 39.16% |
| PuriDivER memory + C/R/U | 68.26%/62.28% | 64.49%/52.46% | 61.32%/54.63% | 55.84%/43.56% | 40.59% |

## 扣除上传量后的攻击效应

| 方法 | Attack - Repeat old ACC/MF1 | Attack - Repeat new ACC/MF1 | BWT ACC | attack memory poisoned record | poisoned replay draw |
|---|---:|---:|---:|---:|---:|
| Plain ER | -2.85 pp/-3.70 pp | -4.78 pp/-5.78 pp | +0.05 pp | 80.30% | 80.22% |
| SPR-style ER | -2.10 pp/-2.95 pp | -3.06 pp/-2.97 pp | +2.04 pp | 81.20% | 80.13% |
| PuriDivER memory + CE | -0.23 pp/-2.67 pp | -2.26 pp/-3.64 pp | +2.33 pp | 73.30% | 77.45% |
| PuriDivER memory + C/R/U | -6.10 pp/-5.51 pp | -8.04 pp/-7.36 pp | +0.09 pp | 64.80% | 73.09% |

`Attack - Repeat-clean` 扣除了额外 clean 上传、额外训练步数和同量 memory admission 的影响。负数表示攻击仍造成退化；BWT 不能单独用于判断攻击是否有效，因为攻击也会降低每个任务刚适配完成时的起点。

## 相对 Plain ER 的残余退化变化

| 方法 | old ACC/MF1 recovery | new ACC/MF1 recovery |
|---|---:|---:|
| Plain ER | 0.00 pp/0.00 pp | 0.00 pp/0.00 pp |
| SPR-style ER | +0.75 pp/+0.76 pp | +1.71 pp/+2.81 pp |
| PuriDivER memory + CE | +2.62 pp/+1.03 pp | +2.51 pp/+2.13 pp |
| PuriDivER memory + C/R/U | -3.25 pp/-1.81 pp | -3.27 pp/-1.58 pp |

正值表示该方法的 Attack - Repeat-clean 退化小于 Plain ER；它不是跨方法的绝对性能排名，也不是多 seed 显著性结论。

## 方法边界

- `SPR-style ER` 仅在任务结束时使用 student epoch embedding 和 admission pseudo-label 做 SPR 风格的 epoch mask；被拒绝 epoch 在 replay CE 中使用 ignore index。它不是原始 SPR 的完整 delayed expert/self-supervised 在线流程。
- `PuriDivER memory + CE` 使用 task-end purity/diversity sequence pruning，但训练损失仍为 hard pseudo-label CE。
- `PuriDivER memory + C/R/U` 在当前和 replay batch 的 student snapshot loss/uncertainty 上拟合两层 GMM，并使用 clean/relabel/unlabeled 分支损失。memory 仍是 task-end 选择，因此应称为 PuriDivER-style EEG hybrid，不能等同宣称为原论文的逐 minibatch PuriDivER 复现。
- 本轮为单 seed 固定流实验。要支持部署级防御结论，仍需独立 attack-generation seed、至少 3 个 paired training seeds、低预算/低频率 sweep、clean FPR/selection cost 和真实 EEG 伪迹约束。

## 产物

- Runner: `experiments/replay_cl_eeg.py`
- 汇总器: `experiments/summarize_fixed_shared_replay_defenses.py`
- 固定攻击源: `/home/undefined/Desktop/bci/code/tta_security/BrainUICL/experiments/replay_cl_eeg_runs/proxy_dual_harm_plain_er_full49/attack_shared`
