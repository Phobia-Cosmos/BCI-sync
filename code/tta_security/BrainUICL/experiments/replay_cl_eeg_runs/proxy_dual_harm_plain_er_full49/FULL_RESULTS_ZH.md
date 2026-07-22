# Dual-harm 对 Plain ER-EEG 的验证

> 本实验不使用 BrainUICL 持续学习算法。BrainUICL 网络只作为与正则化实验一致的 source-pretrained backbone；CL 机制是固定容量 reservoir experience replay。

## 协议

- 三条件：Clean ER、等量 Repeat-clean ER、Dual-harm ER。
- 同一 49-task 顺序、CPC guide、hard pseudo-label、10+10 epochs、`cl_lr=1e-6`、冻结 BN。
- Reservoir 容量 1000 sequence；每个 current batch 抽取等量 replay sequence，当前:replay=1:1。
- 无 confidence filter、CEA、DCB、source replay、EWC/SI/MAS penalty。
- Attack 与 Repeat-clean 在奇数 25 个任务均为 N->4N 上传；每次 occurrence 都参加相同 reservoir admission。
- Dual-harm 使用与正则化实验相同的双 proxy、一步 classifier unroll、20% relative L2、`0.5×std` L∞ 和 repeat=3；正则曲率 adapter 在 ER 中关闭。

## 绝对结果

| 条件 | old ACC/MF1 | new ACC/MF1 | BWT ACC | 最终 memory | 最终伪标签纯度 |
|---|---:|---:|---:|---:|---:|
| clean | 67.19%/64.75% | 62.69%/54.60% | -3.02% | 1000 | 65.12% |
| repeat_clean | 65.80%/64.05% | 62.38%/55.99% | -2.30% | 1000 | 68.32% |
| attack_shared | 62.95%/60.35% | 57.60%/50.21% | -2.25% | 1000 | 34.00% |

## 配对差值

| 对比 | old ACC/MF1 | new ACC/MF1 | BWT ACC |
|---|---:|---:|---:|
| repeat_clean_minus_clean | -1.39 pp/-0.70 pp | -0.31 pp/+1.39 pp | +0.71 pp |
| attack_shared_minus_clean | -4.24 pp/-4.40 pp | -5.09 pp/-4.39 pp | +0.77 pp |
| attack_minus_repeat_clean | -2.85 pp/-3.70 pp | -4.78 pp/-5.78 pp | +0.05 pp |

BWT ACC 在 Attack 与 Repeat-clean 间只变化约 0.05 pp，不能据此判断攻击无效：攻击已经降低各任务刚学完时的起点，BWT 只衡量最终值相对该起点的变化。这里必须以最终 old/new 绝对 ACC/MF1 和等量差分为主。

## Replay 持久化诊断

- Attack 最终 reservoir 中 poisoned record 比例：`80.30%`。
- 全程 replay draw 中 poisoned sequence 比例：`80.22%`。
- Attack 最终 memory 总伪标签纯度：`34.00%`；其中 poisoned records 为 `27.73%`，未污染 records 为 `59.57%`。
- 攻击任务 attacked-stream pseudo ACC：`28.32%`；poisoned-CPC 后 clean-current pseudo ACC：`58.85%`。

## 判定

共享 dual-harm 在扣除等量 repeat-clean 后仍使 plain ER 的 old/new ACC 同时下降至少 1 pp，因此对 replay 有效；当前不需要为了“产生效果”而改攻击。

下一步应做机制消融，而不是继续放大预算：比较 current-only 不入库、poison 入库但不重复、repeat 0/1/3、memory capacity 和 replay ratio，确认下降中有多少来自 replay 持久化。

## 如何设计同时影响正则化和 replay 的方法

共同攻击核心应保持输入级和预算一致：双侧 old/new proxy、guide pseudo-label 劫持、一步更新后伤害和等量 repeat 控制。不同 CL 家族只增加内部 adapter：正则化 adapter 使用 importance/anchor 曲率旁路；replay adapter 使用 reservoir 样本、admission/survival 和未来 replay 梯度。共同主表使用冻结 shared proxy，family-aware adapter 作为独立白盒 upper-bound，不能用不同输入的绝对 ACC 直接排名。

正式结论还需要至少 3 seeds，并增加 random-noise、current-only/no-store、store-once、repeat 0/1/3、memory capacity 和 replay ratio 消融。当前单 seed 只能回答可行性和机制路径。

## 复现入口

- Runner：`experiments/replay_cl_eeg.py`
- Orchestrator：`scripts/run_proxy_dual_harm_plain_er_full49.sh`
- 机器结果：`FULL_RESULTS.json`
