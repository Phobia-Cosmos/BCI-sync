## EWC / Plain ER 嵌套 N→N 强度实验

固定条件：seed 4321，同一 clean Finetune surrogate 生成一次最大 frozen payload；所有阶段共享相同 EEG/EOG 数组并使用严格嵌套的 task 与 sequence mask。每条 relative-L2≤20%、L∞/std≤0.50、5 步，N→N、repeat=0、标签不变。Clean 基线只复用、不重复训练。

| 阶段 | K/q | Proxy/全流 | EWC Δold ACC/MF1 | EWC Δnew ACC/MF1 | Plain ER Δold ACC/MF1 | Plain ER Δnew ACC/MF1 | ER memory/replay proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| k01_q20 | 1/20% | 10/2148 (0.47%) | +0.04 pp/+0.11 pp | +0.09 pp/+0.07 pp | +0.19 pp/+0.11 pp | +0.17 pp/+0.19 pp | 0.60%/0.35% |
| k05_q20 | 5/20% | 46/2148 (2.14%) | -0.22 pp/-0.59 pp | -0.59 pp/-0.86 pp | -0.50 pp/-1.00 pp | -1.01 pp/-1.12 pp | 2.20%/2.85% |
| k05_q50 | 5/50% | 112/2148 (5.21%) | -1.28 pp/-2.31 pp | -2.13 pp/-2.46 pp | -0.99 pp/-1.59 pp | -1.89 pp/-1.87 pp | 5.20%/7.15% |
| k10_q50 | 10/50% | 219/2148 (10.20%) | -1.43 pp/-2.45 pp | -2.37 pp/-2.68 pp | -1.68 pp/-2.44 pp | -2.18 pp/-2.20 pp | 10.20%/12.97% |
| k25_q50 | 25/50% | 543/2148 (25.28%) | -2.38 pp/-3.83 pp | -3.89 pp/-4.72 pp | -2.80 pp/-3.97 pp | -3.55 pp/-3.81 pp | 25.20%/25.34% |
| k25_q100 | 25/100% | 1072/2148 (49.91%) | -1.06 pp/-1.85 pp | -1.98 pp/-2.52 pp | +2.02 pp/+2.45 pp | +0.10 pp/+0.34 pp | 50.10%/50.65% |

| 阶段 | EWC 受影响 task 平均 Δcurrent/Δpseudo ACC | Plain ER 受影响 task 平均 Δcurrent/Δpseudo ACC |
|---|---:|---:|
| k01_q20 | +0.00 pp/-4.79 pp | -0.11 pp/-4.79 pp |
| k05_q20 | -1.96 pp/-10.80 pp | -3.52 pp/-11.36 pp |
| k05_q50 | -4.65 pp/-21.82 pp | -5.31 pp/-22.36 pp |
| k10_q50 | -3.67 pp/-27.12 pp | -4.87 pp/-27.47 pp |
| k25_q50 | -6.29 pp/-28.61 pp | -6.22 pp/-28.30 pp |
| k25_q100 | -6.63 pp/-45.74 pp | -4.43 pp/-45.36 pp |

在已测试档位中，EWC 和 Plain ER 的 old/new ACC 合计退化都在 `k25_q50` 达到最大（Plain ER 对应 `k25_q50`）。继续提高到 `k25_q100` 后，尽管受影响 task 的 pseudo ACC 下降更大，最终退化却缩小，Plain ER 的 old ACC 甚至提高。这说明当前 frozen proxy 的覆盖率-退化关系不是单调函数；全量替换可能形成更一致的域偏移或数据增强，而 50% clean/proxy 混合造成的任务内冲突更强。该机制解释仍需后续消融确认。

负差值表示 shared proxy 噪声相对 clean 退化。每个阶段都验证完整 49-task 轨迹，并要求首次受影响 task 之前与 clean 逐字段一致。结果为单 seed 强度曲线，不等同于统计显著性。
