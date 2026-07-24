## 多用户 N→N 聚焦验证：EWC 与 Plain ER

固定条件：seed 4321，受扰动 task [26, 31, 37, 43, 49]，对应 subject [10, 70, 83, 99, 13]；名义每用户 q=50%，实际共替换 108/212 条受影响用户 sequence（50.94%），占完整上传流 5.03%。保持 N→N、repeat=0、relative-L2≤20% 与 L∞/std≤0.50，使用同一 frozen manifest `a043e9559a679e7e0c4902d5f70f6bbfe42f99766b51444b9b1c31c9ec7752cc`。

| 方法 | Clean old ACC/MF1 | Shared old ACC/MF1 | Δ old ACC/MF1 | Clean new ACC/MF1 | Shared new ACC/MF1 | Δ new ACC/MF1 |
|---|---:|---:|---:|---:|---:|---:|
| EWC | 68.99%/66.43% | 68.60%/65.92% | -0.38 pp/-0.51 pp | 62.51%/53.47% | 62.37%/53.21% | -0.14 pp/-0.26 pp |
| Plain ER | 67.19%/64.75% | 67.74%/64.95% | +0.55 pp/+0.20 pp | 62.69%/54.60% | 63.15%/54.97% | +0.46 pp/+0.37 pp |

负差值表示 shared proxy 噪声相对 clean 退化。本表复用 canonical clean 基线；两条 shared 轨迹在首个受扰动 task 前均通过逐字段一致性校验。结果仍是单 seed 点估计。
