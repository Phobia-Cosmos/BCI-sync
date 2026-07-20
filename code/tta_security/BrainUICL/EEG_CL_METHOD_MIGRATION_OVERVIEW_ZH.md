# EEG 持续学习方法迁移总览

本文档是当前仓库四条 EEG 持续学习实验线的上层导航：SPR-EEG、PuriDivER-EEG、BrainUICL 和 CL-EEG。当前阶段的首要目标是验证这些方法能否从图像或一般持续学习场景迁移到 EEG，而不是立即宣称它们已经在完全相同的资源和评估条件下分出最终优劣。

## 一页结论

| 方法线 | 当前状态 | Target 人工标签参与训练 | Replay | 推荐主入口 | 当前可以支持的结论 |
|---|---|---:|---:|---|---|
| SPR-EEG | 已完成无标签 full49 迁移 | 否 | 是 | `experiments/spr_eeg_unlabeled/run.py` | SPR 的自监督 expert/base、SCF 图过滤和 purified memory 可以迁移到 EEG；能够降低随机或非相干伪标签错误，但不能可靠识别自洽的错误簇或强自适应投毒 |
| PuriDivER-EEG | 已完成 pure/oracle、无标签和 BrainUICL 防御接入 | 无标签主线为否 | 是 | `experiments/unlabeled_puridiver_eeg.py` | 两层 GMM、C/R/U 划分、soft relabel、consistency 和 purity-diversity memory 可以在 EEG 上运行；对独立标签噪声有效，对高置信且特征自洽的攻击无效 |
| BrainUICL | 已完成 aligned full49 基线 | 当前 target 否；历史 source replay 有标签 | 是 | `experiments/rttdp_brainuicl_full.py` | 动态 CPC guide、置信度筛选、source/target replay 和 CEA 能够完成无标签个体连续适配；在允许历史存储时，当前统一指标下性能最好 |
| CL-EEG | 已完成无 replay full49 和轻量攻击探针 | 否 | 否 | `experiments/regularization_cl_eeg.py` | EWC、Online EWC、SI、MAS 可以直接迁移到 EEG 伪标签流；SI/MAS 无需历史 EEG 即可把遗忘压到接近 0，但新个体更新幅度也较小 |

当前最稳妥的总论是：四类方法的核心机制都已经在 ISRUC EEG 数据上跑通。Replay 类方法提供更强的历史保持和最终性能，但需要存储历史 EEG；正则化方法不存储历史数据，性能略低但隐私和存储约束更好。SPR/PuriDivER 的净化机制主要针对随机、离散或低一致性的标签污染，不能单独作为通用对抗投毒防御。

## 统一目录

```text
/home/undefined/Desktop/bci/code/tta_security/BrainUICL/
├── model/
│   ├── pretrain_net.py                 # BrainUICL 主干网络
│   ├── spr_eeg.py                      # SPR-EEG 图过滤、Beta mixture、purified memory
│   └── regularization_cl.py            # EWC、Online EWC、SI、MAS
├── experiments/
│   ├── rttdp_brainuicl_full.py         # BrainUICL 可复现实验、攻击/防御接入、aligned 评估
│   ├── spr_eeg_pure.py                 # pure/oracle SPR-EEG
│   ├── spr_eeg_unlabeled/              # 无标签 SPR-EEG 主线
│   ├── spr_eeg_random_init/            # SPR 随机初始化与协议诊断
│   ├── pure_puridiver_eeg.py           # pure/oracle PuriDivER-EEG
│   ├── unlabeled_puridiver_eeg.py      # 无标签 PuriDivER-EEG 主线
│   ├── puridiver_eeg.py                # BrainUICL + PuriDivER-style 防御组件
│   ├── regularization_cl_eeg.py        # CL-EEG 干净持续学习 runner
│   ├── regularization_cl_attacks.py    # PACOL-style / BrainWash-style 轻量探针
│   └── regularization_cl_eeg_runs/     # CL-EEG 结果、表格和图
└── tests/                              # 各方法的标签隔离、过滤、正则和指标测试
```

共享运行资源：

```text
Python:     /home/undefined/Disk/python-envs/brainuicl/bin/python
Dataset:    /home/undefined/Disk/datasets/brainuicl/processed/isruc_group1_npy_float32
Checkpoint: /home/undefined/Disk/ai-storage/BrainUICL/model_parameter
```

## SPR-EEG

### 方法从哪里迁移

SPR 原方法面向带噪数据流，使用 delayed buffer、expert 自监督、base Self-Replay、按观测类别建立的特征相似图、eigenvector centrality、两成分 Beta mixture 和 purified memory。EEG 迁移中，图节点从图像样本改为 30 秒 EEG epoch，当前个体承担 delayed stream 的角色，20-epoch sequence 或独立 epoch 进入长期 memory。

SPR 原文使用 SimCLR/NT-Xent；早期 BrainUICL 防御接入为了适配 EEG 时间序列使用 CPC。后续 pure/unlabeled SPR 主线已经补充 NT-Xent expert/base，因此应优先引用这些实现，不应把早期 `--defense-mode spr` 探针当成完整 SPR 复现。

### 推荐代码与结果

| 类型 | 路径 | 用途 |
|---|---|---|
| 核心组件 | `model/spr_eeg.py` | 随机图、中心性、Beta mixture、epoch-to-sequence purification、memory |
| 无标签主线 | `experiments/spr_eeg_unlabeled/run.py` | CPC guiding model 生成全部 target 伪标签，SPR expert/base 和 memory 完成持续学习 |
| 过滤组件 | `experiments/spr_eeg_unlabeled/filtering.py` | 无标签 SCF 过滤与诊断标签隔离 |
| Pure/oracle | `experiments/spr_eeg_pure.py` | 使用观测标签验证 SPR 核心机制，不能代表无标签部署 |
| 随机初始化诊断 | `experiments/spr_eeg_random_init/` | 区分 source checkpoint、随机 student 和协议本身的贡献 |
| 主结果 | `experiments/rttdp_brainuicl_runs/spr_unlabeled_full49_e10_seed4321/metrics.json` | 49 个 target subjects、10/10/10/10 epochs |
| 结果报告 | `experiments/spr_eeg_unlabeled/RESULTS.md` | full49、额外伪标签噪声、purity 与 paired statistics |
| 早期防御报告 | `SPR_EEG_DEFENSE_REPORT.md` | SPR 作为 BrainUICL buffer filter 的能力与失败边界 |

### 已验证结果

无标签 full49 运行在固定 old-generalization 集上的最终 ACC/MF1 为 `0.7109/0.6815`，新个体刚适配后的平均 ACC/MF1 为 `0.6694/0.5951`，FR 为 `0.0120`。SCF 在没有 confidence gate 的情况下将加权伪标签错误率从 `30.09%` 降至 `26.35%`，49 个个体中有 45 个的接收集错误率下降。

该结果证明 SPR 的特征中心性过滤和自监督 replay 可以迁移到 EEG。它同时说明 purity 与 subject/class diversity 必须平衡：过滤过严会删除过多个体特异样本并降低分类性能。对于形成高置信、特征一致错误簇的 proxy-meta 攻击，SPR 会把攻击簇当作中心干净簇，因此不能宣称具备通用攻击防护能力。

### 尚未完成

SPR full49 尚未使用当前 BrainUICL/CL-EEG 的统一“任务 49 后重评全部 49 个体并计算 BWT”实现。它的 old 指标和 new-after 指标可以说明迁移成功，但进入六方法统一排名前还需要补齐 final-seen/BWT，并统一 BN、学习率和 memory 资源表。

## PuriDivER-EEG

### 方法从哪里迁移

PuriDivER 原方法使用逐样本 loss-GMM 估计 clean posterior，再对 noisy set 使用 uncertainty-GMM 分成可 soft relabel 的 R 集和只做增强一致性的 U 集；长期 memory 通过 purity-diversity 分数控制容量。EEG 迁移中，observed label 可以是真实/人工注入噪声标签，也可以由 source guiding model 生成伪标签。

### 推荐代码与结果

| 类型 | 路径 | 用途 |
|---|---|---|
| Pure/oracle | `experiments/pure_puridiver_eeg.py` | 单 epoch compact EEG 模型，验证 PuriDivER 核心机制，不包含 BrainUICL |
| 无标签主线 | `experiments/unlabeled_puridiver_eeg.py` | source guide 生成全部伪标签，PuriDivER memory + C/R/U robust replay |
| BrainUICL 防御接入 | `experiments/puridiver_eeg.py` | 在 BrainUICL replay/teacher/CEA 上增加 PuriDivER-style 净化，不是 pure PuriDivER |
| 无标签主结果 | `experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_seed4321/metrics.json` | 70% adaptation / 30% held-out、49 subjects、memory 1000 |
| BrainUICL-compatible 结果 | `experiments/rttdp_brainuicl_runs/full49_unlabeled_pseudo_puridiver_brainuicl_eval_seed4321/metrics.json` | 固定 old/new order 的兼容评估 |
| 动态 guide 扩展 | `experiments/rttdp_brainuicl_runs/full49_unlabeled_puridiver_cpcguide_randomstudent_brainuicl_eval_seed4321/metrics.json` | CPC-dynamic guide + random student，不能与 frozen-guide 主实验视为单一变量 |
| 总报告 | `UNLABELED_PURIDIVER_EEG_REPORT.md` | 方法边界、GMM、主实验、扩展和噪声防护 |
| 防御接入报告 | `PURIDIVER_EEG_DEFENSE_REPORT.md` | BrainUICL + PuriDivER-style 的 clean/noisy/adaptive-attack 结果 |
| Pure 报告 | `PURE_PURIDIVER_EEG_REPORT.md` | 有标签/人工噪声条件下的核心方法验证 |

### 已验证结果

无标签 frozen-guide 主实验在 70/30 held-out 协议下的最终已见个体 ACC/MF1 为 `0.6781/0.5784`。在 BrainUICL-compatible 固定 old 集协议下，最终 old ACC/MF1 为 `0.6950/0.7014`，新个体刚适配后的平均 ACC/MF1 为 `0.6771/0.6062`。这些数字来自 compact EEG backbone，不应直接与 BrainUICL 大模型结果排名。

PuriDivER 对独立额外伪标签翻转表现出明确净化能力：在同一 frozen guide、同一 flip mask 和 memory 1000 条件下，PuriDivER 将加噪后的 final memory purity 从普通 replay 的 `58.0%` 提高到 `84.1%`。这支持“PuriDivER 的标签噪声鲁棒 replay 可以迁移到 EEG”。

高置信且模型内部自洽的输入投毒仍是失败边界。此类攻击会同时改变特征、预测和 loss 分布，使 GMM 难以将其与干净簇分开。PuriDivER-EEG 应定位为 noisy-label/pseudo-label contamination 防护，而不是任意 poisoning defense。

### 尚未完成

当前无标签 PuriDivER 使用 compact backbone，训练/测试为每个 subject 的 70/30 split，和 BrainUICL/CL-EEG 的全 sequence transductive 协议不同。若目标是算法排名，需要把 PuriDivER memory、GMM 和 C/R/U loss 接到同一 BrainUICL backbone，使用统一 final-seen/BWT；若目标只是证明方法可迁移到 EEG，当前 pure + unlabeled + noise 三组实验已经足够。

## BrainUICL

### 方法定位

BrainUICL 是原生 EEG 无监督个体持续学习框架，而不是从图像方法迁移的防御算法。它以 source-pretrained EEG 模型为起点，每个新个体克隆 CPC guiding model，使用高置信伪标签更新当前数据，并将部分序列写入包含 source 历史数据的 replay buffer，同时使用 CEA 约束表征漂移。

### 推荐代码与结果

| 类型 | 路径 | 用途 |
|---|---|---|
| 原始训练入口 | `trainer/trainer.py` | 原仓库 BrainUICL 训练流程 |
| 模型 | `model/pretrain_net.py` | FeatureExtractor、TransformerEncoder、SleepMLP |
| 可复现实验入口 | `experiments/rttdp_brainuicl_full.py` | clean/attack/defense、统一 final-seen/BWT、里程碑 checkpoint |
| 对齐 full49 | `experiments/rttdp_brainuicl_runs/aligned_full49_bn_frozen_lr1e6_seed4321/clean/metrics.json` | 与 CL-EEG 同数据、顺序、checkpoint、学习率、冻结 BN 和最终重评 |
| 对齐说明 | `experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/ALIGNED_COMPARISON_ZH.md` | BrainUICL 与五个无 replay 方法的公平性边界 |
| 流程文档 | `RTTDP_BRAINUICL_FULL_FLOW.md` | 输入、CPC、joint update、buffer、CEA 和攻击入口 |

### 已验证结果

对齐 full49 的最终旧个体 ACC/MF1 为 `0.7326/0.6995`，任务 49 后重评全部新个体得到 `0.6762/0.6067`，BWT ACC 为 `+0.0139`。44.23% 的新序列进入 replay，最终 buffer 为 1980 个 sequence，其中包含初始 source 历史数据。

该结果说明 BrainUICL 在允许 replay 与高置信选择的条件下，对跨个体 EEG 持续学习有效，并且当前统一指标下优于无 replay 正则化方法。它的代价是存储历史 EEG、依赖 source 标签和伪标签闭环；当模型被强攻击破坏后，高置信样本可能停止进入 buffer，或者错误高置信样本被长期 replay。

## CL-EEG

### 方法定位

CL-EEG 是当前仓库对传统参数正则化持续学习的 EEG 迁移线。网络和预训练 checkpoint 与 BrainUICL 相同，但不使用 replay buffer、CEA 或置信度过滤。每个新个体的 CPC guiding model 为全部 epoch 生成硬伪标签，Finetune、EWC、Online EWC、SI、MAS 只在当前个体上更新。

### 推荐代码与结果

| 类型 | 路径 | 用途 |
|---|---|---|
| 正则器 | `model/regularization_cl.py` | EWC、Online EWC、SI、MAS、BN 统计冻结 |
| 主入口 | `experiments/regularization_cl_eeg.py` | 49 个体干净 CL、统一 old/final-seen/BWT、checkpoint 和报告 |
| 攻击探针 | `experiments/regularization_cl_attacks.py` | PACOL-style 梯度匹配和 BrainWash-style 一步双层近似 |
| 干净 full49 | `experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/` | 五方法完整结果 |
| 主报告 | `experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/RESULTS.md` | 参数、结果、BN 消融和 BrainUICL 对齐表 |
| 中文对齐文档 | `experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/ALIGNED_COMPARISON_ZH.md` | 六方法同口径结果与资源差异 |
| 轻量攻击结果 | `experiments/regularization_cl_eeg_runs/attack_compare9_e10_lr1e6_seed4321/RESULTS.md` | PACOL/BrainWash 小规模适配与限制 |

### 已验证结果

| CL 算法 | 最终旧个体 ACC | 最终旧个体 MF1 | 最终新个体 ACC | 最终新个体 MF1 | BWT ACC |
|---|---:|---:|---:|---:|---:|
| Finetune | 60.65% | 59.02% | 54.89% | 47.12% | -9.41% |
| EWC | 69.02% | 66.14% | 62.85% | 53.21% | -2.40% |
| Online EWC | 69.51% | 66.69% | 63.67% | 53.91% | -1.03% |
| SI | 71.20% | 69.27% | 64.85% | 55.56% | -0.17% |
| MAS | 70.69% | 68.48% | 64.32% | 55.24% | -0.34% |

这些结果证明标准正则化 CL 可以迁移到无标签 EEG 个体流。SI 和 MAS 能在不保存历史 EEG 的情况下接近保持预训练模型，但其当前个体平均增益接近 0，优势主要是稳定而不是强适配。冻结学生 BatchNorm running statistics 是必要实现条件；不冻结时，即使参数正则很强，跨个体统计漂移仍会造成旧域性能突降。

PACOL/BrainWash 当前只是 classifier-scope、source-proxy 的轻量 EEG 适配，不是原论文完整复现。它们在 9-task 小实验中只对 Finetune 产生约 0.6 个 ACC 百分点影响，对 SI/MAS 影响很小；这一结果只能说明攻击接口和优化方向已跑通，不能用于证明普遍鲁棒性。

## 哪些结果可以放在同一张主表

目前只有 BrainUICL aligned 与 CL-EEG 五方法完成了以下全部对齐：同一 BrainUICL backbone、同一预训练 checkpoint、同一 split/order、相同学习率和 10+10 epochs、冻结学生 BN、任务 49 后重评全部 49 个体、相同 BWT 实现。因此六方法表可以作为当前论文的 clean native-method comparison，但必须附带 replay 大小和伪标签策略。

SPR-EEG 和 PuriDivER-EEG 暂时应单列为 method-transfer validation：SPR 使用不同的 expert/base/NT-Xent 训练预算和 epoch memory；PuriDivER 主线使用 compact backbone、70/30 held-out 和 memory 1000。直接把它们的已有 ACC 与六方法表排名会混淆 backbone、数据使用方式和评估集合。

## 下一步迁移顺序

1. 冻结当前 BrainUICL/CL-EEG 统一协议，作为 EEG-CL Protocol v1：同一 backbone、checkpoint、split/order、BN、final-seen 和 BWT。
2. 给 SPR-EEG 增加 Protocol v1 evaluator，保留 SPR expert/base 和 purified memory，只统一模型输出与最终评价。
3. 将 PuriDivER 的 memory、loss-GMM、uncertainty-GMM 和 C/R/U replay 接到 BrainUICL backbone，避免 compact model 与主表混比。
4. 为所有 replay 方法同时报告 memory unit、容量、source true-label 数量、target pseudo-label 覆盖率和最终 purity。
5. 先完成至少 3 个 checkpoint seeds 的 clean 比较，再固定防御和攻击参数；不能根据 attacked test 结果重新调 clean 超参数。
6. 攻击阶段分开报告随机标签污染与 adaptive input poisoning。SPR/PuriDivER 可作为前者的防护，不应预设它们能防住后者。

## 文档入口

- SPR 总报告：`SPR_EEG_DEFENSE_REPORT.md`
- SPR 无标签结果：`experiments/spr_eeg_unlabeled/RESULTS.md`
- PuriDivER pure 报告：`PURE_PURIDIVER_EEG_REPORT.md`
- PuriDivER 无标签报告：`UNLABELED_PURIDIVER_EEG_REPORT.md`
- PuriDivER 防御接入：`PURIDIVER_EEG_DEFENSE_REPORT.md`
- BrainUICL aligned 对比：`experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/ALIGNED_COMPARISON_ZH.md`
- CL-EEG 干净结果：`experiments/regularization_cl_eeg_runs/clean49_bn_frozen_e10_lr1e6_seed4321/RESULTS.md`
