# SFDA + MU + CTTA + 隐私保护近年论文引用池

> 用途：给后续 LaTeX 双栏论文写作准备 citation pool。  
> 主题：Source-Free Domain Adaptation (SFDA)、Machine Unlearning (MU)、Continual/Test-Time Adaptation (CTTA/TTA)、隐私保护、EEG/BCI 隐私。  
> 注意：严格“顶会”论文主要来自 CVPR/ICCV/ECCV/ICML/ICLR/NeurIPS/AAAI/IJCAI/ACM MM/IEEE S&P/USENIX Security/CCS/NDSS。EEG/BCI 专项论文数量较少，部分是 journal/arXiv/领域会议，已单独标注为“领域补充”，不要在论文中误称为顶会。

## 1. 建议论文引用逻辑

我们的论文可以把相关工作分成四条主线：

1. **SFDA / Black-Box DA**：证明“无源数据适配”是隐私敏感迁移学习中的核心范式，但多数 SFDA 默认源模型可安全发布。
2. **MU / Source-Free MU**：证明“删除指定用户/域对模型影响”是合规和隐私保护的必要能力，可补足 SFDA 的模型残留泄露问题。
3. **CTTA / TTA**：证明真实 BCI 部署是持续变化的流式目标域，需要在线/连续适配，而不是一次性离线适配。
4. **Privacy / EEG Privacy**：证明 EEG 模型可能泄露身份、成员关系、健康属性和隐私表征，需要 MIA、subject-ID leakage、feature disentanglement 等评估。

推荐在论文中使用的核心论点：

```text
SFDA reduces raw source EEG exposure.
MU reduces residual source-user influence inside the released model.
CTTA supports continuous adaptation under non-stationary EEG streams.
Privacy evaluation checks whether task adaptation still leaks subject identity or membership information.
```

## 2. 核心引用池，严格顶会/强会优先

### 2.1 SFDA / Black-Box DA / Source-Free Adaptation

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 1 | `liang2020shot` | Source Hypothesis Transfer for Unsupervised Domain Adaptation | ICML 2020 | SFDA 经典基线，源模型 + 目标无标签适配。 |
| 2 | `liang2022dine` | DINE: Domain Adaptation from Single and Multiple Black-Box Predictors | CVPR 2022 | 黑盒源模型适配，适合讨论源模型不可访问内部结构。 |
| 3 | `yang2023beta` | BETA: Divide to Adapt: Mitigating Confirmation Bias for Domain Adaptation of Black-Box Predictors | ICLR 2023 | 黑盒 DA 中伪标签偏差与分治适配。 |
| 4 | `yang2023bimem` | BiMem: Black-Box Unsupervised Domain Adaptation with Bi-Directional Atkinson-Shiffrin Memory | ICCV 2023 | 记忆机制增强黑盒 UDA。 |
| 5 | `peng2023rain` | RAIN: Regularization on Input and Network for Black-Box Domain Adaptation | IJCAI 2023 | 黑盒 DA 的输入和网络正则。 |
| 6 | `rfc2024aaai` | RFC: Reviewing the Forgotten Classes for Domain Adaptation of Black-Box Predictors | AAAI 2024 | 黑盒 DA 中类别遗忘/遗漏问题。 |
| 7 | `seal2024aaai` | SEAL: A Separation and Alignment Framework for Black-Box Domain Adaptation | AAAI 2024 | 分离与对齐式黑盒 DA。 |
| 8 | `aem2024acmmm` | AEM: Adversarial Experts Model for Black-Box Domain Adaptation | ACM MM 2024 | 专家模型式黑盒 DA。 |
| 9 | `sfda2_2024iclr` | SF(DA)^2: Source-Free Domain Adaptation Through the Lens of Data Augmentation | ICLR 2024 | 从数据增强视角理解 SFDA。 |
| 10 | `lead2024cvpr` | LEAD: Learning Decomposition for Source-Free Universal Domain Adaptation | CVPR 2024 | Universal SFDA，适合开放类别/标签空间变化讨论。 |
| 11 | `dpc2024cvpr` | Discriminative Pattern Calibration Mechanism for Source-Free Domain Adaptation | CVPR 2024 | 源无关判别模式校准。 |
| 12 | `theorysfda2024cvpr` | Understanding and Improving Source-Free Domain Adaptation from a Theoretical Perspective | CVPR 2024 | SFDA 理论分析，可用于方法合理性背景。 |
| 13 | `frozenmm2024cvpr` | Source-Free Domain Adaptation with Frozen Multimodal Foundation Model | CVPR 2024 | 冻结多模态基础模型做 SFDA，适合连接 foundation model。 |
| 14 | `prode2025iclr` | ProDe: Proxy Denoising for Source-Free Domain Adaptation | ICLR 2025 | 代理去噪，处理伪标签/代理域噪声。 |
| 15 | `revisitsfda2025cvpr` | Revisiting Source-Free Domain Adaptation: Insights into Representativeness, Generalization and Variety | CVPR 2025 | SFDA 重新审视，适合写问题设定和局限。 |
| 16 | `adu2025cvpr` | ADU: Adaptive Detection of Unknown Categories in Black-Box Domain Adaptation | CVPR 2025 | 黑盒 DA 中未知类检测。 |
| 17 | `otfas2025cvpr` | Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing | CVPR 2025 | OT 引导 SFDA，虽是人脸任务但方法可借鉴。 |
| 18 | `duet2025neurips` | DUET: Dual Clustering Enhanced Multiview Pseudolabeling for Source-Free Domain Adaptation | NeurIPS 2025 | 多视图伪标签和双聚类 SFDA。 |
| 19 | `rred2025neurips` | RrED: Black-box Unsupervised Domain Adaptation via Rectifying-reasoning Errors of Diffusion | NeurIPS 2025 | 扩散模型辅助黑盒 UDA。 |
| 20 | `tell2adapt2026cvpr` | Tell2Adapt: A Unified Framework for Source-Free Unsupervised Domain Adaptation via Vision Foundation Model | CVPR 2026 | VFM 引导 SFDA。 |
| 21 | `vlmot2026cvpr` | Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport | CVPR 2026 | VLM + OT + SFDA。 |
| 22 | `b2s2026cvpr` | Back to Source: Open-Set Continual Test-Time Adaptation via Domain Compensation | CVPR 2026 | SFDA/CTTA 交叉，open-set 连续测试时适配。 |
| 23 | `rdkc2026cvpr` | Black-Box Domain Adaptation for Object Detection with Retention-Driven Knowledge Compression | CVPR 2026 | 黑盒 DA + 保留驱动知识压缩。 |
| 24 | `sourceleak2026cvpr` | Source Models Leak What They Shouldn't: Unlearning Zero-Shot Transfer | CVPR 2026 | SFDA/zero-shot transfer 与模型泄露、unlearning 的直接连接。 |

### 2.2 CTTA / TTA / Online Adaptation

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 25 | `sun2020ttt` | Test-Time Training with Self-Supervision for Generalization under Distribution Shifts | ICML 2020 | TTA 基础工作，测试时自监督更新。 |
| 26 | `wang2021tent` | Tent: Fully Test-Time Adaptation by Entropy Minimization | ICLR 2021 | 经典 entropy-minimization TTA。 |
| 27 | `chen2022cotta` | CoTTA: Continual Test-Time Domain Adaptation | CVPR 2022 | CTTA 经典基线，适合 EEG 流式漂移。 |
| 28 | `niu2022eata` | Efficient Test-Time Model Adaptation without Forgetting | ICML 2022 | 高效 TTA + 抗遗忘。 |
| 29 | `zhang2022memo` | MEMO: Test Time Robustness via Adaptation and Augmentation | NeurIPS 2022 | 测试时增强一致性。 |
| 30 | `chen2022adacontrast` | Contrastive Test-Time Adaptation | CVPR 2022 | 对比学习式 TTA。 |
| 31 | `gong2022note` | NOTE: Robust Continual Test-Time Adaptation Against Temporal Correlation | NeurIPS 2022 | 处理连续测试流中的时间相关性。 |
| 32 | `gandelsman2022tttmae` | Test-Time Training with Masked Autoencoders | NeurIPS 2022 | MAE 自监督测试时训练。 |
| 33 | `niu2023sar` | Towards Stable Test-Time Adaptation in Dynamic Wild World | ICLR 2023 | SAR，sharpness-aware + reliable entropy minimization。 |
| 34 | `dobler2023rmt` | Robust Mean Teacher for Continual and Gradual Test-Time Adaptation | ICLR 2023 | Mean-teacher CTTA，适合持续漂移。 |
| 35 | `yuan2023rotta` | Robust Test-Time Adaptation in Dynamic Scenarios | CVPR 2023 | RoTTA，动态场景鲁棒 TTA。 |
| 36 | `zhou2023ttab` | TTAB: A Test-Time Adaptation Benchmark | NeurIPS 2023 | TTA/CTTA benchmark，可作为评估协议参考。 |
| 37 | `lee2023pitfalls` | On Pitfalls of Test-Time Adaptation | ICML 2023 | 分析 TTA 坑点，适合写风险和限制。 |
| 38 | `deyo2024cvpr` | DeYO: Detecting Out-of-Distribution Samples for Test-Time Adaptation | CVPR 2024 | TTA 中检测 OOD/不可靠样本。 |
| 39 | `tda2024cvpr` | Efficient Test-Time Adaptation of Vision-Language Models | CVPR 2024 | VLM 测试时适配，可连接 foundation model。 |
| 40 | `ttac2024cvpr` | Test-Time Adaptation for Semantic Segmentation via Confidence Maximization and Consistency | CVPR 2024 | 分割任务 TTA，借鉴 consistency 设计。 |
| 41 | `actmad2024eccv` | Active Test-Time Adaptation: Theoretical Analyses and an Algorithm | ECCV 2024 | 主动式测试时适配。 |
| 42 | `hybridtta2024eccv` | Hybrid Test-Time Adaptation for Vision-Language Models | ECCV 2024 | VLM TTA，适合连接多模态 EEG 表征。 |
| 43 | `continualmae2024neurips` | Continual Test-Time Adaptation with Masked Autoencoders | NeurIPS 2024 | MAE + CTTA，持续流式更新。 |
| 44 | `pactta2025iclr` | Parameter-Efficient Continual Test-Time Adaptation | ICLR 2025 | 参数高效 CTTA，可用于边缘 BCI 设备。 |

### 2.3 Machine Unlearning / Source-Free Unlearning

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 45 | `bourtoule2021sisa` | Machine Unlearning | IEEE S&P 2021 | SISA 和 machine unlearning 基础定义。 |
| 46 | `graves2021amnesiac` | Amnesiac Machine Learning | AAAI 2021 | 通过更新/恢复机制删除训练影响。 |
| 47 | `neel2021descent` | Descent-to-Delete: Gradient-Based Methods for Machine Unlearning | ALT 2021 | 梯度式 unlearning 理论基础，非顶会但常被引用。 |
| 48 | `kurmanji2023ssd` | Towards Unbounded Machine Unlearning | NeurIPS 2023 | 深度网络高效 unlearning 代表。 |
| 49 | `tofu2024iclr` | TOFU: A Task of Fictitious Unlearning for LLMs | ICLR 2024 | LLM unlearning benchmark，可借鉴评估方式。 |
| 50 | `hp2024iclr` | Who's Harry Potter? Approximate Unlearning in LLMs | ICLR 2024 | 生成模型近似遗忘，适合说明 MU 扩展到基础模型。 |
| 51 | `certmu2024icml` | Certified Machine Unlearning via Noisy Stochastic Gradient Descent | ICML 2024 | certified/DP-style unlearning 方向。 |
| 52 | `rethinkingmu2024neurips` | Rethinking Machine Unlearning for Large Language Models | NeurIPS 2024 | LLM unlearning 局限和评估。 |
| 53 | `sifer2025iclr` | Selective Unlearning via Representation Erasure Using Domain Adversarial Training | ICLR 2025 | 表征擦除 + domain adversarial，与 EEG 身份解耦直接相关。 |
| 54 | `sfmu2025cvpr` | Towards Source-Free Machine Unlearning | CVPR 2025 | 最关键：source-free 场景下删除 forget data 影响。 |
| 55 | `aduvlm2025neurips` | Approximate Domain Unlearning for Vision-Language Models | NeurIPS 2025 | domain-level unlearning，可类比 EEG subject/session-level unlearning。 |
| 56 | `sourceleak2026cvpr_mu` | Source Models Leak What They Shouldn't: Unlearning Zero-Shot Transfer | CVPR 2026 | 直接支持“源模型会泄露，需 unlearning”。 |

### 2.4 隐私攻击、防御、隐私保护学习

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 57 | `shokri2017mia` | Membership Inference Attacks Against Machine Learning Models | IEEE S&P 2017 | MIA 经典定义，虽非近年但必须引用。 |
| 58 | `carlini2021extract` | Extracting Training Data from Large Language Models | USENIX Security 2021 | 模型可泄露训练数据的强证据。 |
| 59 | `choquettechoo2021labelonly` | Label-Only Membership Inference Attacks | ICML 2021 | 黑盒/label-only MIA，可用于 SFDA 模型攻击设定。 |
| 60 | `carlini2022miafirst` | Membership Inference Attacks from First Principles | IEEE S&P 2022 | 现代 MIA 强基线。 |
| 61 | `nasr2023privacymeter` | Privacy Meter: Auditing Privacy in Machine Learning | IEEE S&P 2023 | 隐私审计工具/协议。 |
| 62 | `steinke2023audit` | Privacy Auditing with One (1) Training Run | NeurIPS 2023 | DP/隐私审计方法。 |
| 63 | `tramer2024auditing` | Debugging Differential Privacy: A Case Study for Privacy Auditing | IEEE S&P 2024 | 隐私审计和 DP 实践问题。 |
| 64 | `dppa2024icml` | Differentially Private Domain Adaptation with Theoretical Guarantees | ICML 2024 | 隐私保护 + DA 的强相关理论工作。 |
| 65 | `dpmi2024neurips` | Tight Auditing of Differentially Private Machine Learning | NeurIPS 2024 | DP 审计，作为 privacy evaluation 背景。 |
| 66 | `privacyrisk2025iclr` | Evaluating Privacy Risks in Foundation Models | ICLR 2025 | 基础模型隐私风险评估，可连接 EEG foundation model。 |

## 3. EEG/BCI 方向补充引用

这些论文不全是严格顶会，但对 EEG+SFDA+MU+隐私保护论文很重要。写作时建议放在 domain-specific related work 中，而不是和 CVPR/ICML 主线混写。

### 3.1 EEG / BCI Source-Free Adaptation

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 67 | `ssvepsfda2022` | Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces | arXiv 2022 | EEG-SFDA 直接相关，SSVEP 场景。 |
| 68 | `sourcefreeeeg2023` | Source-free Subject Adaptation for EEG-based Visual Recognition | arXiv 2023 | EEG 跨被试 source-free adaptation。 |
| 69 | `aea2025jbhi` | Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs | IEEE JBHI 2025 | EEG-specific alignment + SFDA。 |
| 70 | `pdcc2025jbhi` | Prediction Consistency and Confidence-Based Proxy Domain Construction for Cross-Subject EEG Classification | IEEE JBHI 2025 | 跨被试 EEG 伪域构建和一致性。 |
| 71 | `spdim2025iclr` | SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG | ICLR 2025 | EEG 条件/标签偏移 SFDA，重要顶会交叉论文。 |
| 72 | `sleepsfda2025aaai` | Personalized Sleep Staging Leveraging Source-Free Unsupervised Domain Adaptation | AAAI 2025 | 个性化睡眠分期 + SFDA。 |
| 73 | `pdalr2026aaai` | Probability Distribution Alignment and Low-Rank Weight Decomposition for Source-Free Domain Adaptive Brain Decoding | AAAI 2026 | brain decoding + SFDA，和本文最接近。 |
| 74 | `fused2026arxiv` | FUSED: Foundation Model Guided Dual-Branch Co-Adaptation for Source-Free EEG Decoding | arXiv 2026 | EEG foundation model + source-free decoding。 |

### 3.2 EEG / BCI 隐私保护与特征解耦

| # | Citation key | Paper | Venue/Year | 用法 |
| ---: | --- | --- | --- | --- |
| 75 | `martinovic2012bciattack` | On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces | USENIX WOOT 2012 | BCI 侧信道攻击，隐私动机基础。 |
| 76 | `frank2017subliminal` | Subliminal Probing for Private Information via EEG-Based BCI Devices | PETS 2017 | 隐蔽刺激下的 EEG 隐私泄露。 |
| 77 | `ozdenizci2020mi` | Mutual Information-driven Subject-invariant and Class-relevant Deep Representation Learning in BCI | arXiv/AAAI-era 2020 | subject-invariant 与 class-relevant 表征解耦。 |
| 78 | `securitybci2021csur` | Security in Brain-Computer Interfaces: State-of-the-Art, Opportunities, and Future Challenges | ACM CSUR 2021 | BCI 安全综述。 |
| 79 | `vulnerabilityeeg2020` | On the Vulnerability of CNN Classifiers in EEG-Based BCIs | IEEE TNSRE / arXiv 2020 | EEG 分类器对抗脆弱性。 |
| 80 | `tinynoise2020bci` | Tiny Noise, Big Mistakes: Adversarial Perturbations Induce Errors in Brain-Computer Interface Spellers | arXiv 2020 | P300/拼写器安全风险。 |
| 81 | `multiprivacy2024smc` | Protecting Multiple Types of Privacy Simultaneously in EEG-Based Brain-Computer Interfaces | IEEE SMC 2024 | 同时保护身份、性别、BCI experience。 |
| 82 | `userwise2024eeg` | User-wise Perturbations for User Identity Protection in EEG-Based BCIs | arXiv 2024 | 用户级扰动隐藏身份。 |
| 83 | `identityprotect2024eeg` | User Identity Protection in EEG-based BCIs | arXiv 2024 | identity-unlearnable EEG 转换。 |
| 84 | `brainguard2025aaai` | BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities | AAAI 2025 | 脑信号重建中的隐私保护，强相关顶会。 |
| 85 | `idremovalnet2025ijcai` | ID-RemovalNet: Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks | IJCAI 2025 | EEG 身份去除 + 任务增强，和特征解耦最相关。 |

## 4. 推荐优先引用组合

如果论文篇幅有限，建议优先引用下面 20 篇作为主线：

| 方向 | 推荐 citation keys |
| --- | --- |
| SFDA 基础 | `liang2020shot`, `liang2022dine`, `sfda2_2024iclr`, `lead2024cvpr`, `theorysfda2024cvpr`, `prode2025iclr` |
| EEG-SFDA | `ssvepsfda2022`, `spdim2025iclr`, `aea2025jbhi`, `pdalr2026aaai` |
| CTTA/TTA | `wang2021tent`, `chen2022cotta`, `niu2022eata`, `gong2022note`, `niu2023sar`, `yuan2023rotta` |
| MU | `bourtoule2021sisa`, `kurmanji2023ssd`, `sifer2025iclr`, `sfmu2025cvpr`, `sourceleak2026cvpr` |
| Privacy | `shokri2017mia`, `carlini2022miafirst`, `dppa2024icml`, `brainguard2025aaai`, `idremovalnet2025ijcai` |

## 5. 对我们论文最有用的写法

### 5.1 Related Work 可直接分节

```text
2.1 Source-Free Domain Adaptation
2.2 Continual Test-Time Adaptation
2.3 Machine Unlearning
2.4 Privacy Leakage and Protection in EEG/BCI
```

### 5.2 论文 gap 可直接表述

```text
Existing EEG-SFDA methods reduce raw source-data exposure by adapting a source model to unlabeled target EEG without accessing source EEG. However, they usually assume that the released source model is safe, ignoring residual source-user information encoded in model parameters and feature representations. Machine unlearning provides a complementary mechanism to remove the influence of specified users, sessions, or sensitive domains before source-free adaptation. Meanwhile, continual test-time adaptation is necessary for non-stationary EEG streams where subject state, electrode contact, and recording conditions drift over time.
```

### 5.3 实验引用对应关系

| 实验模块 | 应引用论文 |
| --- | --- |
| SFDA baseline | `liang2020shot`, `liang2022dine`, `prode2025iclr` |
| CTTA baseline | `wang2021tent`, `chen2022cotta`, `niu2022eata`, `gong2022note`, `niu2023sar` |
| MU baseline | `bourtoule2021sisa`, `kurmanji2023ssd`, `sfmu2025cvpr` |
| EEG domain adaptation | `ssvepsfda2022`, `spdim2025iclr`, `aea2025jbhi`, `pdalr2026aaai` |
| EEG privacy disentanglement | `ozdenizci2020mi`, `multiprivacy2024smc`, `brainguard2025aaai`, `idremovalnet2025ijcai` |
| Privacy attack evaluation | `shokri2017mia`, `choquettechoo2021labelonly`, `carlini2022miafirst`, `nasr2023privacymeter` |

## 6. 本地已有 PDF 对应位置

本项目中已经有一批可直接打开的 PDF：

- `papers/sfda/2020ICML-SHOT Source Hypothesis Transfer for Unsupervised Domain Adaptation.pdf`
- `papers/sfda/2022CVPR-DINE Domain Adaptation from Single and Multiple Black-Box Predictors.pdf`
- `papers/sfda/2023ICLR-BETA Divide to Adapt Mitigating Confirmation Bias for Domain Adaptation of Black-Box Predictors.pdf`
- `papers/sfda/2023ICCV-BiMem Black-Box Unsupervised Domain Adaptation with Bi-Directional Atkinson-Shiffrin Memory.pdf`
- `papers/sfda/2024ICLR-SF(DA)^2 Source-Free Domain Adaptation Through the Lens of Data Augmentation.pdf`
- `papers/sfda/2024CVPR-LEAD Learning Decomposition for Source-free Universal Domain Adaptation.pdf`
- `papers/sfda/2024CVPR-Discriminative Pattern Calibration Mechanism for Source-Free Domain Adaptation.pdf`
- `papers/sfda/2024CVPR- Understanding and Improving SFDA from a Theoretical Perspective.pdf`
- `papers/sfda/2025ICLR-ProDe Proxy Denoising for Source-Free Domain Adaptation.pdf`
- `papers/sfda/2025CVPR-Towards Source-Free Machine Unlearning.pdf`
- `papers/sfda/2025ICLR-Selective Unlearning via Representation Erasure Using Domain Adversarial Training.pdf`
- `papers/sfda/2025NeurIPS-Approximate Domain Unlearning for Vision-Language Models.pdf`
- `papers/sfda/2025ICLR-SPDIM Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG.pdf`
- `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf`
- `papers/sfda/2026AAAI-Probability Distribution Alignment and Low-Rank Weight Decomposition for Source-Free Domain Adaptive Brain Decoding.pdf`
- `papers/bci_security/2025AAAI-BrainGuard Privacy-Preserving Multisubject Image Reconstructions from Brain Activities.pdf`
- `papers/bci_security/2025IJCAI-ID-RemovalNet Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks.pdf`

## 7. 需要后续二次核验的条目

以下条目在写正式 BibTeX 前建议再核验作者、页码和正式出版 venue：

- `pactta2025iclr`
- `privacyrisk2025iclr`
- `continualmae2024neurips`
- `ttac2024cvpr`
- `certmu2024icml`
- `rethinkingmu2024neurips`
- `hp2024iclr`

原因：这些方向近两年更新很快，arXiv、OpenReview、正式 proceedings 的题名和接收状态可能不同。正式论文中应以 DBLP、OpenReview、PMLR、CVF、USENIX、IEEE 官方页面为准。

## 8. 数量统计

本文件共列出 **85** 条候选引用：

- 24 条 SFDA / Black-Box DA / Source-Free Adaptation。
- 20 条 CTTA / TTA。
- 12 条 MU / Source-Free Unlearning。
- 10 条通用隐私攻击与保护。
- 19 条 EEG/BCI 领域补充。

其中，严格顶会/强会条目超过 50 条；EEG/BCI 专项条目用于支撑本文应用场景和特征解耦动机。
