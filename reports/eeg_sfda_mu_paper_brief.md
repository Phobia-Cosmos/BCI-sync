# EEG + SFDA + MU 隐私保护论文写作交接 Brief

> 目标：给另一个 Codex 直接使用，用于起草一篇双栏 LaTeX 论文初稿。当前阶段只需要写清楚 introduction、background、method design、experiment design，不需要真的跑完整实验，也不要编造实验数值。

## 1. 一句话论文定位

本文希望讨论一个隐私保护型 EEG/BCI 学习框架：在跨被试 EEG 解码中，使用 Source-Free Domain Adaptation (SFDA) 避免目标适配阶段访问源用户原始 EEG，同时引入 Machine Unlearning (MU) 从已训练源模型中删除指定用户、会话或敏感域的影响，从而缓解“只做 SFDA 仍可能通过源模型泄露隐私”的问题。

核心观点：

- EEG 是高敏感神经数据，可能泄露身份、年龄、健康状态、熟悉信息和任务意图。
- 跨被试 EEG 存在明显 domain shift，实际 BCI 系统需要跨用户/跨会话适配。
- SFDA 能减少原始源数据共享，但源模型参数仍可能携带源用户信息。
- MU 可作为 SFDA 的补充：当某个用户撤回授权，或某类敏感数据不应继续影响模型时，从模型中删除其影响，再进行目标无源适配。

## 2. 推荐论文标题

优先标题：

**Towards Privacy-Preserving EEG Adaptation via Source-Free Domain Adaptation and Machine Unlearning**

备选标题：

- **Source-Free and Forgettable EEG Decoding for Privacy-Preserving Brain-Computer Interfaces**
- **Can Source-Free Adaptation Protect EEG Privacy? A Machine Unlearning Perspective**
- **Privacy-Aware Cross-Subject EEG Decoding with Source-Free Adaptation and User-Level Unlearning**

## 3. 论文要回答的问题

主问题：

> 在跨被试 EEG/BCI 任务中，只使用 SFDA 是否足以保护源用户隐私？如果不够，是否可以通过 MU 删除源模型中的敏感用户/域影响，同时保持目标用户解码性能？

可拆成三个研究问题：

1. **RQ1: Utility**  
   在没有源 EEG 原始数据的情况下，SFDA 能否让模型适配到新目标用户？

2. **RQ2: Privacy Leakage**  
   即使源数据不可见，源模型是否仍会泄露源用户成员关系、身份特征或敏感域信息？

3. **RQ3: Unlearning Trade-off**  
   对源模型执行 user/session-level MU 后，能否降低隐私攻击成功率，同时尽量保持目标适配性能？

## 4. 背景与动机

### 4.1 EEG/BCI 为什么需要隐私保护

EEG/BCI 数据不是普通传感器数据。已有研究说明，脑电可用于身份识别、年龄/健康状态预测、熟悉信息推断和偏好推断。因此，在 BCI 系统中长期保存或共享原始 EEG 会带来神经隐私风险。

可以使用的论据：

- EEG 信号可用于 biometrics / brainprint，说明存在个体可识别特征。
- EEG 可被用于 brain age 或健康状态预测，说明医学属性可能泄露。
- BCI 侧信道攻击和隐蔽刺激研究说明，即便用户没有主动输入，脑电响应也可能泄露敏感信息。
- 神经数据隐私立法，例如 Colorado HB24-1058，说明监管层已将 neural data 视为敏感数据。

本地已有支撑材料：

- `reports/bci_privacy_domain_adaptation_risks.md`
- `papers/bci_security/2021CSUR-Security in Brain-Computer Interfaces State-of-the-Art Opportunities and Future Challenges.pdf`
- `papers/bci_security/2024arXiv-Protecting Multiple Types of Privacy Simultaneously in EEG-based Brain-Computer Interfaces.pdf`
- `papers/bci_security/2025IJCAI-ID-RemovalNet Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks.pdf`
- `papers/bci_security/2025AAAI-BrainGuard Privacy-Preserving Multisubject Image Reconstructions from Brain Activities.pdf`

### 4.2 EEG 为什么需要 Domain Adaptation

跨被试 EEG 解码中，训练用户和目标用户之间存在显著分布偏移。偏移来源包括：

- 不同用户脑结构、头皮传导、电极接触和信噪比不同。
- 同一用户不同会话之间存在疲劳、注意力、电极位置和阻抗变化。
- 不同设备、采样率、通道 montage 和预处理流程不同。
- 实验室数据与真实 BCI 场景不同。

如果不做适配，模型在目标用户上会性能下降。对于医疗 BCI、康复反馈、拼写器或控制设备，这种下降可能从普通分类误差变成安全风险。

本地已有支撑材料：

- `reports/bci_sfda_inventory_and_plan.md`
- `papers/sfda/2020ICML-SHOT Source Hypothesis Transfer for Unsupervised Domain Adaptation.pdf`
- `papers/sfda/2022arXiv-Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces.pdf`
- `papers/sfda/2023arXiv-Source-free Subject Adaptation for EEG-based Visual Recognition.pdf`
- `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf`
- `papers/sfda/2025JBHI-Prediction Consistency and Confidence-Based Proxy Domain Construction for Cross-Subject EEG Classification.pdf`
- `papers/sfda/2025ICLR-SPDIM Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG.pdf`

### 4.3 只做 SFDA 的不足

SFDA 的隐私价值是：目标适配时不需要访问源域原始数据。  
但这不是完整隐私保证，原因是：

- 源模型本身可能记住源用户数据分布。
- 攻击者可能通过输出置信度、loss、embedding 或中间特征进行 membership inference。
- EEG embedding 可能仍保留 subject identity。
- 如果某个源用户要求删除数据，单纯“以后不再访问原始数据”并不能删除已训练模型中的影响。

因此，论文可以提出一个清晰 gap：

> Existing EEG SFDA methods reduce raw-data sharing but do not explicitly address the residual privacy leakage carried by the source model. Machine unlearning provides a natural mechanism to remove user- or session-level influence before source-free target adaptation.

## 5. 方法设计

方法名称建议：**SFDA-MU EEG** 或 **Forget-and-Adapt EEG**。

### 5.1 问题定义

设 EEG 数据为：

- 源域：多个源用户或源会话  
  \(D_s = \{(x_i^s, y_i^s, u_i^s)\}\)，其中 \(u_i\) 表示 subject/session/domain id。
- 目标域：新目标用户无标签 EEG  
  \(D_t = \{x_j^t\}\)。
- 待遗忘数据：某个源用户、会话或敏感子集  
  \(D_f \subset D_s\)。
- 保留数据：\(D_r = D_s \setminus D_f\)，但在 source-free unlearning 设定下不可访问。

任务目标：

1. 删除 \(D_f\) 对源模型的影响。
2. 不访问 \(D_s\) 或 \(D_r\) 原始数据。
3. 使用无标签 \(D_t\) 做目标适配。
4. 保持目标任务性能，同时降低隐私攻击成功率。

### 5.2 模型结构

可采用简单 EEG backbone，避免方法设计太重：

- Encoder \(G_\theta\)：EEGNet、ShallowConvNet、TSMNet-like covariance backbone，或轻量 CNN。
- Classifier \(C_\phi\)：任务分类头，例如 MI 左/右手、P300 target/non-target、SSVEP target class。
- Optional privacy head \(A_\psi\)：用于评估或对抗训练的 subject-id classifier。

模型输出：

```text
EEG epoch x -> feature z = G(x) -> task prediction y_hat = C(z)
```

### 5.3 阶段 1：源模型训练

使用源用户有标签 EEG 训练基础模型：

```text
theta_s = Train(G, C; D_s)
```

训练损失：

```text
L_source = CE(C(G(x_s)), y_s)
```

这一步在真实隐私设定中由源机构完成。目标适配方之后只能拿到源模型，不能拿到源 EEG。

### 5.4 阶段 2：Source-Free Machine Unlearning

输入：

- 已训练源模型 \(M_s = G_{\theta_s}, C_{\phi_s}\)。
- 待遗忘数据 \(D_f\)。
- 不可访问 \(D_r\)。

输出：

- 已遗忘模型 \(M_u = G_{\theta_u}, C_{\phi_u}\)。

简单可写的 MU 设计：

1. **Forget-gradient step**  
   对待遗忘样本进行梯度上升或随机标签训练，使模型降低对 \(D_f\) 的记忆。

2. **Source-free retain surrogate**  
   由于不能访问 \(D_r\)，用源模型自身、扰动样本、参数正则或 Hessian 近似约束模型不要整体崩坏。

3. **Feature privacy regularization**  
   可选：对 \(D_f\) 的 subject-id 特征做混淆，使 embedding 难以识别该用户。

可写成抽象目标：

```text
L_MU = - CE(C(G(x_f)), y_f)
       + lambda_w ||theta - theta_s||_2^2
       + lambda_kd KL(M_u(aug(x_f)) || M_s(aug(x_f)))
       + lambda_priv L_subject_confusion
```

注意：如果论文不打算实现复杂 MU，可以把 MU 作为设计模块，实验中比较几类 baseline：

- NegGrad：只对 forget data 做梯度上升。
- Random Labels：对 forget data 分配错误标签再微调。
- Source-free Hessian / perturbation-based MU：参考 `2025CVPR-Towards Source-Free Machine Unlearning` 的思想。
- Oracle Retrain：用 \(D_r\) 重新训练，作为上界，不属于 source-free。

本地已有 MU 代码理解：

- `code/sfda/2025CVPR-source-free-unlearning/docs/source_free_unlearning_code_logic.md`
- `code/sfda/2025CVPR-source-free-unlearning/linear_repro.py`
- `code/sfda/2025CVPR-source-free-unlearning/baseline_repro.py`

### 5.5 阶段 3：Source-Free Domain Adaptation

输入：

- 已遗忘源模型 \(M_u\)。
- 无标签目标 EEG \(D_t\)。
- 不访问源 EEG。

输出：

- 目标适配模型 \(M_t\)。

可采用 SHOT-style SFDA：

- 固定或部分固定 classifier \(C\)。
- 使用目标样本的信息最大化、熵最小化和多样性约束。
- 用目标伪标签更新 feature extractor。
- 加 EEG-specific alignment，例如 Euclidean Alignment 或 covariance recentering。

目标适配损失可写为：

```text
L_SFDA = L_entropy(D_t)
         + beta L_diversity(D_t)
         + gamma L_pseudo_label(D_t)
         + eta L_consistency(aug1(x_t), aug2(x_t))
```

EEG-specific preprocessing / alignment：

- Motor imagery: 8-30 Hz bandpass, epoch 0.5-4 s, Euclidean Alignment or covariance log-space alignment。
- P300: 0.1-30 Hz, epoch -0.2-0.8 s, baseline correction。
- SSVEP: occipital channels, target frequency/phase label, harmonics-aware filtering。

### 5.6 整体流程

推荐在论文中画一个三阶段 pipeline：

```text
Labeled source EEG
    -> train source EEG decoder
    -> user/session-level machine unlearning
    -> unlearned source model
    -> unlabeled target EEG source-free adaptation
    -> privacy-aware target EEG decoder
```

关键对比：

```text
SFDA only:
source model -> target adaptation

SFDA + MU:
source model -> remove sensitive/withdrawn user influence -> target adaptation
```

## 6. 实验设计

当前阶段不需要真的跑实验，但论文方法部分应写出可执行的实验设定。

### 6.1 最小可行实验

建议最小实验只做一个公开 EEG 数据集：

**Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMI)**

原因：

- 公开、常见、可做 motor imagery 分类。
- 每个 subject 可作为一个 domain。
- 可以模拟跨被试 SFDA。
- 可以指定一个 subject/session 作为 forget domain。

任务：

- Binary motor imagery classification：left fist vs right fist。
- Source domains：多个源 subjects。
- Target domain：leave-one-subject-out 的目标 subject。
- Forget domain：从源 subjects 中选一个 subject 或一个 session。

最小实验流程：

1. 用源 subjects 训练 EEG decoder。
2. 选择一个源 subject 作为 \(D_f\)，执行 MU。
3. 在目标 subject 的无标签 EEG 上执行 SFDA。
4. 用目标 subject 标签只做最终评估。
5. 训练隐私攻击器评估 forget subject 和 subject identity 泄露。

本地已有数据/清单：

- `reports/bci_sfda_inventory_and_plan.md`
- `data/processed/manifests/physionet_eegmmi_S001_annotations.csv` 如果在另一个机器不存在，可按该报告重新生成。

### 6.2 可扩展实验

如果论文需要更像完整实验，可以加两个数据集：

1. **Tsinghua SSVEP Benchmark**  
   用于多类 SSVEP 识别，subject-level adaptation。

2. **OpenNeuro P300 / ERP dataset**  
   用于 P300 target/non-target 二分类，强调医疗/辅助沟通场景。

不建议初稿一上来写太多数据集，否则实验承诺过重。双栏初稿可以先写 one primary dataset + optional extension。

### 6.3 Baselines

主 baseline：

| Method | 说明 |
| --- | --- |
| Source Only | 源模型直接测试目标用户，不做适配 |
| SFDA Only | 只做 source-free target adaptation，不做 unlearning |
| MU Only | 只做 machine unlearning，不做目标适配 |
| NegGrad + SFDA | 简单遗忘后做 SFDA |
| Random Label + SFDA | 随机标签遗忘后做 SFDA |
| Proposed MU-SFDA | 本文方法 |
| Oracle Retrain + SFDA | 用 retain data 重训后做 SFDA，上界，不满足 source-free |
| Supervised Target | 使用目标标签训练/微调，上界，不满足无标签设定 |

### 6.4 Utility Metrics

任务性能指标：

- Accuracy。
- Balanced Accuracy，适合类别不均衡的 P300。
- Macro-F1。
- Target adaptation gain：`Acc(method) - Acc(source-only)`。
- 对 SSVEP 可选 ITR，但初稿可以不强制。

稳定性指标：

- 不同 target subject 的 mean ± std。
- 不同 forget subject 的 mean ± std。
- Calibration / confidence，例如 ECE，可选。

### 6.5 Privacy / Forgetting Metrics

隐私指标要比单纯 accuracy 更关键。

建议至少设计三类：

1. **Membership Inference Attack (MIA)**  
   攻击者根据模型输出 confidence/loss 判断某 EEG epoch 是否来自训练集。  
   指标：attack accuracy、AUC。  
   期望：MU 后 MIA 接近随机，即 AUC 接近 0.5。

2. **Subject Identity Leakage**  
   冻结 encoder，用 embedding 训练 subject-id classifier，判断特征是否保留用户身份。  
   指标：subject-id accuracy。  
   期望：MU 或 privacy regularization 后，forget subject 的可识别性下降。

3. **Forgetting Gap**  
   比较 unlearned model 与 oracle retrain model 在 forget data 和 retain/target data 上的行为差异。  
   指标可以是 prediction KL、confidence difference、forget accuracy drop、retain accuracy preservation。

注意措辞：

- 如果没有正式差分隐私或 certified unlearning，不要声称“guaranteed privacy”。
- 应写成“privacy leakage mitigation”或“empirical privacy protection”。

### 6.6 Ablation Study

推荐 ablation：

| Ablation | 目的 |
| --- | --- |
| w/o MU | 验证只做 SFDA 是否仍有隐私泄露 |
| w/o SFDA | 验证只遗忘无法解决目标域性能 |
| w/o EEG alignment | 验证 EEG-specific alignment 的作用 |
| different forget ratios | 验证忘记一个 subject、一个 session、部分 trials 的影响 |
| different target unlabeled sizes | 验证少量目标 EEG 下是否稳定 |
| different privacy weights | 画 privacy-utility trade-off |

### 6.7 论文中可放的表和图

Figure 1：整体 pipeline  

- 左侧：source users train model。
- 中间：forget request triggers MU。
- 右侧：unlabeled target user performs SFDA。
- 下方：privacy attacks evaluate leakage。

Table 1：主实验结果  

列：

```text
Method | Target Acc | Target F1 | MIA AUC | Subject-ID Acc | Forget Gap
```

Table 2：ablation  

```text
Method Variant | Target Acc | MIA AUC | Subject-ID Acc
```

Figure 2：privacy-utility trade-off  

横轴：MIA AUC 或 subject-id accuracy。  
纵轴：target accuracy。  
目标：越靠左上越好。

## 7. Introduction 写作逻辑

建议 introduction 按 5 段写：

1. **BCI/EEG 应用重要性**  
   EEG-based BCI 正在用于运动想象、SSVEP、P300 拼写器、康复和情绪/认知状态监测。为了个性化和跨用户部署，模型通常需要利用多用户 EEG 数据。

2. **隐私问题**  
   EEG 信号包含高度敏感的神经和医学信息，可能泄露身份、健康状态、熟悉刺激和用户意图。直接集中训练或共享原始 EEG 在实际医疗/消费 BCI 中不可接受。

3. **SFDA 的机会与不足**  
   SFDA 允许模型在没有源数据的情况下适配到目标用户，天然适合隐私敏感 EEG 场景。但 SFDA 通常默认源模型可以安全发布，忽略了源模型本身可能携带源用户记忆和身份信息。

4. **MU 的引入**  
   Machine unlearning 研究如何从已训练模型中删除指定训练样本、用户或域的影响。将 MU 与 EEG-SFDA 结合，可以支持用户撤回授权、删除敏感会话，并在 source-free adaptation 前降低模型泄露。

5. **本文贡献**  
   提出一个 EEG privacy-aware adaptation 框架，将 user/session-level unlearning 与 source-free target adaptation 结合；设计隐私和任务性能联合评估；给出可复现的跨被试 EEG 实验协议。

可写贡献点：

- We formulate privacy-preserving cross-subject EEG decoding as a combined source-free adaptation and user-level unlearning problem.
- We design a three-stage framework that first unlearns sensitive source domains and then adapts to unlabeled target EEG without accessing source data.
- We propose an evaluation protocol that measures both target decoding utility and residual privacy leakage using MIA and subject-identity attacks.
- We provide an experimental plan on public EEG benchmarks to study the utility-privacy trade-off.

## 8. Related Work 结构

建议分四小节：

1. **EEG-based BCI and Cross-Subject Decoding**  
   讲 EEG 解码任务、跨被试泛化困难、非平稳性。

2. **Source-Free Domain Adaptation for EEG**  
   讲 SHOT、SSVEP-SFDA、EEG source-free subject adaptation、AEA、SPDIM、PDCC。

3. **Privacy and Security in BCI**  
   讲 EEG identity leakage、BCI side-channel、privacy-preserving EEG representation、BrainGuard、ID-RemovalNet。

4. **Machine Unlearning**  
   讲 sample-level / class-level / domain-level unlearning，强调 source-free MU 与普通 MU 的区别。

## 9. Method Section 推荐结构

双栏论文可以这样写：

```text
3. Method
3.1 Problem Formulation
3.2 Source EEG Model Training
3.3 Source-Free User/Session Unlearning
3.4 Source-Free Target Adaptation
3.5 Privacy and Utility Evaluation
```

每节要点：

### 3.1 Problem Formulation

定义 \(D_s, D_f, D_t\)，强调 \(D_r\) 不可访问。

### 3.2 Source EEG Model Training

写基本 supervised EEG decoder。不要让这一节太复杂。

### 3.3 Source-Free User/Session Unlearning

写 MU 模块。可以把具体算法写为伪代码：

```text
Algorithm 1: Forget-and-Adapt EEG
Input: source model M_s, forget set D_f, unlabeled target EEG D_t
1: M_u <- SourceFreeUnlearn(M_s, D_f)
2: initialize M_t <- M_u
3: for each target batch x_t:
4:     compute predictions and pseudo labels
5:     update feature encoder with SFDA loss
6: return M_t
```

### 3.4 Source-Free Target Adaptation

写 SHOT-style loss 和 EEG alignment。

### 3.5 Privacy and Utility Evaluation

把 MIA、subject-id attack、forget gap 写成评估协议，而不是方法损失。

## 10. Abstract 草稿

可让下一个 Codex 直接改写：

> Electroencephalography (EEG)-based brain-computer interfaces require subject adaptation because neural signals are highly non-stationary across users and sessions. However, EEG data are privacy-sensitive and may reveal identity, health-related attributes, and cognitive responses. Source-free domain adaptation (SFDA) mitigates raw-data sharing by adapting a source model to unlabeled target EEG without accessing source data, but the released source model itself may still retain private information about source users. In this work, we study privacy-preserving cross-subject EEG decoding through the joint lens of SFDA and machine unlearning. We propose a three-stage framework that trains a source EEG decoder, removes the influence of specified users or sessions from the source model, and then performs source-free adaptation on unlabeled target EEG. We further design an evaluation protocol that measures both target decoding utility and privacy leakage through membership inference and subject-identity attacks. The proposed design provides a practical blueprint for EEG adaptation systems that must support both personalization and user data deletion.

## 11. 写作时必须避免的夸大

不要写：

- “SFDA guarantees privacy.”
- “MU completely removes all EEG privacy risk.”
- “Our method is certified private.”
- “We prove formal unlearning guarantees.”

除非后续真的做理论证明或差分隐私，否则应该写：

- “reduces raw source data exposure”
- “mitigates model-level privacy leakage”
- “empirically reduces membership and subject-identity leakage”
- “supports user/session-level deletion requests”

## 12. 可引用的本地论文清单

SFDA / EEG adaptation：

- `papers/sfda/2020ICML-SHOT Source Hypothesis Transfer for Unsupervised Domain Adaptation.pdf`
- `papers/sfda/2022arXiv-Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces.pdf`
- `papers/sfda/2023arXiv-Source-free Subject Adaptation for EEG-based Visual Recognition.pdf`
- `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf`
- `papers/sfda/2025JBHI-Prediction Consistency and Confidence-Based Proxy Domain Construction for Cross-Subject EEG Classification.pdf`
- `papers/sfda/2025ICLR-SPDIM Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG.pdf`

Machine unlearning：

- `papers/sfda/2025CVPR-Towards Source-Free Machine Unlearning.pdf`
- `papers/sfda/2025ICLR-Selective Unlearning via Representation Erasure Using Domain Adversarial Training.pdf`
- `papers/sfda/2025NeurIPS-Approximate Domain Unlearning for Vision-Language Models.pdf`

Privacy / security：

- `papers/bci_security/2021CSUR-Security in Brain-Computer Interfaces State-of-the-Art Opportunities and Future Challenges.pdf`
- `papers/bci_security/2024arXiv-Protecting Multiple Types of Privacy Simultaneously in EEG-based Brain-Computer Interfaces.pdf`
- `papers/bci_security/2025IJCAI-ID-RemovalNet Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks.pdf`
- `papers/bci_security/2025AAAI-BrainGuard Privacy-Preserving Multisubject Image Reconstructions from Brain Activities.pdf`

BCI real-time / application background：

- `papers/bci_realtime/2024FrontComputNeurosci-Electroencephalogram-Based Adaptive Closed-Loop Brain–Computer Interface in Neurorehabilitation A Review.pdf`
- `papers/bci_realtime/2025arXiv-Toward Practical BCI A Real-Time Wireless Imagined Speech EEG Decoding System.pdf`
- `papers/bci_realtime/2004TBME-BCI2000 A General-Purpose Brain–Computer Interface (BCI) System.pdf`
- `papers/bci_realtime/2019GrazBCI-Timeflux An Open-Source Framework for the Acquisition and Near Real-Time Processing of Signal Streams.pdf`

## 13. 给下一个 Codex 的 LaTeX 起草任务

请在另一个机器上基于这个 brief 起草一篇双栏论文初稿，建议结构：

```text
\title{Towards Privacy-Preserving EEG Adaptation via Source-Free Domain Adaptation and Machine Unlearning}

\begin{abstract}
...
\end{abstract}

\section{Introduction}
\section{Background and Motivation}
\section{Problem Formulation}
\section{Method}
\section{Experimental Design}
\section{Discussion}
\section{Conclusion}
```

写作要求：

- 重点写 introduction、background、method、experimental design。
- 不要生成假的实验结果数值。
- 可以放空表格模板，例如 “to be filled after experiments”。
- 使用双栏会议论文风格，例如 IEEEtran、ACM 或 CVPR-like template，具体模板按目标会议再定。
- 所有引用先用占位 citation key，例如 `\cite{shot2020, ssvep_sfda2022, source_free_unlearning2025}`，后续再补 BibTeX。
- 语气上把本文定位为方法设计/实验协议，不要声称已经完成全面验证。

## 14. 最小实验复现路线

如果后续要补一个 very small proof-of-concept，不要一开始追求完整论文实验。建议最小路线：

1. Dataset：PhysioNet EEGMMI。
2. Task：left/right motor imagery 二分类。
3. Domains：subject 或 session。
4. Model：EEGNet 或简单 CNN。
5. Source Training：多源 subject 有标签训练。
6. Forget：指定一个 source subject 作为 \(D_f\)。
7. MU Baselines：NegGrad、Random Label、source-free unlearning 近似。
8. SFDA：SHOT-style entropy + diversity + pseudo-label。
9. Utility：target accuracy / F1。
10. Privacy：MIA AUC + subject-id attack accuracy。

如果时间不足，只写实验设计表，不跑。

## 15. 论文主张边界

这篇论文最稳妥的主张不是“提出一个已经完全验证的新算法”，而是：

> We argue and design a privacy-aware EEG adaptation framework where SFDA addresses raw-source-data exposure and MU addresses model-level residual privacy leakage.

这样写的优点：

- 与当前已有材料一致。
- 不需要承诺完整实验已经跑完。
- 能自然解释 EEG + SFDA + MU 三者为什么必须结合。
- 后续可以逐步补实验，而不会让初稿过度依赖未完成结果。
