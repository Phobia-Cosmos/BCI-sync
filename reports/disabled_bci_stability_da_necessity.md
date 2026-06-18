# Disabled 个体稳定性视角下：为什么 BCI 必须做 Domain Adaptation

> 目的：为组会/导师汇报提供一条更贴近真实使用者的论证链。核心不是泛泛地说“DA 提升准确率”，而是从 disabled 个体长期使用 BCI 的稳定性需求出发，说明为什么 BCI 从实验室 demo 走向临床、康复和居家辅助设备时，Domain Adaptation 是必要条件。

## 1. 一句话结论

对 disabled 个体而言，BCI 的价值不是某一次离线实验 accuracy 高，而是能否在跨天、跨状态、跨佩戴条件、跨康复阶段中稳定提供通信、控制或康复反馈能力。EEG/BCI 信号天然存在跨用户和跨 session 非平稳性；disabled 用户还叠加疲劳、注意波动、病程变化、药物、康复训练和电极佩戴困难等因素。因此，不做 DA 会直接表现为长时间重复校准、错误通信、机器人/外设误控制、康复反馈不稳定和设备弃用风险。DA 是 disabled-oriented BCI 从一次性实验系统变成长期可用辅助设备的基础。

## 2. 为什么 disabled 个体视角更强调“稳定性”

普通消费级 EEG 场景中，识别失败可能只是体验差；但在 disabled 个体使用 BCI 的场景中，稳定性直接决定系统是否有实际辅助价值。

| 使用场景 | disabled 用户缺失/受损能力 | BCI 的补偿方式 | 稳定性失败的真实后果 |
|---|---|---|---|
| 通信/拼写 | ALS、locked-in syndrome、严重运动障碍、语言输出受限 | EEG/SCP/P300/SSVEP 拼写器，把脑响应转成字符或选择 | 输出错误、沟通效率极低、用户疲劳、照护者误解用户意图 |
| 机器人手/臂控制 | stroke、SCI、四肢瘫、上肢运动障碍 | MI/ME EEG 控制机器人手、机械臂或外部设备 | 命令不稳定、误触发、抓取失败、安全风险、用户失去信任 |
| 康复训练 | stroke、SCI 后运动功能恢复 | EEG-BCI 闭环反馈、虚拟 avatar、外骨骼、FES/机器人辅助训练 | 错误反馈削弱训练闭环，影响康复效率和患者依从性 |
| 居家长期辅助 | 长期运动障碍或慢性病患者 | 低成本、非侵入式 EEG 设备持续使用 | 每天重新长校准不可接受，设备最终无法坚持使用 |

因此，disabled BCI 的关键指标不应只看单次 accuracy，还应看：

- **跨天稳定性**：今天训练的模型明天是否还能用。
- **跨状态稳定性**：疲劳、注意、情绪、药物、病情变化后是否还能用。
- **低校准负担**：是否需要用户反复完成长时间标注任务。
- **连续在线稳定性**：实时输出是否抖动、误触发、漂移。
- **康复阶段适配性**：用户神经状态随训练变化后模型是否能跟上。

## 3. disabled 个体中 domain shift 更严重的原因

BCI 中的 domain 不只是“不同数据集”。在真实设备中，domain 可以是不同用户、不同天次、不同 session、不同电极佩戴、不同设备、不同任务阶段，甚至同一患者康复过程中的不同神经状态。

### 3.1 跨用户差异：每个 disabled 用户的神经和生理状态不同

不同用户的头皮结构、脑区功能保留程度、病灶位置、运动想象能力、注意能力、疲劳水平都不同。对于 stroke/SCI/ALS 等用户，这种差异通常比健康受试者更复杂。通用模型直接迁移到新用户时，模型学到的 source subject 模式未必对应 target subject 的有效任务特征。

本地论文支撑：

- `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf` 指出 EEG-BCI 中 subjects 间 variance 很大，不可能存在适用于所有人的 universal optimal model，因此需要新 subject 的 target data 做适配。
- `papers/sfda/2021AAAI-Plug-and-Play Domain Adaptation for Cross-Subject EEG-based Emotion Recognition.pdf` 指出 EEG 强 subject-dependent，差异来自 mental states、electrode impedance、head shapes 等因素。
- `papers/bci_decoding/2026AAAI-MindCross Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals.pdf` 把 fast new subject adaptation with limited data 明确作为真实 brain decoding 需求。

### 3.2 跨 session 非平稳：同一 disabled 用户也不是静态域

同一用户在不同日期和不同佩戴条件下，EEG 分布会发生变化。disabled 用户更容易出现额外变化源：

- 疲劳、疼痛、药物和注意力波动；
- 电极佩戴位置、阻抗和皮肤状态变化；
- 康复训练导致的神经可塑性变化；
- 病情进展或功能恢复导致的运动意图表征变化；
- 长期使用中用户学习如何控制 BCI，脑响应本身也会改变。

本地论文支撑：

- `papers/sfda/2025ICLR-SPDIM Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG.pdf` 明确指出 EEG 的 non-stationary nature 会在 days 和 subjects 间引入 distribution shifts；没有 labeled calibration data 时，问题就是 source-free unsupervised domain adaptation。
- `reports/nature_noninvasive_bci_real_world_and_da.md` 整理的 Scientific Data 2022 跨 session MI 数据集显示，within-session classification 最高平均 68.8%，cross-session classification 下降到 53.7%，cross-session adaptation 提升到 78.9%。这说明同一受试者跨天不适配时性能会显著退化。
- `papers/bci_decoding/2025Nature-EEG-based BCI Enables Real-Time Robotic Hand Control at Individual Finger Level.pdf` 在实时机器人手指控制中使用 same-day fine-tuning 缓解 inter-session variability，并使用 online smoothing 稳定控制输出。

### 3.3 康复闭环导致 domain 本身持续变化

康复型 BCI 不是静态分类器，而是“用户—模型—反馈—神经可塑性”的闭环系统。用户在训练过程中会学习调节脑活动，神经网络也可能因康复训练发生变化。也就是说，模型要识别的目标分布不是固定的，训练越深入，target domain 反而越可能改变。

本地论文支撑：

- `reports/nature_noninvasive_bci_real_world_and_da.md` 中整理的 Chaudhary et al. 2016 Nature Reviews Neurology 区分 assistive BCI 与 rehabilitative BCI，强调 BCI 可作为通信/控制工具，也可作为康复训练工具。
- Donati et al. 2016 Scientific Reports 报告慢性 SCI 患者经过 12 个月 BMI-based gait neurorehabilitation 后出现感觉和自主肌肉控制改善。这说明康复 BCI 的用户神经状态并非静态。
- `papers/bci_realtime/09_Real_Time_EEG_BCI_Stroke_Rehab_Latent_Features_2025.pdf` 的本地提取内容显示 stroke rehab BCI 中存在 subject-specific variability，需要 high stability across sessions 和 brief guided calibration。

## 4. 不做 DA 对 disabled 用户意味着什么

从 disabled 个体稳定性角度看，不做 DA 的后果不是“模型掉几个点”，而是辅助功能本身变得不可依赖。

### 4.1 通信型 BCI：错误输出和校准负担

SSVEP/P300/SCP 拼写器常面向 speech difficulties、locked-in 或重度瘫痪用户。若模型无法适配新用户或新 session，就会出现两类问题：

1. 需要长时间标注校准。disabled 用户可能疲劳、注意维持困难，长校准会降低舒适性和依从性。
2. 直接迁移会导致字符/选项识别错误。对无法通过其他方式纠正错误的用户，错误输出可能被照护者误解为真实意图。

本地论文支撑：

- `papers/sfda/2022Arxiv-Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces.pdf` 明确指出 SSVEP BCI speller 用于帮助 speech difficulties 用户通信；高 ITR 方法需要 extensive calibration，而典型用户可能是 disabled individual，因此去除校准负担对 user comfort 和 plug-and-play 很重要。
- Birbaumer et al. 1999 Nature 的 EEG spelling device for the paralysed 证明瘫痪患者可用 EEG 拼写，但也暴露出速度低、训练负担大、稳定性有限的问题。

### 4.2 控制型 BCI：机器人/外设误控制

机器人手、机械臂、轮椅、光标等控制型 BCI 需要实时输出稳定。如果跨 session 漂移导致 MI 类别混淆，后果就是命令误触发或输出抖动。

本地论文支撑：

- `papers/bci_decoding/2025Nature-EEG-based BCI Enables Real-Time Robotic Hand Control at Individual Finger Level.pdf` 中 2 指 MI 在线控制平均 accuracy 为 80.56%，3 指 MI 为 60.61%。这说明任务越精细，稳定控制越困难；论文需要 same-day fine-tuning 和 online smoothing 来减轻 inter-session variability 和输出不稳定。
- `reports/nature_noninvasive_bci_real_world_and_da.md` 中整理的 Meng et al. 2016 Scientific Reports 显示非侵入式 EEG 可以控制机器人臂 reach-and-grasp，但仍依赖多 session 训练和任务分解，说明自然、长期、稳定控制仍未解决。

### 4.3 康复型 BCI：错误反馈影响训练闭环

康复 BCI 的价值来自“运动意图—脑电解码—反馈/辅助动作”的闭环。如果模型不适配患者当前状态，就可能把错误脑状态反馈给用户，破坏训练一致性。

本地论文支撑：

- `papers/bci_realtime/09_Real_Time_EEG_BCI_Stroke_Rehab_Latent_Features_2025.pdf` 的本地提取内容显示不同 subject accuracy 从 60% 到 86%，反映 subject-specific variability in EEG signal quality and motor control；论文还提到 session 间结果可能变化，需要 brief guided calibration。
- Donati et al. 2016 Scientific Reports 的 SCI 步态康复研究需要 12 个月闭环训练，说明康复 BCI 是长期动态过程，不适合假设一次训练模型永久有效。

## 5. DA 在 disabled-oriented BCI 中具体解决什么

DA 的作用不是简单追求更高离线分数，而是把模型从 source domain 调整到当前 target user/session，使其在真实设备中保持可用。

| DA 对象 | 对应真实问题 | disabled 稳定性意义 |
|---|---|---|
| Cross-subject DA | 新用户与训练用户不同 | 降低每个 disabled 用户从零采集大量标注数据的负担 |
| Cross-session DA | 同一用户跨天/跨佩戴变化 | 避免每天重新长校准，提高居家可用性 |
| Online / CTTA | 使用过程中 EEG 持续漂移 | 让设备在疲劳、注意变化、康复进展中持续更新 |
| Source-free DA | 医疗/神经数据不能访问源 EEG | 满足隐私和合规限制下的个性化适配 |
| Label/conditional shift adaptation | 用户状态分布改变 | 适应康复阶段、睡眠/情绪/注意状态比例变化 |

可以把 DA 的必要性表述成三层：

1. **可用性层面**：减少校准时间，让 disabled 用户能更快开始使用。
2. **稳定性层面**：缓解跨天、跨状态、跨佩戴造成的性能退化。
3. **安全性层面**：减少错误通信、错误控制和错误康复反馈。

## 6. 为什么 disabled 视角会自然导向 SFDA

传统 DA 往往假设源数据可访问，但 disabled/临床 BCI 中这个假设不现实。

原因包括：

- EEG 属于高敏感神经数据，可能包含身份、年龄、性别、疾病状态、认知和情绪信息。
- 医疗机构或康复中心之间很难共享原始 EEG。
- 用户可能撤回授权，历史数据不能继续访问。
- 居家设备需要本地、轻量、即时适配，不可能每次上传所有 source users 数据。

本地论文支撑：

- `papers/sfda/2025JBHI-Lightweight Source-Free Domain Adaptation based on Adaptive Euclidean Alignment for BCIs.pdf` 明确指出为保护 source subjects 隐私，source data sometimes unavailable during DA，因此 SFDA 只依赖 source-trained model 扩展到 target subjects。
- `papers/sfda/2026AAAI-Probability Distribution Alignment and Low-Rank Weight Decomposition for Source-Free Domain Adaptive Brain Decoding.pdf` 将 cross-subject variations、privacy concerns 和 data storage burden 作为 SFDA 动机。
- `papers/sfda/2022Arxiv-Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces.pdf` 指出 SFDA 不需要保存 source data，可减少隐私与存储问题，同时面向 disabled users 的舒适性和 plug-and-play 需求。

因此，从 disabled 个体稳定性出发，逻辑不是“我们为了算法新颖性做 SFDA”，而是：

> disabled 用户需要长期稳定使用 BCI；长期稳定使用必然面对跨用户/跨 session/domain shift；解决 domain shift 需要 DA；但临床/神经数据不能长期集中访问源 EEG，所以真实部署更接近 SFDA。

## 7. 与 MU/隐私保护的衔接

如果后续论文要做 SFDA + MU，可以这样连接：

1. disabled 用户是敏感群体，BCI 数据具有医疗与身份敏感性。
2. DA 需要利用历史用户知识来帮助新用户适配，但这会引入源用户数据和模型残留隐私问题。
3. SFDA 减少对 source EEG 的访问，但源模型仍可能保留 source users/domains 的影响。
4. 当用户撤回授权、某个医院/设备域不再授权、或某类隐私属性需要移除时，仅删除原始 EEG 不足以消除模型中的残留影响。
5. 因此 MU 可作为 SFDA 后的模型级隐私修正机制：在保持 target adaptation 和主任务性能的同时，尽量忘记撤回用户、撤回 domain 或敏感属性。

这条链条可以写成论文动机：

> Assistive EEG-BCIs require stable personalization for disabled users, but personalization conflicts with neural data privacy. Source-free domain adaptation reduces the need to revisit source EEG, while machine unlearning further addresses the residual influence of revoked users or sensitive source domains in the adapted model.

## 8. 汇报时可以直接使用的中文版本

### 8.1 简短版

我们做 DA 的核心动机不是单纯提升 benchmark accuracy，而是 disabled 用户使用 BCI 时对长期稳定性的刚性需求。对于瘫痪、ALS、stroke 或 SCI 用户，BCI 承担的是通信、外设控制和康复反馈功能。一旦模型跨天、跨状态或跨佩戴条件失效，后果不是普通体验下降，而是错误通信、误控制、康复反馈不稳定和设备弃用。

EEG 信号本身低信噪比、非平稳，并且强烈 subject-dependent。同一个用户不同天的电极位置、阻抗、疲劳、注意、药物和康复阶段都会改变 EEG 分布；不同 disabled 用户的病灶、运动能力和神经可塑性差异更大。因此 BCI 的真实部署天然是 cross-subject、cross-session 和 online domain adaptation 问题。

进一步地，disabled/临床 BCI 涉及高敏感神经数据，不能假设源 EEG 可以长期集中保存和重复访问。因此，真实部署更符合 source-free DA：只使用源模型和目标用户持续产生的未标注 EEG 来做个性化适配。如果还考虑用户撤权或源模型残留隐私，就需要把 machine unlearning 加入 SFDA 框架。

### 8.2 适合放在 PPT 的逻辑链

1. **真实需求**：disabled 用户需要 BCI 弥补通信、控制、康复能力。
2. **关键指标**：辅助设备最重要的是长期稳定，而不是单次离线 accuracy。
3. **核心矛盾**：EEG 跨用户、跨天、跨状态非平稳；disabled 用户变化更复杂。
4. **不做 DA 的后果**：长校准、错误通信、误控制、错误康复反馈、设备弃用。
5. **DA 的必要性**：让模型适配当前用户、当前 session、当前康复状态。
6. **SFDA 的必要性**：源 EEG 是敏感医疗/神经数据，真实部署中经常不可访问。
7. **MU 的必要性**：用户撤权或源域不再授权时，需要去除模型中残留影响。

### 8.3 适合写入 Introduction 的段落

Non-invasive EEG-based BCIs are especially valuable for disabled individuals because they can provide alternative channels for communication, robotic control, and neurorehabilitation when normal neuromuscular pathways are impaired. However, for these users, the central requirement is not only high offline accuracy but also long-term stability. A spelling error, an unstable robotic command, or an incorrect rehabilitation feedback signal can directly undermine communication, safety, and clinical adherence. Unfortunately, EEG signals are highly non-stationary and subject-dependent. Their distributions vary across users, sessions, electrode placements, fatigue levels, attention states, and rehabilitation stages. These variations are even more critical in disabled populations, where neurological impairment, medication, and recovery-induced plasticity introduce additional domain shifts.

Therefore, domain adaptation is a practical necessity for assistive BCI deployment. Cross-subject adaptation reduces the burden of collecting extensive labeled EEG from each new disabled user, while cross-session and online adaptation maintain stability during long-term use. In clinical and home settings, however, source EEG data are often unavailable due to privacy, storage, and consent constraints. This motivates source-free domain adaptation, where a source-trained model is adapted to a target user using only target-side data. Furthermore, when users revoke consent or sensitive source domains must be removed, machine unlearning becomes necessary to eliminate residual source influence from the model while preserving assistive performance.

## 9. 证据与引用

[1] Birbaumer N, Ghanayim N, Hinterberger T, et al. A spelling device for the paralysed[J]. Nature, 1999, 398: 297-298. DOI: 10.1038/18581.

[2] Chaudhary U, Birbaumer N, Ramos-Murguialday A. Brain-computer interfaces for communication and rehabilitation[J]. Nature Reviews Neurology, 2016, 12: 513-525. DOI: 10.1038/nrneurol.2016.113.

[3] Donati A R C, Shokur S, Morya E, et al. Long-term training with a brain-machine interface-based gait protocol induces partial neurological recovery in paraplegic patients[J]. Scientific Reports, 2016, 6: 30383. DOI: 10.1038/srep30383.

[4] Meng J, Zhang S, Bekyo A, et al. Noninvasive electroencephalogram based control of a robotic arm for reach and grasp tasks[J]. Scientific Reports, 2016, 6: 38565. DOI: 10.1038/srep38565.

[5] Ma J, Yang B, Qiu W, et al. A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface[J]. Scientific Data, 2022, 9: 531. DOI: 10.1038/s41597-022-01647-1.

[6] Ding Y, Udompanyawit C, Zhang Y, et al. EEG-based brain-computer interface enables real-time robotic hand control at individual finger level[J]. Nature Communications, 2025, 16: 5401. DOI: 10.1038/s41467-025-61064-x.

[7] Guney O B, Kucukahmetler D, Ozkan H. Source-free domain adaptation for SSVEP-based brain-computer interfaces[J]. arXiv preprint arXiv:2305.17403, 2025.

[8] Zhao L M, Yan X, Lu B L. Plug-and-play domain adaptation for cross-subject EEG-based emotion recognition[C]//AAAI Conference on Artificial Intelligence. 2021.

[9] Li S, Kawanabe M, Kobler R J. SPDIM: Source-free unsupervised conditional and label shift adaptation in EEG[C]//International Conference on Learning Representations. 2025.

[10] Wang H, Han H, Gan J Q, Wang H. Lightweight source-free domain adaptation based on adaptive Euclidean alignment for brain-computer interfaces[J]. IEEE Journal of Biomedical and Health Informatics, 2025. DOI: 10.1109/JBHI.2024.3463737.

[11] Xu G, Long J, Zhang J. Probability distribution alignment and low-rank weight decomposition for source-free domain adaptive brain decoding[C]//AAAI Conference on Artificial Intelligence. 2026.

[12] Real-Time EEG BCI Stroke Rehab Latent Features 2025. Local file: `papers/bci_realtime/09_Real_Time_EEG_BCI_Stroke_Rehab_Latent_Features_2025.pdf`.

## 10. 最适合你当前课题的最终表述

如果老师问“为什么 disabled 个体稳定性会推出 DA + SFDA + MU”，可以按下面回答：

> 因为 disabled 用户使用 BCI 不是娱乐或一次性实验，而是依赖它完成通信、控制或康复训练，所以核心要求是长期稳定。EEG 在跨用户、跨天、跨佩戴、跨疲劳/注意状态下都会产生 domain shift；对 stroke、SCI、ALS 等用户，还会叠加病情、药物、康复进展和神经可塑性变化。不做 DA，系统会出现长时间校准、错误通信、误控制和康复反馈不稳定。做 DA 是为了让模型适配当前用户和当前 session。但 disabled/临床 BCI 的源 EEG 数据高度敏感，无法假设长期可访问，因此需要 SFDA。进一步，如果用户撤回授权或某个源域不再可用，单纯不访问源数据还不够，模型中可能仍有源用户残留影响，所以需要 MU 来做模型级遗忘。我们的研究就是在真实 disabled-oriented BCI 部署约束下，同时解决稳定适配和隐私撤权问题。
