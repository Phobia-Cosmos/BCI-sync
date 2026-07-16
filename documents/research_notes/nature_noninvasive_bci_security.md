# Nature 及子刊中非侵入式 BCI 安全问题：论文清单与问题边界

> 范围：优先聚焦 Nature / Nature Portfolio 文章中与非侵入式 BCI、EEG/fMRI 神经解码、neurodata privacy、neuroethics、robust EEG-BCI、临床/真实使用安全相关的证据。结论要点是：Nature 上直接以“BCI cybersecurity”为题的非侵入式 EEG 技术论文很少，但 Nature 系列已经从 neurodata privacy、mental privacy、identity/agency、模型鲁棒性、临床安全和设备失败后果等角度明确讨论了 BCI 安全问题。

## 1. 先区分：BCI security 不是单一问题

在 BCI 领域，“安全问题”至少包含四类，不应混用：

| 类型 | 中文含义 | 典型风险 | 与非侵入式 BCI 的关系 |
|---|---|---|---|
| Privacy / neurodata privacy | 神经数据隐私 | EEG/fMRI 可泄露身份、认知状态、语言语义、健康状态、年龄/性别等 | 最直接；非侵入式设备低门槛、可规模化采集，更容易进入消费/居家场景 |
| Cybersecurity / adversarial security | 网络与模型攻击安全 | 数据流篡改、对抗扰动、模型投毒、后门、错误指令 | 非侵入式实时 EEG-BCI 一旦控制外设/康复机器人，错误分类可转化为物理风险 |
| Safety / clinical safety | 医学与使用安全 | 错误控制、康复误导、闭环反馈诱发不良训练、设备失效后患者被抛弃 | Nature 临床/神经工程文章更常采用 safety/ethics 而非 cybersecurity 表述 |
| Autonomy / identity / agency | 自主性、身份与责任 | 用户是否仍是行为主体、BCI 输出归责、长期使用改变自我认同 | Nature neuroethics 文章重点强调，尤其与 assistive BCI 和 neurotechnology 商业化相关 |

对于我们的“EEG + SFDA + MU + 隐私保护”论文，最相关的是 **privacy / neurodata privacy** 和 **model-level security**：源模型可能记忆用户 EEG 特征；跨用户/跨 session 适配需要更多目标数据；撤回授权后需要从模型中删除指定用户/域影响。

## 2. Nature / Nature Portfolio 相关论文清单

### 2.1 核心 neurosecurity / neurodata privacy / ethics 文章

| 编号 | 论文 | 期刊 | 与非侵入式 BCI 安全的关系 |
|---|---|---|---|
| N1 | Yuste et al., 2017, *Four ethical priorities for neurotechnologies and AI* | Nature | 提出 neurotechnology + AI 的四个伦理优先事项：privacy/consent, identity/agency, enhancement/fairness, bias；是 BCI 隐私和身份安全的顶层动机 |
| N2 | Drew, 2019, *The ethics of brain–computer interfaces* | Nature | Nature Outlook 文章，讨论 BCI 在瘫痪/ALS 等医疗应用中的伦理问题；强调 BCI 越接近临床/商业化，隐私、身份、自主性、长期责任越重要 |
| N3 | Yuste, 2023, *Advocating for neurodata privacy and neurotechnology regulation* | Nature Protocols | 明确主张 neurodata privacy 和 neurotechnology regulation；可作为“神经数据需要特殊隐私保护”的 Nature 子刊支撑 |
| N4 | Eaton and Illes, 2007, *Commercializing cognitive neurotechnology—the ethical terrain* | Nature Biotechnology | 早期讨论认知神经技术商业化中的伦理与监管问题；适合支撑消费级/非侵入式神经设备的风险背景 |
| N5 | Clausen, 2009, *Man, machine and in between* | Nature | 讨论神经接口下的人机边界、责任与身份问题；可作为 BCI agency/identity 风险的补充 |

### 2.2 非侵入式神经解码与隐私泄露风险

| 编号 | 论文 | 期刊 | 为什么和安全有关 |
|---|---|---|---|
| N6 | Tang et al., 2023, *Semantic reconstruction of continuous language from non-invasive brain recordings* | Nature Neuroscience | 虽然不是传统 EEG-BCI，而是非侵入式 fMRI 语言解码，但它直接证明非侵入式脑信号可重建连续语义；是 mental privacy / cognitive privacy 的强证据 |
| N7 | Kosnoff et al., 2024, *Transcranial focused ultrasound to V5 enhances human visual motion brain-computer interface by modulating feature-based attention* | Nature Communications | 说明非侵入式 BCI 性能可以通过神经调制增强；安全含义是闭环/调制型 BCI 不只是读取，还可能影响注意与神经状态 |
| N8 | Ding et al., 2025, *EEG-based brain-computer interface enables real-time robotic hand control at individual finger level* | Nature Communications | 非侵入式 EEG 实时控制机器人手指；安全含义是错误分类、模型不稳、跨 session shift 将影响物理外设控制 |
| N9 | Meng et al., 2016, *Noninvasive Electroencephalogram Based Control of a Robotic Arm for Reach and Grasp Tasks* | Scientific Reports | 非侵入式 EEG 控制机器人臂 reach-and-grasp；一旦进入真实患者场景，模型鲁棒性和错误动作就是 safety/security 问题 |
| N10 | Donati et al., 2016, *Long-Term Training with a Brain-Machine Interface-Based Gait Protocol Induces Partial Neurological Recovery in Paraplegic Patients* | Scientific Reports | 非侵入式 EEG + 多组件步态康复；安全重点是长期闭环训练、设备依赖、临床有效性和不良反馈风险 |

### 2.3 非侵入式 EEG-BCI 鲁棒性 / 对抗安全

| 编号 | 论文 | 期刊 | 安全含义 |
|---|---|---|---|
| N11 | Samuel et al., 2026, *Adversarial robust EEG-based brain–computer interfaces using a hierarchical convolutional neural network* | Scientific Reports | Nature Portfolio 中少见的直接把 adversarial robustness 与 EEG-BCI 结合的技术论文；说明 EEG-BCI 模型可能受到对抗扰动影响，鲁棒性是安全问题 |
| N12 | Ma et al., 2022, *A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface* | Scientific Data | 不是攻击论文，但证明跨 session shift 会导致性能显著下降；真实安全角度看，不稳定模型会导致错误控制和错误反馈 |
| N13 | Dreyer et al., 2023, *A large EEG database with users’ profile information for motor imagery brain-computer interface research* | Scientific Data | 数据集含用户画像信息，适合支撑“EEG 与用户身份/人口统计/认知特质绑定，存在隐私属性推断风险” |
| N14 | Yang et al., 2025, *A multi-day and high-quality EEG dataset for motor imagery brain-computer interface* | Scientific Data | 多天 EEG 数据强调跨天不稳定；对 SFDA/CTTA/隐私保护部署有直接价值 |

### 2.4 综述和临床安全背景

| 编号 | 论文 | 期刊 | 安全含义 |
|---|---|---|---|
| N15 | Chaudhary et al., 2016, *Brain–computer interfaces for communication and rehabilitation* | Nature Reviews Neurology | 综述 BCI 在通信和康复中的临床价值与限制；可用于说明安全不是单纯网络攻击，还包括可靠性、患者选择、训练负担和临床可用性 |
| N16 | Sitaram et al., 2017, *Closed-loop brain training: the science of neurofeedback* | Nature Reviews Neuroscience | 闭环脑训练可改变神经活动；安全含义是反馈系统设计不当可能影响神经调节方向，涉及 clinical safety 和 agency |
| N17 | Drew, 2022, *Abandoned: the human cost of neurotechnology failure* | Nature | 不是非侵入式 EEG-BCI 技术论文，但强调神经技术失败/公司退出可能对植入或依赖设备患者造成真实伤害；可作为 neurotechnology lifecycle safety 论据 |

## 3. BCI 安全问题主要讲什么？

### 3.1 神经数据隐私：EEG/fMRI 不只是普通生理数据

BCI 数据的敏感性来自两点：

1. **任务相关意图**：MI、P300、SSVEP、语言解码等信号本身可能表示用户当前意图、注意对象或选择。
2. **任务无关个人属性**：同一段 EEG 还可能包含身份、年龄、性别、疲劳、疾病、情绪、认知能力、药物状态等背景信息。

Nature 相关支撑：

- Yuste et al. 2017 将 privacy/consent 作为 neurotechnology + AI 的首要伦理问题之一。
- Yuste 2023 明确提出 neurodata privacy 和 neurotechnology regulation。
- Tang et al. 2023 说明非侵入式脑记录也可能重建语言语义，强化 mental privacy 风险。
- Dreyer et al. 2023 的 MI-BCI EEG 数据库包含 user profile information，说明 EEG-BCI 研究天然会接触用户画像。

对我们课题的落点：SFDA 只减少源 EEG 数据暴露，不自动保证源模型不泄露源用户信息；MU 可以针对 subject/session/domain 做后训练删除或表征擦除。

### 3.2 模型鲁棒性：错误分类会变成错误控制

非侵入式 EEG-BCI 常用于机器人臂、机器人手、拼写器、康复训练。此时模型输出不再只是分类结果，而是外设动作、通信内容或康复反馈。

风险包括：

- EEG 噪声、电极漂移、疲劳导致分布偏移；
- 对抗扰动或恶意信号注入导致错误命令；
- 模型过度依赖个体身份/设备特征，换用户或换 session 后失效；
- 实时闭环系统把错误反馈反复呈现给用户，影响学习和康复。

Nature 相关支撑：

- Ding et al. 2025 的机器人手指控制明确需要 fine-tuning 缓解 inter-session variability。
- Ma et al. 2022 显示 cross-session classification 显著低于 within-session，而 adaptation 可恢复性能。
- Samuel et al. 2026 直接讨论 adversarial robust EEG-based BCI。

### 3.3 临床安全：可靠性、长期依赖和设备生命周期

BCI 面向残疾人时，安全问题不只是“数据泄露”，还包括：

- 错误输出可能造成物理风险，例如机器人臂、手指、轮椅、外骨骼误动作；
- 康复型 BCI 的错误反馈可能降低训练效果或诱发错误可塑性；
- 患者长期依赖某个神经技术后，设备停服、维护失败、公司退出会造成医疗和心理后果；
- 真实临床使用需要监管、责任划分和长期随访。

Nature 相关支撑：

- Chaudhary et al. 2016 总结 BCI 在通信/康复中的临床限制和患者适用性问题。
- Sitaram et al. 2017 说明闭环神经反馈会塑造神经活动，因此反馈系统安全性重要。
- Drew 2022 讨论 neurotechnology failure 的人类代价。

### 3.4 自主性、身份与责任：谁在控制？谁负责？

当 BCI 输出被用于通信或外部设备控制时，系统错误可能被误认为用户真实意图。尤其在 ALS、locked-in syndrome、stroke、SCI 等人群中，BCI 可能成为用户与外界交互的主要通道。

风险包括：

- 拼写器输出错误被误解为患者真实意愿；
- 模型根据历史数据或群体模式“补全”用户意图，削弱 agency；
- 长期使用 BCI/神经调制后，用户自我认同和责任归属发生变化；
- 商业设备可能把用户神经行为转化为可分析、可交易数据。

Nature 相关支撑：

- Yuste et al. 2017 强调 identity and agency。
- Drew 2019 专门讨论 BCI ethics。
- Eaton and Illes 2007 讨论 cognitive neurotechnology 商业化伦理地形。

## 4. 为什么要先聚焦非侵入式 BCI？

非侵入式 BCI 的安全问题和侵入式不同：

| 维度 | 非侵入式 EEG/fNIRS/fMRI BCI | 侵入式 BCI |
|---|---|---|
| 主要优势 | 低风险、低成本、可扩展、适合家庭/消费场景 | 信号质量高、控制精细、通信吞吐高 |
| 主要安全风险 | 大规模数据采集、用户画像泄露、消费级监管不足、模型漂移导致错误输出 | 手术风险、植入设备网络安全、硬件维护、长期生物相容性 |
| 与 SFDA/MU 的关系 | 多用户、多设备、多 session 部署更需要隐私保护适配 | 数据更稀缺且更敏感，也需要隐私，但源域规模通常较小 |
| 论文切入 | EEG 源模型可能泄露 subject/session/domain；SFDA + MU 正好处理源数据不可访问和撤回问题 | 更偏医疗器械安全、植入设备维护和高风险临床监管 |

所以如果论文目标是 EEG + SFDA + MU，非侵入式 BCI 是更自然的切入：它有更强的数据规模化、跨域适配和用户隐私问题。

## 5. 可直接写进论文的安全动机段落

### 中文版本

BCI 安全问题并不局限于传统网络攻击。对于非侵入式 EEG-BCI，安全首先表现为神经数据隐私：脑电信号既包含任务相关意图，也可能携带身份、人口统计属性、认知状态和疾病相关信息。Nature 关于 neurotechnology and AI 的伦理讨论已将 mental privacy、identity 和 agency 作为核心风险；Nature Protocols 也明确呼吁 neurodata privacy 与 neurotechnology regulation。与此同时，非侵入式 BCI 正从实验室分类任务进入机器人手、机器人臂、拼写器和康复训练等真实闭环场景，模型错误、跨 session 漂移或对抗扰动可能直接导致错误控制、错误通信或错误反馈。因此，隐私保护和鲁棒适配应被视为实际 BCI 部署的基础安全要求。

### English version

Security in non-invasive BCIs is broader than conventional cybersecurity. EEG-based BCIs may expose task-related intentions as well as user-specific attributes such as identity, demographics, cognitive states, and health-related information. Nature Portfolio discussions on neurotechnology ethics have identified mental privacy, identity, agency, and regulation as central concerns. Meanwhile, non-invasive BCIs are increasingly used in closed-loop scenarios such as robotic hand control, robotic arm control, spelling interfaces, and neurorehabilitation, where model errors, session shifts, or adversarial perturbations can translate into incorrect actions, communication errors, or unsafe feedback. These risks motivate privacy-preserving and robust adaptation mechanisms for real-world EEG-BCI deployment.

## 6. 对 SFDA + MU 论文的具体启发

1. **安全目标不要只写 privacy**：还应写 robust adaptation 和 clinical safety。EEG 模型跨 session 掉点本身就是安全隐患。
2. **SFDA 的安全价值有限但重要**：它减少 raw source EEG 暴露，但 source model 仍可能携带源用户信息。
3. **MU 的合理对象是 subject/session/domain/attribute**：不是忘记 MI 的“左手类”或“右手类”，而是忘记某个用户、某次 session、某个医院/设备域或某类敏感属性。
4. **评估指标要同时报告 utility 和 leakage**：target accuracy、cross-session adaptation gain、identity inference、membership inference、attribute inference。
5. **结论不要夸大为 formal privacy guarantee**：除非实现 DP 或 certified unlearning，否则应写 empirical privacy leakage mitigation。

## 7. 参考文献（GB/T 7714 风格草稿）

[1] Yuste R, Goering S, Agüera y Arcas B, et al. Four ethical priorities for neurotechnologies and AI[J]. Nature, 2017, 551: 159-163. DOI: 10.1038/551159a.

[2] Drew L. The ethics of brain-computer interfaces[J]. Nature, 2019, 571: S19-S21. DOI: 10.1038/d41586-019-02214-2.

[3] Yuste R. Advocating for neurodata privacy and neurotechnology regulation[J]. Nature Protocols, 2023, 18: 2869-2875. DOI: 10.1038/s41596-023-00873-0.

[4] Eaton M L, Illes J. Commercializing cognitive neurotechnology—the ethical terrain[J]. Nature Biotechnology, 2007, 25: 393-397. DOI: 10.1038/nbt0407-393.

[5] Tang J, LeBel A, Jain S, et al. Semantic reconstruction of continuous language from non-invasive brain recordings[J]. Nature Neuroscience, 2023, 26: 858-866. DOI: 10.1038/s41593-023-01304-9.

[6] Samuel J, Murugan T K, Govindaraj L, et al. Adversarial robust EEG-based brain-computer interfaces using a hierarchical convolutional neural network[J]. Scientific Reports, 2026, 16: 4353. DOI: 10.1038/s41598-025-34024-0.

[7] Ding Y, Udompanyawit C, Zhang Y, et al. EEG-based brain-computer interface enables real-time robotic hand control at individual finger level[J]. Nature Communications, 2025, 16: 5401. DOI: 10.1038/s41467-025-61064-x.

[8] Meng J, Zhang S, Bekyo A, et al. Noninvasive Electroencephalogram Based Control of a Robotic Arm for Reach and Grasp Tasks[J]. Scientific Reports, 2016, 6: 38565. DOI: 10.1038/srep38565.

[9] Donati A R C, Shokur S, Morya E, et al. Long-Term Training with a Brain-Machine Interface-Based Gait Protocol Induces Partial Neurological Recovery in Paraplegic Patients[J]. Scientific Reports, 2016, 6: 30383. DOI: 10.1038/srep30383.

[10] Ma J, Yang B, Qiu W, et al. A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface[J]. Scientific Data, 2022, 9: 531. DOI: 10.1038/s41597-022-01647-1.

[11] Dreyer P, Roc A, Pillette L, et al. A large EEG database with users’ profile information for motor imagery brain-computer interface research[J]. Scientific Data, 2023, 10: 580. DOI: 10.1038/s41597-023-02445-z.

[12] Yang B, Rong F, Xie Y, et al. A multi-day and high-quality EEG dataset for motor imagery brain-computer interface[J]. Scientific Data, 2025, 12: 488. DOI: 10.1038/s41597-025-04826-y.

[13] Kosnoff J, Yu K, Liu C, et al. Transcranial focused ultrasound to V5 enhances human visual motion brain-computer interface by modulating feature-based attention[J]. Nature Communications, 2024, 15: 4382. DOI: 10.1038/s41467-024-48576-8.

[14] Chaudhary U, Birbaumer N, Ramos-Murguialday A. Brain-computer interfaces for communication and rehabilitation[J]. Nature Reviews Neurology, 2016, 12: 513-525. DOI: 10.1038/nrneurol.2016.113.

[15] Sitaram R, Ros T, Stoeckel L, et al. Closed-loop brain training: the science of neurofeedback[J]. Nature Reviews Neuroscience, 2017, 18: 86-100. DOI: 10.1038/nrn.2016.164.
