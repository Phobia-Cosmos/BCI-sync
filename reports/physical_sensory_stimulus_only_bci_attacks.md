# 严格物理/感官刺激视角下的 BCI 攻击论文调研

## 问题边界

这里采用更严格的攻击假设：攻击者 **不能** 修改 EEG 原始信号、不能接触 BCI 设备、不能控制采集链路、不能改模型、不能投毒训练集，只能影响用户实际接收到的外部刺激或使用环境，例如：

- 屏幕上的闪烁、颜色、亮度、形状、布局、刷新频率、动画和视觉干扰；
- 恶意网页、游戏界面、拼写器界面或第三方 App 中的刺激呈现；
- 环境灯光闪烁、LED、投影、AR/VR 显示；
- 音频提示、错误反馈、触觉提示等外部感官事件。

在这个边界下，很多 EEG adversarial perturbation、backdoor、adversarial filtering 论文 **不应作为主证据**，因为它们默认攻击者能把扰动加到 EEG 信号、滤波链路或训练数据中，这在真实非侵入式 BCI 使用中通常不现实。

## 总体结论

严格按照上述边界筛选，目前真正直接验证“物理/感官刺激扰动导致 BCI 模型或系统失效”的论文非常少。最直接的是：

- Upadhayay 和 Behzadan 的 *Adversarial Stimuli*：通过改变用户看到的视觉事件，让 MI BCI 控制性能下降。

其他范式如 P300、SSVEP、cVEP、ErrP 目前更多是两类间接证据：

1. **物理刺激隐私攻击已经成立**：攻击者可以通过展示视觉刺激诱发 EEG/ERP 响应，推断 PIN、银行、居住地、熟人等敏感信息。
2. **刺激设计显著影响 BCI 性能**：P300、SSVEP、cVEP、ErrP 本来就是 stimulus-locked 或 feedback-locked 范式，已有大量研究证明刺激形状、颜色、频率、相位、编码序列、注视位置和反馈方式会影响 EEG 特征与分类性能。

因此，更严谨的论文定位应该是：

> 在非侵入式 BCI 中，攻击者难以直接修改 EEG 信号，但可以通过恶意界面或环境操纵用户感官输入。现有研究已经证明感官刺激可造成隐私泄漏，也证明刺激参数会显著影响 P300/SSVEP/cVEP/ErrP 等范式的 EEG 表征；然而，将这些刺激敏感性系统化为“物理 adversarial stimuli 完整性攻击”的研究仍不足。

## 严格符合条件的直接攻击论文

### 1. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events

- **引用**：Upadhayay B, Behzadan V. *Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events*. arXiv:2211.10033, 2022; IEEE SMC 2023.
- **链接**：https://arxiv.org/abs/2211.10033
- **本地文件**：`papers/phyAttack/2023SMC-Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events.pdf`
- **范式**：Motor imagery, MI。
- **攻击者能力**：不接触 EEG 设备和模型，只改变用户看到的游戏视觉事件。
- **攻击方式**：用户使用 MI BCI 控制 Pong 游戏；攻击时游戏中的 paddle/ball 以 20 Hz 闪烁。
- **结果意义**：视觉刺激扰动显著降低 MI BCI 任务表现。
- **为什么重要**：MI 不是典型视觉诱发范式，仍然会被外部视觉刺激干扰；这说明感官通道可以作为 BCI 完整性攻击入口。

这篇是目前最贴合“现实攻击者无法接触设备，只能影响外部刺激”的完整性攻击论文。

## 物理刺激造成隐私泄漏的论文

这类工作不是让 BCI 控制模型失效，但它们严格符合“攻击者通过外部刺激影响用户 EEG”的现实边界。因此，它们可用于证明：**感官刺激本身就是可被攻击者利用的输入通道**。

### 2. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces

- **引用**：Martinovic I, Davies D, Frank M, Perito D, Ros T, Song D. *On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces*. USENIX Security, 2012.
- **链接**：https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic
- **攻击者能力**：恶意应用向用户展示特定视觉刺激，同时读取消费级 BCI 的 EEG 响应。
- **攻击目标**：推断用户是否熟悉 PIN 数字、银行、居住区域、月份等敏感信息。
- **可支撑观点**：攻击者无需修改 EEG 信号，只要控制用户看到的内容，就能利用用户脑响应造成隐私泄漏。
- **与完整性攻击的关系**：目标是 privacy，不是 control failure；但攻击入口与 adversarial stimuli 一样，都是外部视觉刺激。

### 3. Using EEG-Based BCI Devices to Subliminally Probe for Private Information

- **引用**：Frank M, Hwu T, Jain S, Knight R T, Martinovic I, Mittal P, Perito D, Sluganovic I, Song D. *Using EEG-Based BCI Devices to Subliminally Probe for Private Information*. arXiv:1312.6052, 2017.
- **链接**：https://arxiv.org/abs/1312.6052
- **攻击者能力**：在视频流中短暂插入用户未必显式意识到的视觉刺激。
- **攻击目标**：根据 EEG 响应推断用户是否识别特定人物等私密信息。
- **可支撑观点**：即使用户没有主动输入，也可能因为外部刺激导致 EEG 泄漏敏感信息。
- **与完整性攻击的关系**：这说明“刺激不可见/不显眼”不等于安全；未来对 P300/SSVEP/cVEP 做物理刺激攻击时，可以借鉴这种隐蔽刺激模型。

## P300：最适合做物理界面攻击的范式之一

P300 BCI 本身依赖 oddball 视觉事件：目标项和非目标项通过闪烁/高亮呈现，分类器再判断哪些刺激诱发 P300。因此攻击者如果能控制显示界面，就可以改变高亮方式、颜色、形状、时序、语义反馈、相邻刺激布局或注视诱导。

### 4. Evaluation of Flashing Stimuli Shape and Colour Heterogeneity Using a P300 BCI Speller

- **引用**：Fernández-Rodríguez Á, Velasco-Álvarez F, Medina-Juliá M T, Ron-Angevin R. *Evaluation of flashing stimuli shape and colour heterogeneity using a P300 brain-computer interface speller*. arXiv:1812.00836, 2018.
- **链接**：https://arxiv.org/abs/1812.00836
- **物理刺激变量**：高亮刺激的形状和颜色，包括 letters、blocks、white、coloured 等组合。
- **结论价值**：仅改变屏幕上的刺激外观，就会影响 P300 waveform、accuracy 和 correct commands per minute。
- **攻击启发**：恶意界面可通过改变高亮形状、颜色对比、刺激区域或闪烁方式，使用户 EEG 分布偏离训练时刺激条件。

### 5. Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection

- **引用**：Cecotti H, Rivet B. *Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection*. arXiv:1612.03640, 2016.
- **链接**：https://arxiv.org/abs/1612.03640
- **物理刺激变量**：视觉反馈、刺激语义、刺激呈现顺序和 GUI 设计。
- **结论价值**：增强 stimulus meaning 和视觉反馈可提高 P300 检测。
- **攻击启发**：如果语义增强能提升 P300，恶意语义干扰、错误反馈或混淆性 UI 也可能削弱或偏移 P300 响应。

### 6. Is the P300 Speller Independent?

- **引用**：Frenzel S, Neubert E. *Is the P300 Speller Independent?* arXiv:1006.3688, 2010.
- **链接**：https://arxiv.org/abs/1006.3688
- **物理刺激变量**：注视字符和注意字符分离；字符高亮刺激。
- **结论价值**：gaze direction 在 P300 speller 中起重要作用，视觉处理成分会影响 EEG 响应。
- **攻击启发**：P300 speller 并非只读取“意图”，还受到视觉注视和刺激呈现影响；诱导用户看错位置、制造邻近闪烁或注视干扰，可能造成输出错误。

## SSVEP / SSMVEP：物理频率攻击最自然的候选范式

SSVEP BCI 依赖周期性视觉刺激。用户盯着某个频率闪烁的目标，枕叶 EEG 中出现对应频率及其谐波。分类器通常根据频谱或 CCA/FBCCA 匹配目标频率。因此，攻击者若能改变外部 flicker frequency、phase、contrast、refresh timing 或叠加额外闪烁，就可能直接改变用户产生的 SSVEP。

### 7. A Grating Based High-Frequency Motion Stimulus Paradigm for SSMVEP

- **引用**：Atabek B, Yılmaz E, Acarturk C, Çakır M P. *A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials*. arXiv:2312.15682, 2024.
- **链接**：https://arxiv.org/abs/2312.15682
- **物理刺激变量**：传统 flicker、motion stimulus、高频光栅 motion stimulus。
- **结论价值**：视觉刺激频率和刺激形式影响 VEP 响应、用户疲劳与分类表现。
- **攻击启发**：对 SSVEP，攻击可以直接设计成频率/相位错配，例如把目标 12 Hz 轻微偏移、插入邻近频率闪烁、改变刷新时序，导致分类器匹配错误或 SNR 降低。

### 8. A Survey of Stimulation Methods Used in SSVEP-Based BCIs

- **引用**：Zhu D, Bieger J, Garcia Molina G, Aarts R M. *A Survey of Stimulation Methods Used in SSVEP-Based BCIs*. Computational Intelligence and Neuroscience, 2010: 702357.
- **链接**：https://doi.org/10.1155/2010/702357
- **物理刺激变量**：SSVEP 频率、刺激设备、刺激模式、编码方式、视觉舒适度等。
- **结论价值**：SSVEP BCI 的核心问题之一就是如何设计刺激方法；这说明刺激不是外部无关变量，而是系统输入的一部分。
- **攻击启发**：攻击者不必碰 EEG，只要能改变刺激源，就能改变用户大脑产生的目标频率响应。

## cVEP：攻击面在 stimulus code / sequence

cVEP 与 SSVEP 类似，都是 visual evoked potential，但 cVEP 更依赖编码序列。用户看到的目标通常以特定 pseudo-random code 或 white-noise sequence 调制，分类器用模板匹配或时序相关性识别目标。

### 9. High-Performance cVEP-BCI Under Minimal Calibration

- **引用**：Miao Y, Shi N, Huang C, Song Y, Chen X, Wang Y, Gao X. *High-performance cVEP-BCI under minimal calibration*. arXiv:2311.11596, 2023.
- **链接**：https://arxiv.org/abs/2311.11596
- **物理刺激变量**：white-noise stimulus sequences、joint frequency-phase modulation。
- **结论价值**：cVEP 识别依赖刺激序列与 EEG temporal pattern 的匹配。
- **攻击启发**：对 cVEP，攻击不一定是频率扰动，而是改变 stimulus code、sequence alignment、码间相关性或刷新同步，使模板匹配失效。

## ErrP / ERN：物理反馈攻击的候选方向

ErrP/ERN 类 BCI 用用户看到错误反馈或意识到错误时产生的脑电响应进行纠错或自适应。这里攻击者能影响的是 **反馈事件本身**，例如故意显示错误反馈、延迟反馈、随机反馈或不一致反馈。

### 10. Towards the Classification of Error-Related Potentials using Riemannian Geometry

- **引用**：Tang Y, Zhang J J, Corballis P M, Hallum L E. *Towards the Classification of Error-Related Potentials using Riemannian Geometry*. arXiv:2109.13085, 2021.
- **链接**：https://arxiv.org/abs/2109.13085
- **物理刺激变量**：实验任务中的视觉判别和 audio feedback。
- **结论价值**：ErrP 可由用户识别错误/反馈事件诱发，并可被分类。
- **攻击启发**：如果 BCI 用 ErrP 做在线纠错，恶意反馈事件可能诱发错误的 ErrP 分布，使系统误以为用户观察到错误或没有观察到错误。

### 11. Error-related Potential Variability: Exploring the Effects on Classification and Transferability

- **引用**：Poole B, Lee M. *Error-related Potential Variability: Exploring the Effects on Classification and Transferability*. arXiv:2301.06555, 2023.
- **链接**：https://arxiv.org/abs/2301.06555
- **物理/任务变量**：awareness、embodiment、predictability、observational/interactive task setting。
- **结论价值**：ErrP 分类和迁移会受到认知状态、任务设置和反馈上下文影响。
- **攻击启发**：攻击者可以不改 EEG，而是改变反馈上下文和任务可预测性，诱导 ErrP 分布变化。

## 明确排除：不作为主证据的论文类型

以下论文仍然有安全价值，但不符合本报告的严格现实攻击边界：

| 类型 | 为什么排除为主证据 |
| --- | --- |
| EEG adversarial perturbation | 默认攻击者能把扰动加到 EEG trial 上，真实中往往需要接触设备、传输链路或软件。 |
| EEG backdoor / narrow period pulse | 默认攻击者能投毒训练数据或在测试时给 EEG 加触发器。 |
| Adversarial filtering | 默认攻击者能控制信号处理模块或滤波链路。 |
| 采集端电磁/电流注入 | 比直接改 EEG 更物理，但仍是攻击设备/采集通道，不是改变用户接收的自然感官刺激。 |

这些工作可以用于说明“BCI 模型本身很脆弱”，但不能用来回答“攻击者不能碰设备时还能不能攻击”的核心问题。

## 对不同范式的现实攻击可行性排序

| 范式 | 现实物理刺激攻击可行性 | 原因 |
| --- | --- | --- |
| SSVEP | 高 | 分类目标直接由外部 flicker frequency/phase 决定；攻击者只要影响屏幕/灯光/刷新时序就可能改变响应。 |
| cVEP | 高 | 模型依赖 stimulus sequence/code；恶意修改编码或时序可破坏模板匹配。 |
| P300 | 中高 | 依赖 oddball 高亮、语义、注视和视觉注意；恶意 UI/刺激布局可影响 ERP。 |
| ErrP/ERN | 中 | 依赖错误反馈和任务上下文；攻击者可通过错误/延迟/随机反馈影响 ErrP。 |
| MI | 中 | MI 不由外部刺激直接编码，但视觉干扰、注意分散、诱发 SSVEP/mu rhythm 变化仍可影响 MI 控制。2023 SMC 已给出直接证据。 |
| NS/自然刺激 | 中 | 复杂自然刺激本来会引发多模态 EEG 变化，但攻击目标和可重复性需要重新定义。 |

## 可以直接写进论文的研究空白

> Existing BCI security studies mainly assume that attackers can tamper with EEG signals, preprocessing pipelines, or training data. Such assumptions are often unrealistic for deployed non-invasive BCI systems, where an external attacker may have no access to the EEG hardware or decoding pipeline. In contrast, sensory stimuli are often exposed through graphical interfaces, games, spellers, lights, audio cues, or feedback channels. While prior work has shown that visual stimuli can leak private information and that P300/SSVEP/cVEP/ErrP systems are highly stimulus-dependent, systematic studies on physical sensory-stimulus attacks against BCI integrity remain scarce.

中文表述：

> 现有 BCI 安全研究大量关注直接修改 EEG 信号、滤波链路或训练数据，但这类攻击假设在真实非侵入式 BCI 部署中并不总是成立。更现实的外部攻击者往往只能影响用户看到、听到或触摸到的刺激事件。已有研究证明视觉刺激可诱发隐私泄漏，也证明 P300、SSVEP、cVEP、ErrP 等范式对刺激参数高度敏感；然而，面向 BCI 完整性的物理感官刺激攻击仍缺乏系统研究。

## 建议后续论文方向

更贴合你当前问题的题目可以是：

- *Physical Sensory-Stimulus Attacks against Non-Invasive Brain-Computer Interfaces*
- *Attacking Visual-Evoked BCI Systems through Malicious Stimulus Manipulation*
- *When the Screen Becomes the Attacker: Physical Stimulus Attacks on EEG-Based BCIs*

实验设计可优先从 SSVEP 或 P300 入手，因为攻击者只需要控制屏幕刺激：

1. **SSVEP**：训练正常频率界面，测试时轻微改变目标频率、相位、刷新同步或叠加干扰频率。
2. **P300**：训练正常高亮界面，测试时改变高亮颜色、形状、亮度、相邻刺激布局或错误反馈。
3. **cVEP**：训练标准 code sequence，测试时改变 code alignment、sequence correlation 或刷新时序。
4. **MI**：复现 2023 SMC 的视觉闪烁干扰，并测试不同闪烁频率是否诱发 SSVEP/mu rhythm coupling。

## 参考文献

1. Upadhayay B, Behzadan V. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events[EB/OL]. arXiv:2211.10033, 2022. https://arxiv.org/abs/2211.10033.
2. Martinovic I, Davies D, Frank M, Perito D, Ros T, Song D. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces[C]//21st USENIX Security Symposium. 2012: 143-158. https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic.
3. Frank M, Hwu T, Jain S, Knight R T, Martinovic I, Mittal P, Perito D, Sluganovic I, Song D. Using EEG-Based BCI Devices to Subliminally Probe for Private Information[EB/OL]. arXiv:1312.6052, 2017. https://arxiv.org/abs/1312.6052.
4. Fernández-Rodríguez Á, Velasco-Álvarez F, Medina-Juliá M T, Ron-Angevin R. Evaluation of flashing stimuli shape and colour heterogeneity using a P300 brain-computer interface speller[EB/OL]. arXiv:1812.00836, 2018. https://arxiv.org/abs/1812.00836.
5. Cecotti H, Rivet B. Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection[EB/OL]. arXiv:1612.03640, 2016. https://arxiv.org/abs/1612.03640.
6. Frenzel S, Neubert E. Is the P300 Speller Independent?[EB/OL]. arXiv:1006.3688, 2010. https://arxiv.org/abs/1006.3688.
7. Atabek B, Yılmaz E, Acarturk C, Çakır M P. A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials[EB/OL]. arXiv:2312.15682, 2023. https://arxiv.org/abs/2312.15682.
8. Zhu D, Bieger J, Garcia Molina G, Aarts R M. A Survey of Stimulation Methods Used in SSVEP-Based BCIs[J]. Computational Intelligence and Neuroscience, 2010: 702357. https://doi.org/10.1155/2010/702357.
9. Miao Y, Shi N, Huang C, Song Y, Chen X, Wang Y, Gao X. High-performance cVEP-BCI under minimal calibration[EB/OL]. arXiv:2311.11596, 2023. https://arxiv.org/abs/2311.11596.
10. Tang Y, Zhang J J, Corballis P M, Hallum L E. Towards the Classification of Error-Related Potentials using Riemannian Geometry[EB/OL]. arXiv:2109.13085, 2021. https://arxiv.org/abs/2109.13085.
11. Poole B, Lee M. Error-related Potential Variability: Exploring the Effects on Classification and Transferability[EB/OL]. arXiv:2301.06555, 2023. https://arxiv.org/abs/2301.06555.
