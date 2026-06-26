# 物理刺激扰动攻击 BCI 的相关论文调研

## 核心结论

与 `2023SMC-Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events` 严格同类的论文目前很少。该论文的关键创新不是“给 EEG 信号加扰动”，而是把攻击位置前移到 **用户感官输入层**：攻击者改变用户看到、听到或触摸到的外部刺激，从而改变用户产生的 EEG，再导致 BCI 模型输出失效。

调研后可以把相关工作分成四层：

1. **严格同类**：物理/感官刺激扰动，直接造成 BCI 控制或分类性能下降。目前最直接的是 Upadhayay 和 Behzadan 的 MI 视觉闪烁攻击。
2. **物理但非感官刺激**：在采集端通过电磁/电流等方式注入扰动，属于物理域攻击，但不是改变视觉/听觉/触觉刺激。
3. **非物理的 EEG 信号攻击**：直接修改 EEG 信号、滤波链路或训练数据，可覆盖 P300、SSVEP、ERN、MI 等范式，但攻击点不是外部感官事件。
4. **刺激敏感性证据**：P300、SSVEP、cVEP 等范式高度依赖刺激形状、颜色、频率、相位、布局、语义反馈和注视位置；这些论文通常不是安全攻击论文，但能支撑“物理刺激扰动可以成为攻击面”的研究动机。

因此，对于其他范式可以严谨地写成：

> P300、SSVEP 和 cVEP 等视觉诱发 BCI 天然依赖外部刺激设计。已有大量刺激设计论文证明刺激参数会显著影响 ERP/VEP 特征与分类性能；同时，已有安全论文证明 P300/SSVEP/ERN/MI 解码器可被微小信号扰动、后门触发或滤波扰动攻击。但是，将二者结合起来、系统研究“恶意物理刺激扰动如何攻击 P300/SSVEP/cVEP BCI”的工作仍然不足，是一个明确的研究空白。

## 分类表

| 类别 | 代表论文 | 范式 | 是否物理刺激 | 攻击目标 | 与 2023 SMC 的关系 |
| --- | --- | --- | --- | --- | --- |
| 严格同类 | Upadhayay & Behzadan, *Adversarial Stimuli* | MI | 是，视觉闪烁 | 降低 MI BCI 控制性能 | 目标论文本身，最强直接证据 |
| 物理域但非感官 | Wang et al., *Physically-Constrained Adversarial Attacks on BMIs* | MI | 部分是，物理信号注入 | EEGNet MI 分类失效 | 攻击在采集物理层，不是视觉/听觉/触觉刺激 |
| 数字/信号扰动 | Zhang et al., *Tiny Noise, Big Mistakes* | P300、SSVEP | 否 | 拼写器输出任意攻击者字符 | 证明 P300/SSVEP 模型可被微小扰动操控 |
| 数字/信号扰动 | Zhang & Wu, *On the Vulnerability of CNN Classifiers in EEG-Based BCIs* | P300、ERN、MI | 否 | CNN 分类器性能下降 | 证明多范式 EEG CNN 脆弱 |
| 数字/信号扰动 | Liu et al., *Universal Adversarial Perturbations for CNN Classifiers in EEG-Based BCIs* | P300、MI | 否 | 通用扰动实时攻击 | 提高攻击实用性，但仍非物理刺激 |
| 后门/采集扰动 | Meng et al., *EEG-Based BCIs Are Vulnerable to Backdoor Attacks* | ERN、MI、P300 | 否，NPP 加到 EEG | 后门触发错误类别 | 覆盖 ERN/P300/MI，说明实时采集链路风险 |
| 滤波链路攻击 | Meng et al., *Adversarial Filtering Based Evasion and Backdoor Attacks* | ERN、MI、P300 | 否 | 滤波后分类降至机会水平或后门触发 | 覆盖信号处理模块，但不是感官刺激 |
| 隐私刺激攻击 | Martinovic et al., USENIX Security 2012 | ERP/P300 相关 | 是，视觉探测刺激 | 推断 PIN、银行、区域等隐私 | 物理刺激攻击成立，但目标是隐私泄漏，不是模型失效 |
| 隐蔽刺激攻击 | Frank et al., *Subliminal Probing* | ERP/P300 相关 | 是，阈下视觉刺激 | 推断用户是否识别敏感人物/信息 | 物理视觉刺激攻击成立，但目标是隐私泄漏 |
| 刺激敏感性 | Fernández-Rodríguez et al., P300 speller stimuli | P300 | 是，形状/颜色/闪烁方式 | 改变 P300 分类和拼写性能 | 可转化为 P300 物理攻击假设 |
| 刺激敏感性 | Cecotti & Rivet, XP300 | P300 | 是，布局/伪随机顺序/视觉反馈 | 改变 P300 检测率 | 支撑 GUI/刺激语义会影响 P300 |
| 刺激敏感性 | Frenzel & Neubert, *Is the P300 Speller Independent?* | P300 | 是，注视/高亮刺激 | 注视方向影响 EEG 响应 | 支撑 P300 不完全是“纯意图”，视觉处理可被利用 |
| 刺激敏感性 | Atabek et al., high-frequency motion stimulus | SSVEP/SSMVEP | 是，频率/运动/光栅刺激 | 改变 VEP 响应、疲劳和分类表现 | 支撑 SSVEP 对刺激频率和形式敏感 |
| 刺激敏感性 | Miao et al., high-performance cVEP-BCI | cVEP/SSVEP | 是，白噪声码/频率相位编码 | 目标识别依赖刺激编码 | 支撑 cVEP 的攻击面在 stimulus code |

## 严格同类论文

### 1. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events

- **引用**：Upadhayay B, Behzadan V. *Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events*. arXiv:2211.10033, 2022; IEEE SMC 2023.
- **链接**：https://arxiv.org/abs/2211.10033
- **本地文件**：`papers/phyAttack/2023SMC-Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events.pdf`
- **范式**：Motor imagery, MI。
- **攻击方式**：用户通过 MI 控制 Pong 游戏；攻击者让游戏中的 paddle/ball 以 20 Hz 闪烁。
- **关键点**：
  - 论文定义 adversarial stimuli 为攻击者引入的感官事件扰动，可是视觉、听觉或触觉扰动。
  - 实验中使用视觉闪烁，不直接修改 EEG。
  - 结果显示 MI BCI 性能显著下降。
  - 论文还指出类似思路可能影响 SSVEP 和 c-VEP，例如改变视觉闪烁频率，但这部分是未来工作，不是该论文已验证结果。

这是目前最贴合用户问题的论文：**物理视觉刺激 → 用户脑电响应变化 → MI BCI 模型/系统性能下降**。

## 物理域攻击但不是感官刺激攻击

### 2. Physically-Constrained Adversarial Attacks on Brain-Machine Interfaces

- **引用**：Wang X, Hersche M, Siller O R Q, Benini L, Singh G. *Physically-Constrained Adversarial Attacks on Brain-Machine Interfaces*. TSRML workshop co-located with NeurIPS, 2022.
- **本地文件**：`papers/phyAttack/2022NeurIPS-Physically-Constrained Adversarial Attacks on Brain-Machine Interfaces.pdf`
- **范式**：MI。
- **攻击方式**：考虑真实头皮传播、幅值、延迟等物理约束；扰动可以通过环境电磁波或头皮电流刺激等方式进入采集端。
- **为什么相关**：
  - 它证明“物理约束”对 BCI 攻击很关键。
  - 攻击不是在模型输入后端随便加噪，而是考虑信号如何经过头皮传播到电极。
- **为什么不算严格同类**：
  - 它不是改变用户看到/听到/触摸到的任务刺激。
  - 更接近“采集端物理注入攻击”，而不是“感官事件扰动攻击”。

这篇可作为 2023 SMC 的相邻证据：BCI 攻击需要考虑真实物理通道，但它不能替代“视觉/听觉/触觉刺激攻击”的证据。

## P300 / SSVEP / ERN 的已验证攻击论文

这些论文覆盖其他 BCI 范式，但攻击点主要是 EEG 信号、模型或信号处理链路，不是外部物理刺激。

### 3. Tiny Noise, Big Mistakes: Adversarial Perturbations Induce Errors in Brain-Computer Interface Spellers

- **引用**：Zhang X, Wu D, Ding L, Luo H, Lin C T, Jung T P, Chavarriaga R. *Tiny noise, big mistakes: adversarial perturbations induce errors in brain–computer interface spellers*. National Science Review, 2021, 8(4): nwaa233.
- **链接**：https://academic.oup.com/nsr/article/8/4/nwaa233/5903729
- **本地文件**：`papers/phyAttack/2020-Tiny noise, big mistakes: adversarial perturbations induce errors in brain-computer interface spellers.pdf`
- **范式**：P300 speller、SSVEP speller。
- **攻击方式**：给 EEG trial 添加很小的 adversarial perturbation template，让拼写器输出攻击者指定字符。
- **价值**：
  - 这是 P300/SSVEP 拼写器攻击的核心论文。
  - 论文明确指出 P300 和 SSVEP 拼写器都可能被微小扰动严重操控。
- **限制**：
  - 扰动直接加到 EEG 信号上，不是改变外部视觉刺激。

### 4. On the Vulnerability of CNN Classifiers in EEG-Based BCIs

- **引用**：Zhang X, Wu D. *On the Vulnerability of CNN Classifiers in EEG-Based BCIs*. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 27(5): 814-825.
- **链接**：https://arxiv.org/abs/1904.01002
- **范式**：P300 evoked potential detection、feedback ERN detection、MI classification。
- **攻击方式**：在信号预处理和机器学习模块之间插入 jamming module，生成 adversarial EEG examples。
- **价值**：
  - 覆盖 P300、ERN、MI 三类任务。
  - 可用于说明“不同 BCI 范式中的机器学习模型都存在对抗脆弱性”。
- **限制**：
  - 不是物理刺激扰动。

### 5. Universal Adversarial Perturbations for CNN Classifiers in EEG-Based BCIs

- **引用**：Liu Z, Meng L, Zhang X, Fang W, Wu D. *Universal Adversarial Perturbations for CNN Classifiers in EEG-Based BCIs*. Journal of Neural Engineering, 2021, 18(4): 0460a4.
- **链接**：https://arxiv.org/abs/1912.01171
- **范式**：P300、MI。
- **攻击方式**：构造一次性通用扰动 UAP，可在实时 EEG trial 开始后直接加入。
- **价值**：
  - 相比逐 trial 计算扰动，更接近真实实时攻击。
  - 可支撑“实时 BCI 安全风险不只是离线理论问题”。
- **限制**：
  - 仍然是信号域扰动，不是感官刺激扰动。

### 6. EEG-Based Brain-Computer Interfaces Are Vulnerable to Backdoor Attacks

- **引用**：Meng L, Huang J, Zeng Z, Jiang X, Yu S, Jung T P, Lin C T, Chavarriaga R, Wu D. *EEG-Based Brain-Computer Interfaces Are Vulnerable to Backdoor Attacks*. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2023, 31: 2224-2234.
- **链接**：https://arxiv.org/abs/2011.00101
- **本地文件**：`papers/phyAttack/2021Arxiv-EEG-Based Brain–Computer Interfaces are Vulnerable to Backdoor Attacks.pdf`
- **范式**：ERN、MI、P300。
- **攻击方式**：把 narrow period pulse 作为 backdoor key，加到 EEG 采集信号中。
- **价值**：
  - 覆盖 ERN、MI、P300。
  - 攻击触发不需要精确同步 EEG trial，更贴近真实场景。
- **限制**：
  - backdoor key 加到 EEG 信号，不是通过视觉/听觉/触觉刺激诱发。

### 7. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces

- **引用**：Meng L, Jiang X, Chen X, Liu W, Luo H, Wu D. *Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces*. Information Fusion, 2024, 107: 102316.
- **链接**：https://arxiv.org/abs/2412.07231
- **本地文件**：`papers/phyAttack/2024IF-Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces.pdf`
- **范式**：ERN、MI、P300。
- **攻击方式**：设计 adversarial filter，让 EEG 经过滤波后分类性能下降，或作为后门 key。
- **价值**：
  - 说明不仅模型输入和训练数据有风险，BCI 信号处理模块本身也可以成为攻击面。
- **限制**：
  - 不是物理刺激扰动。

## 物理视觉刺激导致隐私泄漏的论文

这两篇不是让 BCI 控制模型失效，而是证明攻击者可以通过设计视觉刺激，诱发 EEG/ERP 变化并推断用户隐私。它们非常适合支撑“感官刺激本身可以成为攻击通道”。

### 8. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces

- **引用**：Martinovic I, Davies D, Frank M, Perito D, Ros T, Song D. *On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces*. USENIX Security, 2012: 143-158.
- **链接**：https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic
- **范式/信号**：EEG/ERP，尤其与 P300 风格的识别响应相关。
- **攻击方式**：恶意应用向用户展示与银行、PIN、居住地、熟人等相关的视觉刺激，分析 EEG 响应推断隐私。
- **价值**：
  - 物理/视觉刺激是攻击的一部分。
  - 证明攻击者不必直接控制 BCI 模型，也能通过刺激设计利用脑电反应。
- **限制**：
  - 目标是隐私泄漏，不是让 P300/SSVEP/MI 控制模型失效。

### 9. Using EEG-Based BCI Devices to Subliminally Probe for Private Information

- **引用**：Frank M, Hwu T, Jain S, Knight R T, Martinovic I, Mittal P, Perito D, Sluganovic I, Song D. *Using EEG-Based BCI Devices to Subliminally Probe for Private Information*. arXiv:1312.6052, 2017.
- **链接**：https://arxiv.org/abs/1312.6052
- **范式/信号**：阈下视觉刺激诱发的 EEG/ERP 响应。
- **攻击方式**：在视频中短暂插入视觉刺激，用户未必意识到刺激存在，但攻击者仍可从 EEG 中推断用户是否识别某人。
- **价值**：
  - 更接近 2023 SMC 的“感官事件扰动”思想。
  - 可作为物理刺激攻击的隐私版证据。
- **限制**：
  - 不是控制模型完整性攻击。

## P300 范式：刺激参数敏感性证据

P300 是典型的 ERP/oddball 范式，用户是否看到目标刺激、目标刺激如何闪烁、是否与注视位置重合、刺激语义是否强，都会影响 EEG 和分类效果。因此，P300 很适合进一步研究物理刺激攻击。

### 10. Evaluation of Flashing Stimuli Shape and Colour Heterogeneity Using a P300 BCI Speller

- **引用**：Fernández-Rodríguez Á, Velasco-Álvarez F, Medina-Juliá M T, Ron-Angevin R. *Evaluation of flashing stimuli shape and colour heterogeneity using a P300 brain-computer interface speller*. arXiv:1812.00836, 2018.
- **链接**：https://arxiv.org/abs/1812.00836
- **范式**：P300 speller。
- **刺激变量**：闪烁刺激的形状和颜色，包括 white letters、white blocks、coloured letters、coloured blocks。
- **结论价值**：
  - block-shaped illumination 在 accuracy、correct commands per minute 和 P300 waveform 上优于传统 letter-shaped flashing。
  - 说明仅改变界面刺激外观，就会改变 P300 BCI 的检测性能。
- **攻击启发**：
  - 攻击者可以不改 EEG，而是改变高亮形状、闪烁区域、颜色对比和显示模式，使训练时的刺激分布与部署时不同，从而降低 P300 分类性能。

### 11. Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection

- **引用**：Cecotti H, Rivet B. *Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection*. arXiv:1612.03640, 2016.
- **链接**：https://arxiv.org/abs/1612.03640
- **范式**：P300 speller。
- **刺激变量**：不使用传统行列高亮，而使用伪随机视觉刺激顺序，并给每个 item 添加视觉反馈以增强 stimulus meaning。
- **结论价值**：
  - XP300 在平均识别率和 single-trial AUC 上高于经典 CP300。
  - 论文明确指出 P300 speller 的 GUI 设计不足会成为理想 P300 检测的障碍。
- **攻击启发**：
  - 如果“增强语义/反馈”能提升 P300，那么恶意或混淆性的语义反馈也可能削弱或偏移 P300 检测。

### 12. Is the P300 Speller Independent?

- **引用**：Frenzel S, Neubert E. *Is the P300 Speller Independent?* arXiv:1006.3688, 2010.
- **链接**：https://arxiv.org/abs/1006.3688
- **范式**：P300 speller。
- **刺激变量**：注视字符和注意字符分离。
- **结论价值**：
  - 结果显示 gaze direction 在 P300 speller 中起重要作用。
  - 高亮字符可诱发与视觉处理相关的响应，不完全等同于用户意图。
- **攻击启发**：
  - P300 BCI 不只是读取“想选哪个字符”，还依赖视觉注意与注视位置。
  - 恶意视觉干扰、诱导注视偏移或相邻刺激混淆可能影响输出。

## SSVEP / SSMVEP 范式：频率、相位、疲劳和刺激形式敏感性证据

SSVEP 的核心机制是外部周期性视觉刺激诱发相同频率及谐波的 EEG 响应。因此，如果攻击者能改变 flickering frequency、phase、contrast、refresh timing 或空间布局，理论上更容易造成 stimulus-response mismatch。

### 13. A Grating Based High-Frequency Motion Stimulus Paradigm for SSMVEP

- **引用**：Atabek B, Yılmaz E, Acarturk C, Çakır M P. *A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials*. arXiv:2312.15682, 2024.
- **链接**：https://arxiv.org/abs/2312.15682
- **范式**：SSVEP/SSMVEP。
- **刺激变量**：传统 flicker、motion stimulus、高频光栅 motion stimulus。
- **结论价值**：
  - SSVEP 由重复视觉刺激诱发，响应频率与外部 flicker frequency 对应。
  - 低频刺激通常有更强 SSVEP amplitude，但高频刺激可能降低不适和安全风险。
  - 刺激类型会影响信号区分度、疲劳与分类表现。
- **攻击启发**：
  - 对 SSVEP，直接改变目标 flicker frequency、phase 或把额外闪烁叠加到界面上，可能让分类器匹配到错误目标或降低 SNR。
  - 这与 2023 SMC 论文中“改变 flickering frequency 可能影响 SSVEP/c-VEP”的推测一致，但仍需要专门实验验证。

### 14. High-Performance cVEP-BCI Under Minimal Calibration

- **引用**：Miao Y, Shi N, Huang C, Song Y, Chen X, Wang Y, Gao X. *High-performance cVEP-BCI under minimal calibration*. arXiv:2311.11596, 2023.
- **链接**：https://arxiv.org/abs/2311.11596
- **范式**：cVEP、SSVEP。
- **刺激变量**：white-noise stimulus sequences、joint frequency-phase modulation。
- **结论价值**：
  - cVEP 依赖 stimulus sequence 与 EEG temporal pattern 的匹配。
  - 论文强调 broadband stimulus 下 spatial-temporal pattern 更复杂，需要校准或迁移学习。
- **攻击启发**：
  - 对 cVEP，攻击面不是简单频率，而是 stimulus code/sequence。
  - 如果攻击者改变白噪声编码序列、码间距离或时序同步，可能破坏模板匹配与目标识别。

## 可用于论文背景的严谨表述

### 表述 1：严格同类工作稀缺

目前 BCI 安全文献主要研究 EEG 信号层、模型层和训练数据层攻击，例如 adversarial perturbation、universal perturbation、backdoor 和 adversarial filtering。真正把攻击点放在用户感官输入层、通过物理刺激扰动改变脑电响应并造成 BCI 输出失效的研究仍然很少。Upadhayay 和 Behzadan 的 adversarial stimuli 研究是该方向的重要早期证据。

### 表述 2：P300/SSVEP/cVEP 更适合扩展物理刺激攻击

P300、SSVEP 和 cVEP 比 MI 更直接依赖外部刺激。P300 依赖 oddball 视觉高亮事件，SSVEP 依赖周期性 flicker frequency/phase，cVEP 依赖刺激编码序列。因此，攻击者如果能操纵 GUI 高亮方式、颜色/形状、频率、相位、刷新时序或编码序列，就可能在不接触 EEG 信号链路的情况下影响用户脑电响应和模型输出。

### 表述 3：已有证据链不是“攻击已完成”，而是“攻击面成立”

已有 P300/SSVEP/cVEP 刺激设计研究证明刺激参数会显著影响 EEG 特征和分类性能；已有 BCI 安全研究证明 P300/SSVEP/ERN/MI 模型可被微小扰动、后门或滤波攻击操控。二者结合说明物理刺激扰动是合理且值得研究的攻击面，但目前其他范式下的系统性 adversarial stimuli 实验仍然不足。

## 后续如果要做论文，可以这样定位

题目方向可以写成：

> Physical Adversarial Stimuli for Visual-Evoked Brain-Computer Interfaces

或更具体：

> Attacking P300 and SSVEP Brain-Computer Interfaces via Perturbed Visual Stimuli

研究问题可以设置为：

1. **P300**：改变高亮形状、颜色、亮度、顺序、相邻刺激布局，是否导致目标/非目标 ERP 可分性下降？
2. **SSVEP**：在目标频率附近加入微小 frequency/phase shift 或额外闪烁，是否导致 CCA/FBCCA/深度模型识别错误？
3. **cVEP**：扰动 stimulus code、sequence alignment 或刷新时序，是否破坏模板匹配？
4. **跨模型泛化**：同一物理刺激扰动是否能攻击传统方法和深度模型？
5. **人因安全**：扰动是否会增加视觉疲劳、注意分散、误操作或用户不适？

## 推荐检索关键词

- `"adversarial stimuli" "brain-computer interface"`
- `"perturbed sensory events" BCI`
- `"P300 speller" stimulus design performance`
- `"P300 speller" adversarial perturbation`
- `"SSVEP" adversarial perturbation BCI speller`
- `"SSVEP" stimulus frequency phase performance`
- `"cVEP" stimulus sequence brain-computer interface`
- `"subliminal probing" EEG BCI private information`
- `"side-channel attacks" brain-computer interfaces`
- `"adversarial filtering" EEG BCI`
- `"backdoor attacks" EEG BCI`

## 参考文献

1. Upadhayay B, Behzadan V. Adversarial Stimuli: Attacking Brain-Computer Interfaces via Perturbed Sensory Events[EB/OL]. arXiv:2211.10033, 2022. https://arxiv.org/abs/2211.10033.
2. Wang X, Hersche M, Siller O R Q, Benini L, Singh G. Physically-Constrained Adversarial Attacks on Brain-Machine Interfaces[C]. TSRML workshop co-located with NeurIPS, 2022.
3. Zhang X, Wu D, Ding L, Luo H, Lin C T, Jung T P, Chavarriaga R. Tiny noise, big mistakes: adversarial perturbations induce errors in brain-computer interface spellers[J]. National Science Review, 2021, 8(4): nwaa233. https://academic.oup.com/nsr/article/8/4/nwaa233/5903729.
4. Zhang X, Wu D. On the vulnerability of CNN classifiers in EEG-based BCIs[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 27(5): 814-825. https://arxiv.org/abs/1904.01002.
5. Liu Z, Meng L, Zhang X, Fang W, Wu D. Universal adversarial perturbations for CNN classifiers in EEG-based BCIs[J]. Journal of Neural Engineering, 2021, 18(4): 0460a4. https://arxiv.org/abs/1912.01171.
6. Meng L, Huang J, Zeng Z, Jiang X, Yu S, Jung T P, Lin C T, Chavarriaga R, Wu D. EEG-Based Brain-Computer Interfaces Are Vulnerable to Backdoor Attacks[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2023, 31: 2224-2234. https://arxiv.org/abs/2011.00101.
7. Meng L, Jiang X, Chen X, Liu W, Luo H, Wu D. Adversarial Filtering Based Evasion and Backdoor Attacks to EEG-Based Brain-Computer Interfaces[J]. Information Fusion, 2024, 107: 102316. https://arxiv.org/abs/2412.07231.
8. Martinovic I, Davies D, Frank M, Perito D, Ros T, Song D. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces[C]//21st USENIX Security Symposium. 2012: 143-158. https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic.
9. Frank M, Hwu T, Jain S, Knight R T, Martinovic I, Mittal P, Perito D, Sluganovic I, Song D. Using EEG-Based BCI Devices to Subliminally Probe for Private Information[EB/OL]. arXiv:1312.6052, 2017. https://arxiv.org/abs/1312.6052.
10. Fernández-Rodríguez Á, Velasco-Álvarez F, Medina-Juliá M T, Ron-Angevin R. Evaluation of flashing stimuli shape and colour heterogeneity using a P300 brain-computer interface speller[EB/OL]. arXiv:1812.00836, 2018. https://arxiv.org/abs/1812.00836.
11. Cecotti H, Rivet B. Toward Improving the Visual Stimulus Meaning for Increasing the P300 Detection[EB/OL]. arXiv:1612.03640, 2016. https://arxiv.org/abs/1612.03640.
12. Frenzel S, Neubert E. Is the P300 Speller Independent?[EB/OL]. arXiv:1006.3688, 2010. https://arxiv.org/abs/1006.3688.
13. Atabek B, Yılmaz E, Acarturk C, Çakır M P. A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials[EB/OL]. arXiv:2312.15682, 2024. https://arxiv.org/abs/2312.15682.
14. Miao Y, Shi N, Huang C, Song Y, Chen X, Wang Y, Gao X. High-performance cVEP-BCI under minimal calibration[EB/OL]. arXiv:2311.11596, 2023. https://arxiv.org/abs/2311.11596.
