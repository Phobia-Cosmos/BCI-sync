# VEP-BCI 中 SSVEP、SSMVEP、c-VEP/c-MVEP 的 DL 解码论文与 Speller 原理

检索日期：2026-06-29  
范围：非侵入式 EEG 视觉诱发电位 BCI，重点覆盖 SSVEP、SSMVEP、c-VEP、c-MVEP，以及 EEGNet / CNN / DNN / Transformer 等深度学习解码方法。

## 1. 核心结论

1. **SSVEP 是目前最成熟、DL 论文最多的 VEP-BCI 范式。**  
   已有从 EEGNet/Compact-CNN、专用 DNN、Transformer、Inception-Transformer，到 SFDA / domain alignment 的一系列工作。原因是 SSVEP 的标签定义清晰：每个字符/目标对应一个频率或频率-相位组合，EEG 中会出现对应频率及谐波。

2. **EEGNet 原文不是 SSVEP 专用模型，但它是 BCI-DL 的通用基础网络。**  
   EEGNet 原文主要验证 P300、ERN、MRCP、SMR 等范式，提出 depthwise/separable convolution 的紧凑 EEG 网络结构。后续很多 SSVEP 或视觉 BCI 工作把 EEGNet 作为 baseline 或结构灵感来源。

3. **SSMVEP 和 c-MVEP 不是“一种任务类型”，而是“视觉刺激编码方式”。**  
   它们可以服务于 speller、菜单选择、轮椅/机器人命令、辅助通信等任务。当前看起来任务少，是因为 motion-based VEP 的多目标刺激、舒适性、SNR、跨被试稳定性和实时分类还没有 SSVEP/c-VEP 成熟。

4. **SSMVEP/c-MVEP 当前主流还不是 EEGNet/DL，而是 CCA、template matching CCA 等传统方法。**  
   2024 年 grating SSMVEP 工作主要做 SNR、CCA/FBCCA 与可感知/不可感知刺激比较；2026 年 c-MVEP 工作用 template-matching CCA 做 4-class online BCI。DL 可作为后续研究方向，但现阶段不能说已经是主流。

5. **传统 SSVEP Speller 的本质是“频率/相位标签化的注意选择”。**  
   屏幕上每个字符以不同频率或频率-相位闪烁；用户注视目标字符后，枕叶 EEG 出现目标频率及谐波；分类器寻找 EEG 与各个候选频率/模板的最大相关性，输出对应字符。

## 2. 关键术语区分

| 范式 | 刺激编码方式 | EEG 响应特征 | 常见解码 | 任务含义 |
|---|---|---|---|---|
| SSVEP | 固定频率/频率-相位闪烁 | 目标频率及谐波明显，主要枕区 | CCA、FBCCA、TRCA、DNN、CNN、Transformer | 可做 speller、多目标选择、控制命令 |
| SSMVEP | 固定频率运动刺激，如缩放、运动光栅 | 类似 steady-state，但通常比 SSVEP 弱、分布更广 | CCA、FBCCA、模板法为主 | 不是任务类型，是低闪烁/低疲劳刺激编码 |
| c-VEP | pseudo-random / m-sequence 码调制闪烁 | 宽带、码序列锁相响应 | 模板匹配、reconvolution CCA、Siamese/CNN 等 | 可做高 ITR speller 或多目标选择 |
| c-MVEP | pseudo-random 码调制运动 | 结合 c-VEP 的 code modulation 与 SSMVEP 的 motion stimulation | 当前代表作使用 template-matching CCA | 新兴 motion-code 范式，不限于一个任务 |

## 3. SSVEP + EEGNet / DL 代表论文

| 年份 | 论文 | 模型/方法 | 与本问题的关系 |
|---|---|---|---|
| 2018 | Lawhern et al., **EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces** | EEGNet：depthwise + separable convolution | BCI-DL 基础网络；不是 SSVEP 专用，但常作为 EEG baseline。原文验证 P300、ERN、MRCP、SMR。 |
| 2018 | Waytowich et al., **Compact Convolutional Neural Networks for Classification of Asynchronous Steady-state Visual Evoked Potentials** | Compact-CNN | 明确用于 SSVEP；强调在缺少刺激先验/异步 SSVEP 中，CNN 可直接从 raw EEG 学特征。 |
| 2022 | Guney et al., **A Deep Neural Network for SSVEP-based Brain-Computer Interfaces** | SSVEP 专用 DNN | 面向 SSVEP speller；使用 benchmark + BETA 两个 40-character 数据集，按 harmonic sub-band/channel/time 做卷积。 |
| 2022 | Chen et al., **A Transformer-based deep neural network model for SSVEP classification** | SSVEPformer / FB-SSVEPformer | 将 Transformer 用于 SSVEP；面向跨被试 inter-subject 场景，尝试减少校准依赖。 |
| 2022 | Luo et al., **A Hybrid Brain-Computer Interface Using Motor Imagery and SSVEP Based on Convolutional Neural Network** | Two-stream CNN | 不是纯 SSVEP，而是 MI+SSVEP hybrid BCI；说明 SSVEP 可与其他范式组合成更稳定的控制系统。 |
| 2023/2025 | Guney et al., **Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces** | 预训练 DNN + source-free adaptation | 直接说明 SSVEP speller 实际应用中存在校准负担；仅用 unlabeled target data 适配新用户。 |
| 2023 | Chen et al., **SSVEP-DAN: A Data Alignment Network for SSVEP-based Brain Computer Interfaces** | Data Alignment Network | 对 session/subject/device domain shift 做神经网络对齐，服务于少校准 SSVEP。 |
| 2025 | Huang et al., **IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification** | Inception + Transformer + filter bank | 新近 SSVEP DL 模型，结合多尺度时序卷积、注意力和滤波器组。 |

### 3.1 对选题的含义

- 如果你要做 **物理视觉刺激攻击 + DL 模型失效分析**，SSVEP 是最稳妥范式：公开数据多、DL baseline 多、speller 应用清晰。
- 如果你要做 **舒适性/低闪烁/不可感知刺激攻击**，SSMVEP 或 c-MVEP 更有新意，但需要自己补足基线模型和任务设计。
- 如果你要做 **domain adaptation / test-time adaptation**，SSVEP 更容易论证，因为已有 SSVEP-DAN、SFDA-SSVEP-BCI 等直接证据说明跨被试、跨 session、跨设备 shift 是核心问题。

## 4. SSMVEP、c-VEP、c-MVEP 的 DL 状态

### 4.0 先给结论：非 SSVEP 范式的 DNN 文献很少

| 范式 | 是否已有明确 DNN 解码论文 | 可确认的 DNN/非 DNN 解码器 | 当前判断 |
|---|---|---|---|
| SSMVEP | 很少，未检索到成熟的 EEGNet/DNN 主线论文 | 代表工作主要用 CCA、FBCCA、template matching | 当前主流不是 DNN，仍以传统相关性/模板解码为主 |
| c-VEP | 有，但数量明显少于 SSVEP | CNN、Siamese network、CCA+BLDA、Corr+BLDA | c-VEP 是三者中最接近可做 DNN baseline 的方向 |
| c-MVEP | 目前未看到成熟 DNN 解码论文 | 2026 代表作使用 template-matching CCA | 新范式，DL 解码基本仍是空白点 |

因此，如果论文需要“EEGNet 等 DL 网络”作为主线，**SSVEP 最合适，c-VEP 次之；SSMVEP/c-MVEP 更适合作为新刺激范式或低闪烁/运动刺激扩展，而不是现成 DNN 解码文献主线。**

### 4.1 SSMVEP

代表论文：

- Atabek et al., **A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials**.

要点：

- 该工作提出 imperceptible high-frequency motion grating，用于降低传统 VEP 闪烁带来的不适。
- 实验包含 SSVEP、SSMVEP、imperceptible grating SSMVEP 三类刺激。
- 解码/分析主要基于 SNR、CCA、FBCCA 等传统方法。
- 该工作明确受限于不可感知 motion 刺激的多目标呈现，因此没有完成该条件下完整多目标分类准确率分析。

结论：

- SSMVEP 不是只能表示一个任务，而是当前很多研究还处于刺激范式验证阶段。
- 可感知 SSMVEP 可以做多频目标选择；不可感知或高频 grating SSMVEP 的多目标设计仍是开放问题。
- 从攻击实验看，SSMVEP 的价值在于研究“非闪烁运动刺激是否也会被外部物理刺激干扰”，而不是仅复现 SSVEP。

### 4.2 c-VEP

代表论文：

- Nair and Cecotti, **Deep Learning Architectures for Code-Modulated Visual Evoked Potentials Detection**.

要点：

- c-VEP 使用 code-modulated flicker，通常每个目标用 pseudo-random sequence 或 m-sequence 的不同 circular shift 编码。
- 传统解码多用模板匹配、CCA/reconvolution CCA。
- 2025 年预印本开始系统比较 CNN、Siamese network 与 CCA baseline，说明 c-VEP 的 DL 解码正在出现，但整体数量仍少于 SSVEP。

结论：

- c-VEP 也不是一种任务，而是一种码调制刺激方式。
- c-VEP 适合 speller，因为多个字符可共享同一 m-sequence 的不同移位，从而实现多目标编码。

### 4.3 c-MVEP

代表论文：

- Scheppink et al., **Beyond Flickering: Introducing Code-Modulated Motion Visual Evoked Potentials for Brain-Computer Interfacing**.

要点：

- c-MVEP 用 pseudo-random sequence 调制运动刺激，相当于把 c-VEP 的 code modulation 从 flicker 换成 motion。
- 该文比较 c-MVEP、c-VEP、SSMVEP、SSVEP。
- 在线实验实现了 4-class BCI：c-MVEP 平均准确率 85.67%，低于 c-VEP 97.81% 和 SSVEP 93.42%，高于 SSMVEP 64.91%。
- 分类器使用统一的 template-matching CCA，不是 EEGNet/DL。

结论：

- c-MVEP 已经不只是单刺激神经响应实验，而是做了 4-class online BCI。
- 但它仍是新范式，公开 DL baseline、公开大规模数据集、多目标 speller 系统都还不成熟。


### 4.4 非 SSVEP 范式可直接引用的解码论文清单

| 范式 | 论文 | 解码器 | 是否 DNN | 可用于论文中的表述 |
|---|---|---|---|---|
| SSMVEP | Xie et al., **Steady-state motion visual evoked potentials produced by oscillating Newton's rings: implications for brain-computer interfaces**, PLoS ONE, 2012 | 传统频率识别/相关性分析 | 否 | SSMVEP 的早期范式来源，证明运动刺激可诱发 steady-state motion VEP |
| SSMVEP | Atabek et al., **A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials**, 2024 | SNR、CCA、FBCCA | 否 | 新型不可感知 motion grating SSMVEP；非 DNN；多目标分类仍受限 |
| SSMVEP | Scheppink et al., **Beyond Flickering: Introducing Code-Modulated Motion Visual Evoked Potentials for Brain-Computer Interfacing**, 2026 | template-matching CCA；同时评估 SSMVEP | 否 | 同一 online 4-class 框架下比较 SSMVEP、SSVEP、c-VEP、c-MVEP |
| c-VEP | Martínez-Cagigal et al., **Brain–computer interfaces based on code-modulated visual evoked potentials (c-VEP): a literature review**, Journal of Neural Engineering, 2021 | 综述模板匹配、CCA/reconvolution CCA 等 | 否 | c-VEP 主流传统解码综述，可说明 DL 不是该领域传统主线 |
| c-VEP | Miao et al., **High-Performance c-VEP-BCI Under Minimal Calibration**, Expert Systems with Applications, 2024 | 高性能 c-VEP 低校准解码，偏传统模板/CCA路线 | 否 | 支撑 c-VEP 在少校准条件下已有强传统 baseline |
| c-VEP | Nair and Cecotti, **Deep Learning Architectures for Code-Modulated Visual Evoked Potentials Detection**, 2025 | CNN、Siamese network、CCA+BLDA、Corr+BLDA | 是 | 当前最直接的 c-VEP DNN 解码论文，可作为 c-VEP-DL baseline 入口 |
| c-MVEP | Scheppink et al., **Beyond Flickering: Introducing Code-Modulated Motion Visual Evoked Potentials for Brain-Computer Interfacing**, 2026 | template-matching CCA | 否 | c-MVEP 首个系统范式论文之一；说明 c-MVEP-DNN 仍是空白方向 |

注意：有一类 **motion-onset VEP / m-VEP** 工作使用过 deep learning 或 compressed sensing，但它不是 SSMVEP，也不是 c-MVEP。SSMVEP 是 steady-state motion stimulation，c-MVEP 是 code-modulated motion stimulation，不能把 motion-onset VEP 的 DNN 论文直接当成 SSMVEP/c-MVEP 的 DNN 证据。

## 5. “SSMVEP、c-MVEP 是否只能表示一种任务类型？”

答案：**不是。它们表示的是刺激编码范式，不是任务类型。**

更准确的逻辑是：

1. **范式层**：SSVEP、SSMVEP、c-VEP、c-MVEP 定义“如何让视觉刺激携带可区分标签”。  
   - SSVEP：不同频率/相位。  
   - SSMVEP：不同运动频率/运动参数。  
   - c-VEP：不同 code sequence 或同一 code 的不同 circular shift。  
   - c-MVEP：用 motion 承载 code sequence。

2. **任务层**：speller、菜单选择、轮椅控制、机械臂控制、智能家居控制、游戏控制等。  
   同一个刺激范式可以映射到不同任务：例如 4 个目标可对应“上/下/左/右”，40 个目标可对应字符，8 个目标可对应菜单命令。

3. **为什么当前 SSMVEP/c-MVEP 看起来任务少**：  
   - SSMVEP 信号通常弱于 SSVEP。  
   - motion 刺激的多目标同时呈现更难，尤其是不可感知 motion。  
   - c-MVEP 是 2026 年才系统提出的新范式。  
   - 缺少像 SSVEP benchmark/BETA 这样的 40-target 大规模数据集。  
   - DL baseline 不成熟，很多工作仍用 CCA/template matching。

因此，如果要做实验，不能说“SSMVEP/c-MVEP 只能表示一种任务”；应表述为：**当前 motion-based VEP 的多目标、实时、高准确率应用还处于早期阶段，任务范围受工程成熟度限制，而不是受范式定义限制。**

## 6. 传统 SSVEP Speller 原理

### 6.1 编码

屏幕显示一个字符矩阵，例如 40 个字符。每个字符块都以一个唯一标签闪烁：

- 方式 1：每个字符一个不同频率，例如 A=8.0 Hz，B=8.2 Hz，C=8.4 Hz。
- 方式 2：频率 + 相位联合编码，例如同一频率下用不同相位扩充目标数。
- 方式 3：更现代的系统可能加入空间编码、高密度 EEG 或宽带随机刺激。

### 6.2 用户行为

用户只需要注视想输入的字符。因为视觉系统对周期性刺激会产生稳态响应，枕区 EEG 中会出现该字符对应的刺激频率及其谐波。

例如用户看 B，B 的刺激为 8.2 Hz，则 EEG 在 Oz/O1/O2 等枕区通道中会出现 8.2 Hz、16.4 Hz、24.6 Hz 等频率成分。

### 6.3 解码

传统方法不是“读出字符语义”，而是做模板匹配：

1. 对每个候选字符构造参考信号。  
   例如 8.2 Hz 参考模板包含 sin(2πft)、cos(2πft) 及若干谐波。

2. 计算 EEG 与每个候选模板的相似度。  
   常见方法：CCA、FBCCA、TRCA/eTRCA、template matching。

3. 选择相似度最高的目标。  
   如果 EEG 与 8.2 Hz 模板相关性最大，则输出 B。

4. DL 方法则直接学习从多通道 EEG 到字符类别的映射。  
   但其本质标签仍来自刺激编码：模型学到的是“该 EEG 更像哪个频率/相位/模板诱发的响应”。

### 6.4 为什么能区分很多字符？

SSVEP speller 的可扩展性来自频率、相位、空间、时间窗口的组合：

- 频率越多，可编码目标越多，但频率间隔太近会互相混淆。
- 加入相位后，同一频段可以扩展更多目标。
- 更长时间窗口通常提高准确率，但降低输入速度。
- 更强算法如 FBCCA/TRCA/DNN 可缩短刺激时间。
- BETA 数据集就是 70 名被试、40-target cued-spelling EEG 数据，用来评估真实 speller 场景。

## 7. 对“物理刺激攻击实验”的建议

如果你的目标是“通过物理刺激模拟攻击，让 BCI 模型产生危险响应，并找出哪些物理刺激更容易让模型失效”，建议优先级如下：

1. **首选 SSVEP Speller / 多目标选择。**  
   原因：应用真实、DL baseline 多、标签清楚、模型失效可直接表现为字符/命令误选。

2. **第二选择 c-VEP。**  
   原因：c-VEP 是高性能 speller 方向，code-based 标签与时间扰动、延迟、刺激遮挡、屏幕刷新异常关系更直接；但 DL 文献比 SSVEP 少。

3. **探索性选择 SSMVEP/c-MVEP。**  
   原因：motion-based 刺激更贴近“低闪烁舒适 BCI”的未来方向，攻击新颖；但需要承担多目标任务设计、基线构建和数据采集成本。

4. **不要只做神经科学式“刺激是否诱发差异”。**  
   BCI 安全论文应聚焦：物理刺激如何改变模型决策边界、错误命令率、攻击成功率、恢复时间、用户疲劳、安全风险，而不仅是 EEG 频谱是否变化。


### 7.1 SSVEP 实验中混入高频 SSMVEP 是否会影响 EEGNet？

结论不能简单说“一定影响”或“一定被过滤”。实际取决于 **高频 SSMVEP 是否在 EEG 中形成可测响应、预处理滤波是否保留该频段、以及扰动是否通过注意/疲劳/低频包络间接改变 SSVEP 特征**。

1. **如果高频 SSMVEP 成分被预处理滤掉，直接频率成分影响会很小。** 例如 72Hz SSMVEP 若使用 6-40Hz、7-30Hz 或只围绕 SSVEP 目标频率的 band-pass/filter-bank，72Hz 主峰通常不会进入 EEGNet 输入。
2. **如果预处理保留高频，EEGNet 可能受影响。** EEGNet 的 temporal convolution 会学习频域滤波器；如果训练集中没有这种高频 motion-induced response，测试时混入 72Hz 或其谐波/旁瓣，会形成 out-of-distribution 输入，可能降低分类置信度或造成误分类。
3. **即使 72Hz 主频被滤掉，也可能通过间接路径影响 SSVEP。** 高频 motion 刺激可能改变注意分配、视觉疲劳、枕区背景状态，或产生低频 envelope、刷新率 aliasing、谐波/亚谐波、眼动/肌电伪迹；这些成分若落入 SSVEP 特征带，就可能影响分类。
4. **如果 SSMVEP 频率与 SSVEP 目标/谐波频段隔离，并且刺激位置/亮度不干扰目标注视，影响可能较弱。** 这类情况更像被滤除的无关高频背景。
5. **如果高频 motion 与目标字符空间重叠，或产生与某个 SSVEP 类别相近的频率/相位结构，影响会显著增大。** 这时攻击不一定靠 72Hz 本身，而是靠改变目标 SSVEP 的幅值、相位、SNR 或注意资源。

实验上应同时测试四种输入：clean SSVEP、SSVEP+高频 SSMVEP raw broadband、SSVEP+高频 SSMVEP after SSVEP-band filter、SSVEP+高频 SSMVEP with only low-frequency envelope retained。若 broadband 下出错而 band-limited 后恢复，说明主要是高频/OOD 影响；若 band-limited 后仍出错，说明扰动通过注意、低频包络或 SSVEP 特征调制产生影响。

## 8. 可引用论文与链接

1. Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces. arXiv:1611.08024, 2018. https://arxiv.org/abs/1611.08024
2. Waytowich N R, Lawhern V J, Garcia J O, et al. Compact Convolutional Neural Networks for Classification of Asynchronous Steady-state Visual Evoked Potentials. arXiv:1803.04566, 2018. https://arxiv.org/abs/1803.04566
3. Guney O B, Oblokulov M, Ozkan H. A Deep Neural Network for SSVEP-based Brain-Computer Interfaces. IEEE Transactions on Biomedical Engineering, 2022. https://arxiv.org/abs/2011.08562
4. Chen J, Zhang Y, Pan Y, et al. A Transformer-based deep neural network model for SSVEP classification. arXiv:2210.04172, 2022. https://arxiv.org/abs/2210.04172
5. Luo W, Yin W, Liu Q, et al. A Hybrid Brain-Computer Interface Using Motor Imagery and SSVEP Based on Convolutional Neural Network. arXiv:2212.05289, 2022. https://arxiv.org/abs/2212.05289
6. Guney O B, Kucukahmetler D, Ozkan H. Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces. Journal of Neural Engineering, 2025. https://arxiv.org/abs/2305.17403
7. Chen S Y, Chang C M, Chiang K J, Wei C S. SSVEP-DAN: A Data Alignment Network for SSVEP-based Brain Computer Interfaces. arXiv:2311.12666, 2023. https://arxiv.org/abs/2311.12666
8. Huang Y, Chen Y, Cao L, et al. IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification. arXiv:2502.13972, 2025. https://arxiv.org/abs/2502.13972
9. Liu B, Huang X, Wang Y, et al. BETA: A Large Benchmark Database Toward SSVEP-BCI Application. Frontiers in Neuroscience, 2020. https://arxiv.org/abs/1911.13045
10. Nair K, Cecotti H. Deep Learning Architectures for Code-Modulated Visual Evoked Potentials Detection. arXiv:2511.21940, 2025. https://arxiv.org/abs/2511.21940
11. Atabek B, Yilmaz E, Acarturk C, Cakir M P. A Grating Based High-Frequency Motion Stimulus Paradigm for Steady-State Motion Visual Evoked Potentials. arXiv:2312.15682, 2024. https://arxiv.org/abs/2312.15682
12. Scheppink H, Herpers R, Thielen J, Volosyak I. Beyond Flickering: Introducing Code-Modulated Motion Visual Evoked Potentials for Brain-Computer Interfacing. arXiv:2605.15801, 2026. https://arxiv.org/abs/2605.15801
13. Shi N, Miao Y, Huang C, et al. Estimating and approaching maximum information rate of noninvasive visual brain-computer interface. arXiv:2308.13232, 2023. https://arxiv.org/abs/2308.13232
14. Ming G, Pei W, Tian S, et al. High-Density EEG Enables the Fastest Visual Brain-Computer Interfaces. arXiv:2507.17242, 2025. https://arxiv.org/abs/2507.17242

## 9. 一句话可用于汇报

SSVEP、SSMVEP、c-VEP、c-MVEP 不是任务类型，而是视觉刺激编码方式；其中 SSVEP 已经形成从 CCA/FBCCA/TRCA 到 EEGNet-like CNN、专用 DNN、Transformer、SFDA 的成熟解码体系，适合作为物理刺激攻击和模型鲁棒性实验的主线；SSMVEP/c-MVEP 更适合做低闪烁、低疲劳、motion-based VEP 的前沿扩展，但当前多目标任务和 DL baseline 仍不成熟，需要在实验设计中明确其探索性。
