# BCI 物理刺激攻击实验范式选择依据与应用意义

## 目标

我们需要选择一个或多个 BCI 实验范式，用于研究 **Adversarial Stimuli / Physical Sensory-Stimulus Attack**：攻击者不接触 EEG 设备、模型、采集链路或训练数据，只通过屏幕、灯光、音频、触觉或反馈事件改变用户接收到的物理刺激，使 BCI 模型产生错误甚至危险响应。

范式选择不能只看“能不能攻击”，还要看：

1. 实际应用是否广泛；
2. 错误输出是否有现实危害；
3. 攻击者是否真的能从外部影响刺激；
4. 实验是否容易做出稳定结果；
5. 论文是否有清晰创新点。

## 总体结论

如果只能选择一个主范式，建议选择：

> **SSVEP**

理由：SSVEP 的类别本身由外部视觉频率/相位编码，攻击者只要能影响屏幕、灯光或刷新时序，就可能改变用户产生的 EEG 响应。它的物理攻击路径最直接，实验可控性最高，危险响应也容易定义，例如 `stop -> forward`、`left -> right`。

如果希望论文应用意义更强，建议采用：

> **SSVEP 主实验 + P300 辅助实验**

理由：

- SSVEP 适合证明物理刺激攻击的可行性和可控性；
- P300 更贴近残障辅助沟通、yes/no 医疗选择、拼写器等高社会意义场景；
- 二者都是视觉诱发 BCI，攻击者可通过界面刺激影响系统；
- 二者结合比单独 MI 更有新意，因为 2023 SMC 已经做了 MI 视觉闪烁攻击。

## 范式应用场景对比

| 范式 | 主要实际应用 | 应用广度 | 错误输出危害 | 与物理刺激攻击的关系 |
| --- | --- | --- | --- | --- |
| SSVEP | 命令选择、拼写器、轮椅/机器人控制、无人机控制、智能家居、VR/AR 控制、游戏交互 | 高 | 高 | 类别直接由视觉频率/相位刺激编码，最适合物理刺激攻击。 |
| P300 | ALS/locked-in 辅助沟通、P300 speller、yes/no 选择、护理/医疗菜单、环境控制 | 高 | 高 | 依赖 oddball 高亮刺激，攻击者可改变颜色、形状、时序、语义和注视。 |
| MI | 神经康复、卒中康复、假肢/机械臂控制、轮椅控制、外骨骼、运动意图识别 | 高 | 高 | 应用意义强，但外部刺激对类别不是直接编码，攻击链更间接。 |
| cVEP | 高速视觉拼写器、多目标命令选择、低校准视觉 BCI | 中 | 中高 | 依赖 stimulus code/sequence，攻击面清楚，但实验实现时序要求高。 |
| ErrP/ERN | BCI 错误检测、在线纠错、共享控制机器人、人机协作、自适应系统 | 中 | 中高 | 攻击点是反馈事件，适合作为安全层攻击，但不是主要控制范式。 |
| Passive/NS | 情绪识别、疲劳监测、注意力监测、工作负荷评估、VR/教育/驾驶辅助 | 高 | 中 | 应用广，但危险响应定义不如控制/选择范式清晰。 |

## 各范式具体分析

### 1. SSVEP

#### 实际应用

SSVEP 常用于高吞吐量命令选择系统。典型应用包括：

- BCI speller；
- 虚拟键盘；
- 轮椅方向控制；
- 机械臂/机器人控制；
- 无人机控制；
- 智能家居开关控制；
- VR/AR 中的视觉菜单选择；
- 游戏和消费级 BCI 交互。

#### 为什么应用范围广

SSVEP 的优点是：

- 信噪比较高；
- 训练时间相对短；
- 信息传输率较高；
- 目标数量可扩展；
- 模型成熟，例如 CCA、FBCCA、TRCA、EEGNet。

#### 为什么适合物理刺激攻击

SSVEP 的类别就是由外部闪烁频率或频率-相位组合编码的。因此攻击者只要能影响视觉刺激，就能影响用户产生的 EEG。

可攻击参数：

- flicker frequency；
- phase；
- contrast；
- luminance；
- background flicker；
- harmonic interference；
- refresh jitter；
- frame drop。

#### 危险响应示例

| 用户意图 | 被攻击后输出 | 危险含义 |
| --- | --- | --- |
| stop | forward | 虚拟轮椅/机器人继续前进。 |
| left | right | 控制方向相反。 |
| emergency | normal command | 紧急求助失败。 |
| off | on | 智能设备错误启动。 |

#### 结论

SSVEP 是最推荐的主范式。它兼具应用广度、实验可控性、攻击现实性和结果解释性。

### 2. P300

#### 实际应用

P300 主要用于辅助沟通和选择系统，尤其适用于严重运动障碍用户：

- P300 speller；
- ALS/locked-in 用户沟通；
- yes/no 二分类选择；
- 护理需求表达；
- 医疗菜单选择；
- 环境控制；
- 简单网页/菜单导航。

#### 为什么应用意义大

P300 的社会意义非常强，因为它直接关系到残障用户表达意图的能力。对于无法说话或无法运动的人，错误选择不仅是分类错误，还可能影响护理、沟通和自主性。

#### 为什么适合物理刺激攻击

P300 依赖 oddball 刺激，高亮方式、颜色、形状、时序、注视和语义都会影响目标/非目标 ERP 可分性。

可攻击参数：

- highlighter color；
- shape；
- luminance；
- stimulus onset asynchrony；
- adjacent flashing；
- semantic cue；
- gaze lure；
- false feedback。

#### 危险响应示例

| 用户意图 | 被攻击后输出 | 危险含义 |
| --- | --- | --- |
| yes | no | 医疗/护理选择被反转。 |
| help | no help | 求助失败。 |
| pain | no pain | 痛苦表达失败。 |
| target character | wrong character | 沟通内容被篡改。 |

#### 结论

P300 是最适合作为辅助范式的选择。它的物理攻击不如 SSVEP 直接，但应用意义更强，尤其适合强调 disabled-aided BCI 和医疗辅助安全。

### 3. MI

#### 实际应用

MI 是 BCI 中最经典的主动控制范式之一，常用于：

- 卒中后运动康复；
- 神经反馈训练；
- 假肢控制；
- 机械臂控制；
- 轮椅控制；
- 外骨骼控制；
- 手部/脚部运动意图识别。

#### 为什么应用意义大

MI 对运动障碍、卒中康复和辅助控制非常重要。错误输出可能导致虚拟轮椅错误转向、机械臂误动作或康复反馈错误。

#### 为什么不是最推荐主范式

MI 的问题是：

- 类别不是由外部刺激直接编码；
- 物理刺激对 MI 的影响更间接，通常通过注意分散、视觉诱发响应、反馈干扰或 mu/beta rhythm 改变实现；
- 2023 SMC 已经做了 MI 视觉闪烁攻击，继续单独做 MI 容易被认为是复现或小改动。

#### 适合的角色

MI 更适合作为：

- 2023 SMC 的复现实验；
- 与 SSVEP/P300 的对照；
- 证明非视觉诱发范式也会被视觉刺激影响的补充实验。

#### 结论

MI 应用意义很大，但不建议作为唯一主范式。它适合作为对照或扩展，而不是论文的核心创新点。

### 4. cVEP

#### 实际应用

cVEP 常用于：

- 高速视觉拼写器；
- 多目标视觉命令选择；
- 低校准视觉 BCI；
- 基于编码序列的视觉交互。

#### 优点

- 信息传输率高；
- 目标数量可扩展；
- stimulus code 可以精确设计；
- 攻击面非常清楚，即 code/sequence/timing。

#### 局限

- 实验实现要求高；
- 需要精确控制显示刷新和编码序列；
- 应用普及程度不如 SSVEP 和 P300。

#### 结论

cVEP 很适合作为后续深入方向，但不建议作为第一篇主范式。除非你的团队已经有稳定 cVEP 实验平台。

### 5. ErrP / ERN

#### 实际应用

ErrP/ERN 主要不是独立控制范式，而是安全与纠错模块：

- BCI 输出错误检测；
- 在线纠错；
- 共享控制机器人；
- 人机协作；
- 自适应 BCI；
- reinforcement learning from human EEG feedback。

#### 为什么有意义

ErrP/ERN 是安全闭环的重要部分。如果攻击者通过错误反馈、延迟反馈或多模态冲突影响 ErrP，系统可能错误纠正或漏纠正。

#### 局限

- 危险响应不如 SSVEP/P300 直观；
- 实验设计复杂；
- 需要构造 feedback-based closed-loop；
- 不适合作为第一阶段主范式。

#### 结论

ErrP/ERN 适合作为后续扩展，尤其当论文想讨论 BCI 安全闭环时。

### 6. Passive BCI / Natural Stimulus / 情绪识别

#### 实际应用

这类范式应用范围很广：

- 情绪识别；
- 疲劳监测；
- 注意力监测；
- 工作负荷评估；
- 驾驶辅助；
- VR/AR 自适应界面；
- 教育反馈；
- 神经营销；
- 心理健康监测。

#### 为什么不适合第一篇

虽然应用范围广，但问题是：

- 输出不是明确控制命令；
- “危险响应”难定义；
- 物理刺激本身就可能合理改变情绪/注意，容易被质疑不是攻击；
- 机制解释复杂。

#### 结论

Passive BCI 适合做隐私或状态误判研究，不适合作为第一篇 physical adversarial stimuli integrity attack 的主范式。

## 范式选择评分矩阵

评分范围 1-5，分数越高越适合当前研究目标。

| 范式 | 应用广度 | 攻击现实性 | 危险响应清晰度 | 实验可控性 | 结果稳定性 | 新意 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SSVEP | 5 | 5 | 5 | 5 | 5 | 4 | 29 |
| P300 | 5 | 4 | 5 | 4 | 4 | 4 | 26 |
| cVEP | 3 | 5 | 4 | 4 | 4 | 5 | 25 |
| MI | 5 | 3 | 5 | 3 | 3 | 2 | 21 |
| ErrP/ERN | 3 | 4 | 4 | 3 | 3 | 4 | 21 |
| Passive/NS | 5 | 3 | 2 | 3 | 3 | 3 | 19 |

## 最终推荐

### 最推荐方案：SSVEP 主实验 + P300 辅助实验

这是当前最平衡的组合。

#### 为什么选 SSVEP 做主实验

- 应用广泛：拼写器、机器人、轮椅、智能家居、VR/AR。
- 攻击现实：攻击者只需要控制屏幕/灯光/刷新时序。
- 参数明确：频率、相位、对比度、谐波、背景闪烁。
- 危险响应清楚：stop/go、left/right、emergency/normal。
- 模型成熟：CCA、FBCCA、TRCA、EEGNet。
- 机制好解释：看 SSVEP SNR、harmonic power、CCA correlation margin。

#### 为什么加 P300

- P300 更贴近 disabled-aided communication。
- yes/no、help/no help、pain/no pain 等错误选择非常有现实危害。
- 可以强调医疗辅助和残障用户安全。
- 与 SSVEP 共同覆盖视觉诱发 BCI 的两个代表方向。

### 可选方案：MI 作为对照

如果时间允许，可以加入 MI 作为小规模对照：

- 复现 2023 SMC 的视觉闪烁攻击；
- 与 SSVEP/P300 比较：视觉诱发范式是否更容易被物理刺激攻击；
- 说明物理刺激攻击不仅限于视觉诱发范式，连 MI 也会受影响。

但 MI 不建议作为主实验，因为已有直接前作，且攻击机制更间接。

## 论文定位建议

如果选择 SSVEP + P300，论文可以这样定位：

> We focus on visual-evoked BCI paradigms because their decoding targets are directly coupled with externally presented sensory stimuli. This makes them both practically important and naturally exposed to physical sensory-stimulus attacks. Among them, SSVEP provides a highly controllable frequency/phase-coded command interface, while P300 represents assistive communication systems where incorrect selections can have serious consequences for disabled users.

中文表述：

> 我们选择 SSVEP 和 P300 作为主要实验范式，是因为它们广泛应用于命令选择、拼写器、辅助沟通和环境控制等真实 BCI 场景，并且其解码目标与外部视觉刺激直接耦合。这使得攻击者无需接触 EEG 设备，仅通过操纵屏幕或环境刺激就可能影响用户脑电响应和模型输出。其中，SSVEP 适合验证频率/相位物理攻击的可行性，P300 则更能体现残障辅助沟通场景中的安全危害。

## 一句话结论

> 如果只选一个范式，选 SSVEP；如果想让论文应用意义更强，选 SSVEP + P300；如果想和 2023 SMC 建立直接联系，再加入 MI 作为复现和对照。
