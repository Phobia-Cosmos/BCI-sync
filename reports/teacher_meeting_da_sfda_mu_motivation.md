# 明日汇报材料：为什么做 DA + SFDA + MU 的隐私保护 EEG-BCI

> 目标：用一条清晰逻辑向老师说明：我们的研究不是为了硬凑 DA/SFDA/MU，而是由 **BCI 个性化部署 + 神经数据隐私法规 + 用户撤权 + 源数据不可再访问** 共同推出的技术问题。

## 0. 一句话汇报主线

**EEG-BCI 必须个性化，所以需要 DA；但个性化适配通常需要访问源用户或目标用户 EEG，而脑电是敏感神经数据，用户又有撤回/删除权；现实中源数据往往因法规、伦理、机构隔离或用户撤权不可再访问，所以需要 SFDA；而 SFDA 只是不访问源数据，不能删除源模型里已经记住的用户影响，因此还需要 MU。**

可以压缩成：

```text
BCI 个性化需求
    ↓
跨用户/跨 session/domain shift
    ↓
需要 DA 适配新用户/新 session
    ↓
传统 DA 依赖源 EEG/目标校准数据
    ↓
神经数据敏感 + GDPR/PIPL/神经数据法规 + 用户撤权
    ↓
源数据不可再访问 → 需要 SFDA
    ↓
源模型仍可能残留源用户信息 → 需要 MU
```

## 1. 真实问题是什么？

### 1.1 BCI 不是普通分类器，而是个性化设备

EEG-BCI 的模型必须适配具体用户，因为：

- 不同用户的头皮传导、脑区位置、运动想象策略、注意力和疲劳状态不同；
- 同一用户不同天的电极位置、阻抗、精神状态、任务熟练度也不同；
- BCI 用户会随着反馈训练改变自己的神经策略，模型面对的目标分布会持续变化；
- 临床/家庭/学校/企业场景的噪声和设备条件不同。

所以在 BCI 中，一个用户、一个 session、一个设备、一个医院都可以看成一个 **domain**。

### 1.2 不做 DA 的后果不是抽象掉点

| 场景 | 不做 DA 的结果 | 现实后果 |
|---|---|---|
| 机器人手/机器人臂 BCI | 解码错误或控制不稳定 | 用户抓错、误动、失去信任；临床/家庭不可用 |
| 拼写器/通信 BCI | P300/SSVEP 识别错误 | locked-in/ALS 用户表达错误意愿 |
| 康复 BCI | 错误识别运动意图 | 错误反馈，训练效果下降 |
| 家庭长期使用 | 每天 EEG 分布不同 | 反复校准，用户负担高，设备被弃用 |
| 多中心医疗部署 | 医院/设备/人群差异 | A 医院模型到 B 医院失效 |

因此 DA 不是为了多做一个算法模块，而是 BCI 真实部署的必要条件。

## 2. 有哪些论文支撑“必须做 DA”？

### 2.1 Scientific Data 2022：同一用户跨天就会掉点

Ma et al. 2022 的多 session MI-BCI 数据集包含 25 名被试、5 天 session。论文报告：

- within-session classification：最高平均 68.8%；
- cross-session classification：下降到 53.7%；
- cross-session adaptation：提升到 78.9%。

这说明同一用户不同天已经发生明显 domain shift。不适配时性能接近不可用，适配后显著恢复。

引用：Ma J, Yang B, Qiu W, et al. *A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface*. Scientific Data, 2022.

### 2.2 Nature Communications 2025：实时机器人手指控制需要 fine-tuning

Ding et al. 2025 的非侵入式 EEG 机器人手指控制使用 EEGNet + same-day fine-tuning。论文明确指出 fine-tuning 用来缓解 inter-session variability，并且 fine-tuned models 优于 base models。

这说明即使是有经验 BCI 用户，实时物理控制也需要 session-specific adaptation。

引用：Ding Y, Udompanyawit C, Zhang Y, et al. *EEG-based brain-computer interface enables real-time robotic hand control at individual finger level*. Nature Communications, 2025.

### 2.3 Scientific Data 2023：个性化 BCI 与用户画像相关

Dreyer et al. 2023 的 MI-BCI 数据库包含 87 名参与者，同时记录 demographic、personality、cognitive traits 和 BCI performance。

这说明 BCI 个性化不仅与 EEG 信号有关，也与用户画像/认知特质相关。换句话说，模型为了个性化可能会学习隐私属性。

引用：Dreyer P, Roc A, Pillette L, et al. *A large EEG database with users’ profile information for motor imagery brain-computer interface research*. Scientific Data, 2023.

## 3. 隐私保护为什么是刚需？

### 3.1 EEG/神经数据比普通数据更敏感

EEG 可能包含：

- 任务意图：左/右手运动想象、视觉选择、P300 目标响应；
- 身份信息：个体 EEG pattern；
- 状态信息：疲劳、注意、情绪、认知负荷；
- 健康信息：神经疾病、睡眠、认知异常；
- 熟悉性/记忆反应：用户是否认识某个刺激。

Nature 2017 的 neurotechnology ethics 文章强调 AI 和 BCI 必须保护 privacy、identity、agency 和 equality。Nature Protocols 2023 进一步指出，AI 和数据聚合工具可以解码/分析包含高度敏感信息的 neurodata，威胁 neuroprivacy。

### 3.2 法规已经把“撤权/删除”变成现实要求

#### GDPR

GDPR Article 17 规定数据主体有权要求删除个人数据，特别是在：

- 数据不再为原目的所必需；
- 数据主体撤回同意且没有其他合法依据；
- 数据被非法处理等情况。

这对应我们的“用户撤权/删除请求”场景。

#### PIPL

中国《个人信息保护法》强调个人信息处理需要合法基础和同意，个人可以撤回同意，并享有删除、更正、访问等权利。敏感个人信息包括医疗健康、生物识别、金融、位置等；脑电虽然未被逐字列为“神经数据”，但在 BCI 场景中通常与生物识别、健康和认知状态高度相关，应按高敏感数据处理。

#### 神经数据专项立法趋势

- Colorado HB24-1058 已将 biological data / neural data 纳入敏感数据保护范围；
- California SB-1223 将 neural data 纳入 sensitive personal information，并定义为测量中枢或外周神经系统活动生成的信息。

这说明“神经数据隐私”已经不是纯伦理话题，而是合规问题。

## 4. 为什么传统 DA 不够？

传统 DA 的默认假设经常是：

```text
有源域数据 D_s + 有目标域数据 D_t
通过对齐 / fine-tuning / pseudo-labeling 适配目标用户
```

但 BCI 真实场景中这个假设不成立：

| 现实限制 | 为什么源数据不可访问 |
|---|---|
| 医疗伦理 | 患者 EEG 不能随意跨机构共享 |
| 法规合规 | GDPR/PIPL/神经数据法规限制二次处理和跨境流转 |
| 用户撤权 | 用户撤回同意后不能继续使用其 EEG |
| 商业部署 | 厂商只发布模型，不发布源训练数据 |
| 数据稀缺 | 残疾/临床用户数据少，采集成本高 |
| 多中心差异 | 医院、设备、协议不同，集中重训困难 |
| 元数据不完整 | 历史 subject/session/device 元数据可能缺失或不可共享 |

所以我们的第一个转向是：从普通 DA 转向 **Source-Free DA**。

## 5. 为什么需要 SFDA？

SFDA 的设定更贴近真实 BCI：

```text
可用：源模型 f_s + 未标注目标 EEG D_t
不可用：原始源 EEG D_s
目标：适配新用户/新 session
```

它解决的是：

- 不再需要把源用户 EEG 交给目标端；
- 符合医院/学校/企业/消费设备中数据隔离要求；
- 支持新用户个性化；
- 减少 raw source EEG exposure。

但是必须强调：**SFDA 不是完整隐私保护。**

因为源模型本身可能已经记住：

- 某个源用户是否参与训练；
- 源用户的身份特征；
- 源 session / 源设备 / 源医院的 domain 特征；
- 某些健康或认知状态相关表示。

所以，SFDA 只能解决“不访问源数据”，不能解决“源模型残留隐私”。

## 6. 为什么还需要 MU？

### 6.1 用户撤权后，仅删除原始 EEG 不够

如果一个用户说：

> 我撤回授权，请删除我的脑电数据。

系统可以删除数据库中的原始 EEG 文件，但模型参数中可能已经学习了这个用户的模式。此时如果只删文件，不处理模型，撤回是不完整的。

### 6.2 不能重新训练，所以需要 MU

理论上最干净的方法是：

```text
从训练集中删除用户 u 的数据
用剩余数据从头重新训练模型
```

但在真实 BCI 中很难：

- 源数据已经因法规/伦理不可访问；
- 多机构数据无法重新汇总；
- 训练成本高；
- retain data 可能也不可访问；
- 历史元数据不完整；
- 设备已经部署在目标端，不能频繁全量重训。

因此需要 **Machine Unlearning**：在不完全重训的情况下，尽量删除指定 subject/session/domain 对模型的影响。

### 6.3 EEG 中 forget unit 应该是什么？

不要把 forget unit 设成任务类别，例如“左手 MI 类”。那会破坏 BCI 功能。

更合理的是：

- forget subject：某个用户撤权；
- forget session：某次采集不再授权；
- forget hospital/device/domain：某机构或设备域不再可用；
- forget attribute：降低年龄、性别、身份等隐私属性泄漏。

## 7. 为什么 FL / DP 不能单独解决？

这里要说得严谨：FL 和 DP 不是没用，而是**解决的问题不同，不足以覆盖我们的场景**。

| 方法 | 能解决什么 | 为什么不够 |
|---|---|---|
| FL | 训练时不集中上传原始数据 | 需要源客户端/数据仍参与训练；不能处理已训练模型中的撤回用户影响；通信和异构设备成本高；仍可能有梯度/更新泄漏 |
| DP | 训练时限制单个样本对模型影响 | 必须在训练阶段设计；强噪声会损害低信噪比 EEG 性能；不能针对某个撤回用户做 post-hoc 删除；不能恢复 source-free 适配性能 |
| 普通 DA | 提升目标域性能 | 通常需要源数据或目标标注/校准数据，不符合隐私和撤权约束 |
| SFDA | 不访问源数据做目标适配 | 源模型仍可能残留源用户隐私 |
| MU | 删除指定数据/用户影响 | 单独 MU 不解决新目标用户适配，需要和 SFDA/DA 结合 |

因此我们的组合是合理的：

```text
DA：解决 BCI 个性化和跨域性能问题
SFDA：解决源 EEG 不可访问/不能共享问题
MU：解决用户撤权和源模型残留隐私问题
```

## 8. 我们要解决的问题定义

### 8.1 现实场景

一个 BCI 系统由多个历史用户/医院/session 训练出源模型。现在：

1. 新用户要使用设备，需要个性化适配；
2. 原始源 EEG 因隐私/法规/机构隔离不可访问；
3. 某个源用户或某个 session 撤回授权；
4. 系统仍需要保留对新用户的解码性能。

### 8.2 技术问题

给定：

```text
source model: f_s
forget request: subject/session/domain z_f
target data: unlabeled target EEG D_t
source data: unavailable
retain source data: unavailable or restricted
```

目标：

```text
1. remove influence of z_f from f_s
2. adapt model to target user/session without raw source EEG
3. preserve target decoding accuracy
4. reduce privacy leakage: membership/identity/attribute inference
```

## 9. 我们的方法该如何讲？

可以这样描述：

> 我们提出一个面向个性化 BCI 的隐私感知 source-free adaptation 框架。首先，针对用户撤权或敏感源域，执行 subject/session/domain-level machine unlearning，减少源模型中的残留隐私影响；然后，在不访问源 EEG 的条件下，利用未标注目标 EEG 做 source-free domain adaptation，获得目标用户的个性化解码器；最后，同时评估任务性能和隐私泄漏，包括 target accuracy、adaptation gain、membership inference、identity inference 和 attribute inference。

流程：

```text
Historical EEG users → Source model
              ↓
      User/session withdrawal
              ↓
Source-free / approximate MU
              ↓
Sanitized source model
              ↓
Unlabeled target EEG
              ↓
SFDA / CTTA personalization
              ↓
Personalized privacy-aware BCI decoder
```

## 10. 明天汇报可用的 2 分钟话术

老师，我们这个工作的核心不是单纯把 DA、SFDA、MU 拼在一起，而是来自 BCI 设备真实部署的三个约束。

第一，BCI 本身是高度个性化的。不同用户、不同天、不同设备下 EEG 分布差异很大。Nature Communications 2025 的实时 EEG 机器人手指控制需要 same-day fine-tuning；Scientific Data 2022 的多 session MI 数据集也显示，不做 cross-session adaptation 时性能会明显下降。因此 DA 是 BCI 真实可用的基础。

第二，BCI 的数据又非常敏感。EEG 不只包含任务意图，还可能包含身份、健康、认知状态和用户画像。Nature 2017 和 Nature Protocols 2023 都强调 neurotechnology/AI 会带来 neuroprivacy 风险。现在 Colorado 和 California 也已经把 neural data 纳入敏感数据保护范围，GDPR 和 PIPL 也都支持用户撤回和删除权。因此，在 BCI 中不能假设源 EEG 可以一直被访问和重复使用。

第三，用户撤权会带来模型层面的问题。即使删除了原始 EEG，源模型可能已经记住了用户特征。重新训练理论上最干净，但真实情况下源数据和 retain data 可能因为法规、伦理或机构隔离不可再访问。因此我们需要 source-free setting：用源模型和未标注目标 EEG 做个性化适配；同时需要 machine unlearning：删除撤回用户/session/domain 在模型中的残留影响。

所以我们的研究问题是：在源 EEG 不可访问、存在用户撤权的情况下，如何让 BCI 仍能对新用户进行个性化适配，同时降低源模型中的隐私泄漏。这就是我们做 DA + SFDA + MU 的原因。

## 11. 导师可能追问与回答

### Q1：为什么不用普通 DA？

普通 DA 通常需要源数据或源/目标联合训练。但 BCI 源 EEG 是敏感神经数据，可能受 GDPR/PIPL、医院伦理、用户撤权和机构隔离限制，无法持续访问。因此需要 SFDA。

### Q2：为什么不用 FL？

FL 适合训练阶段多客户端协作，减少原始数据集中上传。但我们的场景是模型已经训练好，用户之后撤权，且源客户端/源数据可能不再可用。FL 不能直接删除已训练模型中某个撤回用户的影响。

### Q3：为什么不用 DP？

DP 是训练时保护，必须提前加入噪声机制。它不能在训练后针对某个具体撤回用户执行删除；而 EEG 信号低信噪比、小样本，强 DP 噪声还可能显著损害解码性能。DP 可以作为补充 baseline，但不能替代 MU。

### Q4：MU 为什么要和 SFDA 结合？

MU 解决撤回用户影响，SFDA 解决目标用户个性化适配。单独 MU 不能让模型适配新用户；单独 SFDA 不能删除源模型残留隐私。两者结合才对应真实 BCI 场景。

### Q5：如果源数据和元数据都不能访问，怎么知道忘记谁？

至少需要撤回请求对应的 consent metadata，例如 subject ID、session ID、device/domain ID。若完全没有元数据，就无法做精确用户级 unlearning，只能做通用隐私表征擦除或隐私审计。因此我们的设定应明确：原始源 EEG 不可访问，但撤回对象的最小标识信息可用。

### Q6：这个工作的新意在哪里？

现有 EEG-SFDA 多数关注目标域 accuracy，不关心源模型是否泄露源用户信息；MU 研究多数在图像/语言，不处理 EEG 的跨用户非平稳和源数据不可访问；我们把两者放到 BCI 个性化和用户撤权场景下，联合评估 utility 和 privacy。

## 12. 参考支撑

### 法规/政策

- GDPR Article 17: right to erasure / right to be forgotten.
- PIPL: consent withdrawal, deletion rights, sensitive personal information protection.
- Colorado HB24-1058: expands sensitive data to include biological/neural data.
- California SB-1223: adds neural data to sensitive personal information.

### 神经隐私/BCI 风险

- Yuste R, et al. Four ethical priorities for neurotechnologies and AI. Nature, 2017.
- Yuste R. Advocating for neurodata privacy and neurotechnology regulation. Nature Protocols, 2023.
- Tang J, et al. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 2023.
- Martinovic I, et al. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces. USENIX WOOT, 2012.

### DA/个性化 BCI

- Ma J, et al. A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface. Scientific Data, 2022.
- Ding Y, et al. EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. Nature Communications, 2025.
- Dreyer P, et al. A large EEG database with users’ profile information for motor imagery brain-computer interface research. Scientific Data, 2023.
- Yang B, et al. A multi-day and high-quality EEG dataset for motor imagery brain-computer interface. Scientific Data, 2025.

