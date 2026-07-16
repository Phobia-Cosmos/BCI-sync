# SFDA、MU、DP 与 BCI 隐私保护的研究动机整理

> 目标：回答“为什么非 BCI 领域使用 SFDA / MU；为什么隐私保护下要考虑 SFDA+MU 而不是只用 DP；MU 如何处理 source/forget/retain performance；BCI 安全隐私可以引用哪些非计算机领域文献”。  
> 写作定位：这份材料可以直接作为论文 introduction、background、related work 和 method motivation 的中文草稿。

## 0. 核心判断

如果把论文定位为 **Source-Free Machine Unlearning for Privacy-Preserving EEG/BCI Domain Adaptation**，最稳的论证链是：

1. **EEG/BCI 需要 domain adaptation**：跨被试、跨会话、跨设备漂移显著，不做适配会导致实时系统错误输出。
2. **SFDA 解决“适配阶段不能访问源数据”**：源 EEG 原始数据因为隐私、医院/机构所有权、数据体量、法规限制不能给目标端。
3. **但 SFDA 不等于隐私保护完成**：源模型参数、输出分布、embedding 仍可能保留源用户身份、训练样本、敏感域或类别原型信息。
4. **MU 解决“模型里已经学到的指定数据影响如何删除”**：用户撤回授权、医院不再允许源数据影响模型、某些敏感用户/会话/域需要被删除时，单纯不访问源数据不够。
5. **DP 与 MU 不是替代关系**：DP 是训练时的全局概率保护；MU 是训练后的选择性删除。DP 更像“预防泄露”，MU 更像“事后删除/撤回授权/模型清理”。
6. **在 EEG 论文中，不建议把 forget unit 设计成任务类别**，例如“左手运动想象类”。更合理的是忘记 **subject / session / hospital / device / privacy attribute**，同时保留 task label 的解码性能。

一句话 gap：

> Existing EEG SFDA methods reduce raw-source-data exposure during target adaptation, but they do not explicitly remove the residual influence of sensitive source subjects or sessions encoded in the source model. Machine unlearning provides a selective post-training mechanism to sanitize the source model before or during source-free target adaptation.

## 1. 非 BCI 领域为什么使用 SFDA

### 1.1 共同背景

非 BCI 领域使用 Source-Free Domain Adaptation 的主要原因不是“更酷”，而是标准 UDA 的一个关键假设在真实部署中经常不成立：**目标端适配时拿不到源域原始数据**。

常见背景包括：

- **隐私与合规**：医院影像、用户照片、驾驶数据、企业数据不能被目标端复制。
- **数据所有权和商业限制**：源数据属于另一家公司或机构，最多只愿意交付训练好的模型或 API。
- **数据体量与传输成本**：源数据太大，跨机构传输不现实。
- **源模型/API 黑盒化**：目标端只能访问预测概率或 hard labels，无法访问源模型参数。
- **去中心化部署**：模型部署到边缘设备或客户环境，源机构不希望继续暴露数据。

### 1.2 代表性 SFDA 论文的“为什么做”

| 论文 | 领域背景 | 为什么使用 SFDA | 聚焦问题 |
| --- | --- | --- | --- |
| SHOT, ICML 2020 | 通用视觉 UDA | 传统 UDA 适配时要访问源数据，存在隐私和去中心化效率问题 | 只用源模型和无标签目标数据；冻结 classifier，用信息最大化和伪标签适配 feature extractor |
| DINE, CVPR 2022 | 黑盒/多源 DA | 目标端只有黑盒源预测器，源数据和模型参数都不可访问 | 从黑盒预测蒸馏目标模型，再用目标结构细调；支持单源、多源、partial-set |
| BETA, ICLR 2023 | 黑盒 predictor DA | 黑盒预测器在目标域上产生 noisy pseudo-label，直接蒸馏会累积 confirmation bias | 将目标域划分为 easy/hard subdomains，用互教网络过滤预测噪声 |
| SF(DA)^2, ICLR 2024 | 图像/点云/长尾 SFDA | 数据增强在 SFDA 中有价值，但真实增强依赖先验且增加算力/内存 | 在 feature space 构造隐式增强和图结构，降低增强成本 |
| Source-Free DA with Frozen Multimodal Foundation Model, CVPR 2024 | 多模态 foundation model | foundation model 可用但不能访问源数据，且大模型通常不希望被整体更新 | 冻结多模态模型，通过目标特征/文本知识做适配 |
| LEAD, CVPR 2024 | Universal DA | 目标域 label space 可能不同于源域，存在 unknown / private classes | 学习分解与对齐，处理 source-free universal adaptation |
| ProDe, ICLR 2025 | 通用 SFDA | SFDA 中 pseudo labels/proxies 噪声严重 | proxy denoising，减少目标伪标签误差 |

### 1.3 这些论文共同聚焦的问题

SFDA 不是单一方法，而是一组在 **无源数据约束** 下解决 domain shift 的技术。核心问题通常包括：

- **目标伪标签噪声**：目标数据无标签，模型初始预测不可靠。
- **confirmation bias**：错误伪标签被反复强化，目标模型越来越偏。
- **source hypothesis mismatch**：源分类器假设和目标特征分布不匹配。
- **label shift / open-set / universal DA**：目标类别空间可能少于、多于或不同于源类别空间。
- **black-box constraint**：只能访问 API 输出，不能访问源模型参数和源数据。
- **foundation model adaptation**：如何在不更新大模型或不访问源数据时做目标域适配。

对 EEG/BCI 的迁移方式：

> 非 BCI 的 SFDA 背景可以直接迁移到 EEG：医院/实验室 A 的源 EEG 不共享给医院/设备 B；目标端只有源模型和新用户无标签 EEG；需要在跨被试、跨会话、跨设备漂移下完成适配。

## 2. 为什么使用 MU

### 2.1 MU 的背景

Machine Unlearning 的核心背景是：模型训练后，某些训练数据不应继续影响模型。

常见动机：

- **Right to be forgotten / 撤回授权**：用户要求删除自己的训练数据影响。
- **隐私泄露风险**：模型可能通过 membership inference、model inversion 或输出置信度泄露训练数据。
- **版权数据删除**：大模型或生成模型中需要删除未经授权内容。
- **错误/有害数据修正**：删除 mislabeled、poisoned、backdoor、biased 或 toxic 数据影响。
- **源数据不可再访问**：无法用 retain data 从头重训，但又必须执行删除请求。

### 2.2 代表性 MU 论文的“为什么做”

| 论文 | 背景 | 为什么使用 MU | 聚焦问题 |
| --- | --- | --- | --- |
| Making AI Forget You, NeurIPS 2019 | 数据删除请求和模型更新成本 | 从算法层面研究如何让模型删除指定训练样本影响 | 数据删除机制和高效更新 |
| Certified Data Removal, ICML 2020 | 好的数据治理要求模型也响应删除请求 | 定义 certified removal，使删除后模型与从未见过该数据训练的模型不可区分 | 线性模型、理论保证、参数不可区分 |
| Machine Unlearning / SISA, IEEE S&P 2021 | 用户数据一旦进入模型，很难撤回 | 通过 sharding/slicing 限制单个数据点影响范围，降低重训成本 | exact/近似重训、工程化删除 |
| Remember What You Want to Forget, NeurIPS 2021 | 不同模型和统计设定下的遗忘理论 | 研究遗忘算法的样本复杂度和理论界 | 理论化 unlearning |
| Towards Unbounded MU, NeurIPS 2023 | 现有 MU 评测目标混乱 | 区分 bias removal、confusion correction、user privacy 等不同遗忘目标 | SCRUB、MIA-based privacy 评测、retain utility |
| TOFU, ICLR 2024 | LLM 中删除具体知识/事实困难 | 构造 LLM unlearning benchmark | 生成模型遗忘评测 |
| Selective Unlearning via Representation Erasure, ICLR 2025 | 深度模型中敏感信息可保留在 representation | 用 domain-adversarial 表征擦除做 selective unlearning | 表征层删除、保留任务性能 |
| Towards Source-Free Machine Unlearning, CVPR 2025 | 传统 MU 默认有 retain/source data，但真实场景可能没有 | 只用 trained model 和 forget data 估计 retain Hessian，实现 source-free unlearning | 无 retain data、理论保证、保留 remaining performance |
| Approximate Domain Unlearning for VLMs, NeurIPS 2025 | 视觉语言模型中可能需要删除某个域/风格/数据源影响 | 做 domain-level approximate unlearning | 域级遗忘与多模态模型 utility |

### 2.3 MU 论文真正关注什么

MU 不是简单地“把某类数据搞乱”。它通常同时优化两个目标：

1. **Forget quality**：待删除数据的影响要下降。  
   常用指标：forget accuracy 下降、MIA 接近随机、输出接近 retrained-without-forget model、参数/分布不可区分。

2. **Retain utility**：不该删的数据性能不能明显下降。  
   常用指标：retain accuracy、test accuracy、target-domain accuracy、calibration、泛化性能。

所以 MU 的技术难点是 **selective removal**：只删指定数据影响，不把整个模型能力毁掉。

## 3. SFDA + MU 的背景和问题是什么

### 3.1 为什么 SFDA 还不够

SFDA 的隐私价值是 **不共享源原始数据**。但这不是完整隐私保护，因为：

- 源模型可能记住训练样本统计特征。
- 源模型 embedding 可能保留身份、年龄、健康状态或设备/医院域信息。
- 黑盒输出的置信度、loss、预测一致性可能被用于 membership inference。
- 如果源用户撤回授权，源模型已经包含该用户影响；“以后不访问原始数据”不能删除已学到的信息。

因此，SFDA 的隐私边界是：

> SFDA protects source data access, but not necessarily source-model leakage.

### 3.2 为什么 MU 适合补上这个 gap

MU 处理的是训练后模型中的指定数据影响。放到 SFDA 中，可以形成如下流程：

```text
source EEG data -> source model
                    |
                    | user/session/domain deletion request
                    v
             source-free / selective MU
                    |
                    v
          sanitized source model
                    |
                    v
       source-free target adaptation on unlabeled target EEG
```

也就是说：

- **SFDA**：目标适配时不需要源 EEG。
- **MU**：源模型交付或目标适配前，删除指定源用户/会话/域影响。
- **CTTA**：部署后面对连续流式 EEG 漂移，持续适配但不回传原始数据。

### 3.3 EEG/BCI 论文中最合理的问题定义

建议不要写成“忘记某个任务类别”，而写成：

- forget unit：某个 source subject、source session、source hospital/device、或含敏感属性的子集。
- retain target：其他 source subjects/sessions 的任务解码能力。
- adaptation target：新目标用户无标签 EEG 的解码性能。
- privacy target：subject-id inference / membership inference / reconstruction attack 下降。

形式化可以写：

```text
Given a source model M_s trained on multi-subject EEG D_s,
and a deletion request D_f from a source subject/session/domain,
learn a sanitized model M_u without accessing D_r = D_s \ D_f,
then adapt M_u to unlabeled target EEG D_t.
```

目标函数可以概念化为：

```text
minimize:  target adaptation loss + retain-behavior regularization
maximize:  privacy leakage reduction / forget influence removal
constraint: no access to raw source retain data
```

## 4. 如果是隐私保护，为什么使用 MU 而不是差分隐私

### 4.1 DP 和 MU 解决的问题不同

| 维度 | Differential Privacy | Machine Unlearning |
| --- | --- | --- |
| 时间点 | 训练时或查询时预防 | 训练后删除/修正 |
| 目标 | 降低单个样本对输出分布的影响 | 删除指定样本/用户/类别/域的模型影响 |
| 粒度 | 通常是全体样本的统一保护 | 可指定某个 user/session/class/domain |
| 是否支持撤回授权 | 不直接等价；DP 不能让模型“忘掉某个已训练用户” | 正是核心目标 |
| 是否需要重训前规划 | 通常需要 DP-SGD 或 DP 机制提前设计 | 可以做 post-hoc unlearning |
| 对 utility 的影响 | EEG 小样本下噪声可能明显损伤性能 | 目标是选择性删除并保留 retain/target utility |
| 形式保证 | 有严格概率隐私定义 | 从 exact/certified 到 empirical unlearning 不等 |

### 4.2 为什么不能只用 DP

在你的 EEG+SFDA+MU 论文中，不能把 DP 说成没用。更准确的说法是：

- DP 是强隐私工具，但它通常需要在源模型训练时就使用。
- 如果源模型已经训练完，且没有 DP-SGD 训练记录，DP 不能直接删除某个用户影响。
- DP 提供的是全局概率隐私保护，不解决“指定用户撤回授权后模型仍包含该用户影响”的删除问题。
- EEG 数据小、噪声大、跨被试差异强；强 DP 噪声可能显著损害 BCI decoding utility。
- DP 不直接区分任务相关信息和身份隐私信息；MU/representation erasure 可以更自然地做选择性清理。

推荐写法：

> Differential privacy and machine unlearning address complementary privacy requirements. DP reduces information leakage during training by bounding the influence of any individual sample, whereas MU aims to remove the influence of specified data after training. In EEG-based BCI, where users may revoke consent and raw source EEG may be unavailable, MU provides a post-hoc and selective mechanism that DP alone cannot provide.

### 4.3 论文中如何使用 DP

最稳妥的实验设计是把 DP 作为 baseline 或 complementary module，而不是敌人：

- `Source-only`：正常源模型。
- `SFDA`：只做 source-free adaptation。
- `DP-SGD + SFDA`：源训练阶段加 DP，观察 utility/privacy trade-off。
- `MU + SFDA`：删除指定 source subject/session 后再做 SFDA。
- `DP + MU + SFDA`：可作为扩展，表示二者互补。

结论不要写“MU 比 DP 更强”，而写：

> MU better matches post-training deletion and consent revocation, while DP better matches training-time global privacy protection.

## 5. 使用 MU 后，source class / retain class 上的性能如何处理

这里要先区分两种不同的“class”。

### 5.1 如果 forget class 是任务类别

例如在图像分类中忘记 `truck` 类，或者在 EEG 中忘记 `left-hand MI` 类。

这种设定下：

- forget class accuracy 应该下降，甚至接近随机或被拒识。
- retain classes accuracy 应该尽量保持。
- 如果模型输出层仍保留 forget class，则通常希望该类不再被稳定预测。
- 如果系统不再支持该类，可以移除输出 neuron 或把该类映射为 unknown。

但在 BCI 中，这个设定通常不适合作为隐私保护主线。因为忘记任务类别会破坏 BCI 功能：用户仍然需要模型识别左手/右手/P300/SSVEP 等任务类别。

### 5.2 如果 forget unit 是用户/会话/域

这是 EEG 隐私保护更合理的设定。

此时 task classes 不应被删除，目标是：

- 删除某个 source subject/session 的身份或成员影响。
- 保持 task classifier 对 left/right/P300/SSVEP/emotion 等任务标签的能力。
- 在 retain source subjects 和 target subject 上保持 accuracy。
- 让攻击者难以判断该用户是否参与训练，或难以从 embedding 中识别其身份。

评测应拆成：

| 指标 | 期望变化 |
| --- | --- |
| Forget subject 的 membership inference AUC | 下降到接近随机 |
| Forget subject 的 identity classification accuracy | 下降 |
| Forget subject 的 reconstruction / inversion quality | 下降 |
| Retain subjects 的 task accuracy | 尽量保持 |
| Target subject 的 SFDA accuracy | 尽量保持或小幅下降 |
| Target adaptation stability / CTTA drift | 不应明显恶化 |

### 5.3 对论文最重要的表述

> For privacy-preserving EEG unlearning, the forgetting target should be a privacy-bearing unit such as a subject, session, hospital, or sensitive domain, rather than a task label. The model should forget who contributed the data while preserving what task-relevant neural patterns mean.

## 6. Forget data 对模型的影响如何解决

MU 的标准目标是让 unlearned model 接近：

```text
M_retrain = Train(D_s \ D_f)
```

但完整重训通常太贵或不可能，所以出现以下方法。

### 6.1 Exact / retraining-based

- 从头用 retain data 重训：最干净的 gold standard，但成本最高。
- SISA：提前把训练数据分 shard/slice，只重训受影响 shard，减少删除成本。

优点：概念清楚，最接近真正删除。  
缺点：需要提前设计和保存检查点；不适合已经训练好的大模型或 source-free 场景。

### 6.2 Influence / Hessian-based

核心思想：

```text
forget influence ≈ H_retain^{-1} * gradient_forget
```

然后对模型参数做反向修正。

传统问题：需要 retain data 计算 retain Hessian。  
Source-free MU 的关键创新：没有 retain data 时，尝试只用 trained model 和 forget data 估计 retain Hessian 或其近似。

### 6.3 Gradient ascent / negative gradient

对 forget data 最大化 loss：

```text
maximize CE(M(x_f), y_f)
```

这会快速破坏模型对 forget data 的记忆。  
问题：容易过度破坏模型，导致 retain/test accuracy 大幅下降。

### 6.4 Random labels

给 forget data 分配错误或随机标签，然后微调：

```text
minimize CE(M(x_f), random_label)
```

这比纯 gradient ascent 温和，但仍可能污染决策边界。

### 6.5 Distillation / retain regularization

在 retain data 或 surrogate data 上，让 unlearned model 保持原模型行为：

```text
KL(M_u(x_r) || M_s(x_r))
```

问题：如果没有 retain data，就需要 surrogate samples、生成样本、目标域样本或参数正则替代。

### 6.6 Representation erasure / adversarial removal

如果隐私目标是身份、医院、设备域，可以训练一个 adversarial head：

```text
feature z should predict task label,
but should not predict subject/domain identity.
```

这适合 EEG，因为 EEG embedding 同时包含任务相关成分和身份/设备/会话成分。

## 7. 为什么 forget class 要打标签

这取决于 unlearning 类型。

### 7.1 class-level unlearning

如果你说“忘记第 c 类”，那么必须知道哪些样本属于第 c 类：

```text
D_f = {(x_i, y_i): y_i = c}
```

标签用途：

- 定义要忘记的集合。
- 对 forget class 做 gradient ascent 或 random label。
- 衡量 forget class accuracy 是否下降。
- 判断 retain classes 是否仍保持性能。

所以 class-level MU 天然需要类别标签。

### 7.2 instance-level / user-level unlearning

如果你说“忘记用户 A 的数据”，不一定需要 task label，但需要 **forget identifier**：

```text
D_f = {(x_i, y_i, user_i): user_i = A}
```

这里的关键标签不是 task label，而是：

- subject ID
- session ID
- hospital/site ID
- device ID
- consent group
- sensitive attribute group

如果没有这些元数据，只能用聚类或 pseudo-label 推断 forget set，可信度会下降。

### 7.3 对 EEG 论文的建议

不要说“forget class = emotion class / MI class”。建议说：

> We define the forget set by privacy-bearing metadata, e.g., subject ID or session ID. Task labels are used to preserve decoding utility, while subject/session labels are used to evaluate and remove privacy leakage.

## 8. 传统 MU 对 forget class 如何处理

传统 class unlearning 通常有四类处理方式：

1. **Retrain without class**  
   直接删除该类样本后重训。forget class 不再是有效类别，模型应无法正确识别该类。

2. **Remove or mask output neuron**  
   如果忘记整个类别，可以移除对应输出头，或把该类归入 unknown/reject。

3. **Negative training / random labels**  
   用梯度上升或随机标签破坏模型对该类的决策边界。

4. **Retain distillation**  
   在 retain classes 上蒸馏原模型，防止其他类别性能塌陷。

但这些方法直接搬到 EEG 隐私保护会有风险：如果 forget class 是 task class，会牺牲 BCI 功能。因此 EEG privacy unlearning 更适合：

- subject-level unlearning
- session-level unlearning
- domain-level unlearning
- identity-feature erasure
- sensitive-attribute unlearning

## 9. EEG 中任务相关特征与身份隐私特征如何解耦

EEG 表征可以粗略拆成：

```text
z = z_task + z_identity + z_session + z_device + noise
```

任务相关特征：

- MI：mu/beta rhythm ERD/ERS、C3/C4 spatial pattern。
- SSVEP：刺激频率及谐波响应。
- P300：刺激后约 300ms 的 ERP 形态。
- emotion/fatigue：频带能量、额叶不对称性、时频图、连接性。

身份隐私相关特征：

- 个体稳定频谱模式。
- 头皮传导和电极接触差异。
- 个体脑结构/神经反应差异。
- session/device/hospital 采集偏差。
- 年龄、性别、健康状态、药物/睡眠等潜在属性。

可用方法：

- **domain-adversarial training**：task head 预测任务，subject head 通过 gradient reversal 被混淆。
- **mutual information minimization**：降低 feature 与 subject identity 的互信息。
- **orthogonal decomposition**：显式拆分 task subspace 和 identity subspace。
- **contrastive learning**：同一任务跨 subject 拉近，同一 subject 不同任务不作为正样本。
- **privacy attacker-guided training**：以 identity classifier / MIA attacker 作为隐私评估器。
- **unlearning after source training**：对指定 subject/session 做 post-hoc representation erasure 或 influence removal。

论文中可以写：

> EEG privacy protection should not erase task-discriminative neural patterns. The key is to suppress identity- or domain-specific components while preserving task-relevant components for BCI decoding.

## 10. 推荐实验设计

### 10.1 数据集与任务

优先选跨被试 EEG 数据：

- MI：BNCI2014001 / BCI Competition IV 2a。
- Emotion：SEED / DEAP。
- SSVEP：Benchmark SSVEP。
- Sleep staging：Sleep-EDF 或跨被试 sleep EEG。

### 10.2 Split

```text
source subjects: S_1 ... S_n
forget subject/session: S_f
retain source subjects: S_r
target subject: S_t
```

### 10.3 Baselines

- Source-only。
- UDA with source data，作为非隐私上界。
- SFDA only。
- DP-SGD + SFDA。
- MU only。
- MU + SFDA。
- Oracle retrain without forget subject + SFDA，作为上界。

### 10.4 Privacy attacks

- Membership inference：判断某个 subject/session 是否参与源模型训练。
- Subject identity inference：从 embedding 预测 subject ID。
- Attribute inference：预测性别/年龄组/设备/医院等属性，若数据集有元数据。
- Reconstruction / inversion：如果任务相关，可测试从 feature 或输出反推 EEG pattern 的质量。

### 10.5 Utility metrics

- Target task accuracy / balanced accuracy / F1。
- Retain subjects task accuracy。
- Calibration / confidence。
- CTTA streaming stability：连续 session 上是否灾难性漂移。
- Latency / memory：实时 BCI 部署可加。

## 11. 非计算机领域 BCI 安全隐私文献

下面这些文献适合支撑 introduction 中“BCI 隐私、安全、伦理不是纯计算机问题，而是医学、神经伦理和社会治理问题”的论证。

| 文献 | 期刊/类型 | 可支撑观点 |
| --- | --- | --- |
| Rafael Yuste et al. 2017. Four ethical priorities for neurotechnologies and AI. Nature 551, 159–163. https://doi.org/10.1038/551159a | Nature comment | 神经技术和 AI 应保护 privacy、identity、agency、equality |
| Liam Drew. 2019. The ethics of brain–computer interfaces. Nature 571, S19–S21. https://doi.org/10.1038/d41586-019-02214-2 | Nature Outlook | BCI 越复杂，伦理问题越复杂；临床神经设备会影响用户体验和身份感 |
| Marcello Ienca and Pim Haselager. 2016. Hacking the brain: brain–computer interfacing technology and the ethics of neurosecurity. Ethics and Information Technology 18, 117–129. https://doi.org/10.1007/s10676-016-9398-9 | Ethics and Information Technology | BCI 可能产生 brain-hacking / neurocrime 风险，涉及神经信息非法访问和操纵 |
| Tamara Denning, Yoky Matsuoka, and Tadayoshi Kohno. 2009. Neurosecurity: security and privacy for neural devices. Neurosurgical Focus 27(1), E7. | Neurosurgical Focus | 神经设备安全和隐私可进入医疗设备安全层面 |
| Pim Haselager, Rutger Vlek, Jeremy Hill, and Femke Nijboer. 2009. A note on ethical aspects of BCI. Neural Networks 22(9), 1352–1357. | Neural Networks | BCI 的责任、期望管理、用户自主性和伦理边界 |
| Femke Nijboer, Jens Clausen, Brendan Z. Allison, and Pim Haselager. 2013. The Asilomar Survey: Stakeholders' Opinions on Ethical Issues Related to Brain-Computer Interfacing. Neuroethics 6, 541–578. | Neuroethics | BCI 社区对伦理、安全、使用边界的 stakeholder 观点 |
| Steffen Steinert and Orsolya Friedrich. 2020. Wired Emotions: Ethical Issues of Affective Brain–Computer Interfaces. Science and Engineering Ethics 26, 351–367. | Science and Engineering Ethics | 情绪识别/情感 BCI 的伦理风险，适合支撑 emotion BCI 隐私问题 |
| Mark A. Attiah and Martha J. Farah. 2014. Minds, motherboards, and money: futurism and realism in the neuroethics of BCI technologies. Frontiers in Systems Neuroscience 8, 86. | Frontiers in Systems Neuroscience | BCI 技术现实能力与未来风险之间的神经伦理分析 |
| Baraka Maiseli et al. 2023. Brain–computer interface: trend, challenges, and threats. Brain Informatics 10, 20. https://doi.org/10.1186/s40708-023-00199-3 | Brain Informatics review | BCI 应用增长迅速，privacy/security 是商业化和社会接受的关键威胁 |
| R. A. Miranda et al. 2015. DARPA-funded efforts in the development of novel brain–computer interface technologies. Journal of Neuroscience Methods 244, 52–67. | Journal of Neuroscience Methods | BCI 与神经技术发展背景，可支撑高风险应用和双重用途 |
| Ienca and Andorno. 2017. Towards new human rights in the age of neuroscience and neurotechnology. Life Sciences, Society and Policy 13, 5. | Life Sciences, Society and Policy | neurorights / mental privacy / cognitive liberty 的法伦理背景 |
| Jens Clausen. 2009. Man, machine and in between. Nature 457, 1080–1081. | Nature | 脑机/神经接口的人机边界、责任和伦理问题 |

## 12. 可直接放到论文中的背景段落

### 12.1 SFDA 动机段

传统无监督域适配通常假设目标端在适配过程中可以访问源域有标签数据。然而，在医疗影像、自动驾驶、遥感和 BCI 等真实部署场景中，源数据往往受到隐私法规、机构所有权、传输成本或商业限制约束，无法被目标端直接访问。因此，SFDA 研究只依赖训练好的源模型和无标签目标数据完成适配。现有 SFDA 工作主要关注目标伪标签噪声、confirmation bias、黑盒源模型、label space mismatch 和 foundation model adaptation 等问题。

### 12.2 MU 动机段

Machine unlearning 关注训练后删除指定数据对模型的影响，其背景来自数据删除请求、撤回授权、版权数据移除、错误或有害训练数据修正以及隐私攻击风险。与重新训练相比，MU 希望以更低成本得到接近 “trained without the forget data” 的模型，同时保持 retain data 上的性能。近年来，MU 从精确重训、SISA、certified removal、influence/Hessian 更新发展到 representation erasure 和 source-free unlearning。

### 12.3 SFDA+MU for EEG 段

在 EEG/BCI 中，SFDA 可以避免目标适配阶段访问源用户原始脑电，但它不能保证源模型本身不泄露源用户信息。EEG embedding 可能同时包含任务相关模式和身份、会话、设备或健康相关特征。如果某个源用户撤回授权，或源机构要求删除某些敏感会话影响，仅仅停止共享原始 EEG 并不能清除模型中已经学习到的信息。因此，本文考虑在 source-free EEG adaptation 中引入 machine unlearning，在不访问 retain source EEG 的条件下删除指定 subject/session/domain 的影响，并随后对目标用户进行无源适配。

### 12.4 DP 对比段

差分隐私和机器遗忘解决的是互补问题。DP 在训练时通过随机化机制限制任一样本对模型输出分布的影响，适合全局隐私保护；MU 则在训练后对指定数据执行选择性删除，适合撤回授权和模型清理。在 EEG 场景中，源模型可能已经训练完成且源数据不可再访问，此时 DP 无法直接删除指定用户的历史影响；同时，强 DP 噪声可能损害小样本、低信噪比 EEG 解码性能。因此，MU 更适合作为 post-hoc selective privacy mechanism，而 DP 可以作为 complementary baseline。

## 13. 推荐引用清单

### SFDA / Black-Box DA

- Jian Liang, Dapeng Hu, and Jiashi Feng. 2020. Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. ICML.
- Jian Liang, Dapeng Hu, Jiashi Feng, and Ran He. 2022. DINE: Domain Adaptation from Single and Multiple Black-Box Predictors. CVPR.
- Jianfei Yang et al. 2023. Divide to Adapt: Mitigating Confirmation Bias for Domain Adaptation of Black-Box Predictors. ICLR.
- Uiwon Hwang et al. 2024. SF(DA)^2: Source-Free Domain Adaptation Through the Lens of Data Augmentation. ICLR.
- Song Tang et al. 2024. Source-Free Domain Adaptation with Frozen Multimodal Foundation Model. CVPR.
- Sanqing Qu et al. 2024. LEAD: Learning Decomposition for Source-Free Universal Domain Adaptation. CVPR.
- Song Tang et al. 2025. Proxy Denoising for Source-Free Domain Adaptation. ICLR.

### MU / Source-Free MU

- Antonio Ginart et al. 2019. Making AI Forget You: Data Deletion in Machine Learning. NeurIPS.
- Chuan Guo et al. 2020. Certified Data Removal from Machine Learning Models. ICML.
- Lucas Bourtoule et al. 2021. Machine Unlearning. IEEE S&P.
- Ayush Sekhari et al. 2021. Remember What You Want to Forget: Algorithms for Machine Unlearning. NeurIPS.
- Muhammad Kurmanji et al. 2023. Towards Unbounded Machine Unlearning. NeurIPS.
- Pratyush Maini et al. 2024. TOFU: A Task of Fictitious Unlearning for LLMs. ICLR.
- Nazanin Mohammadi Sepahvand et al. 2025. Selective Unlearning via Representation Erasure Using Domain Adversarial Training. ICLR.
- Sk Miraj Ahmed et al. 2025. Towards Source-Free Machine Unlearning. CVPR.
- Kodai Kawamura et al. 2025. Approximate Domain Unlearning for Vision-Language Models. NeurIPS.

### DP / Privacy Attacks

- Reza Shokri et al. 2017. Membership Inference Attacks Against Machine Learning Models. IEEE S&P.
- Matt Fredrikson et al. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. CCS.
- Nicholas Carlini et al. 2022. Membership Inference Attacks from First Principles. IEEE S&P.
- Thomas Steinke, Milad Nasr, and Matthew Jagielski. 2023. Privacy Auditing with One (1) Training Run. NeurIPS.
- Raef Bassily, Corinna Cortes, Anqi Mao, and Mehryar Mohri. 2024. Differentially Private Domain Adaptation with Theoretical Guarantees. ICML.

### EEG / BCI Privacy and Ethics

- Ivan Martinovic et al. 2012. On the Feasibility of Side-Channel Attacks with Brain-Computer Interfaces. WOOT.
- Mario Frank et al. 2017. Subliminal Probing for Private Information via EEG-Based BCI Devices. PETS.
- Sergio López Bernal et al. 2021. Security in Brain-Computer Interfaces: State-of-the-Art, Opportunities, and Future Challenges. ACM Computing Surveys.
- Lubin Meng et al. 2024. Protecting Multiple Types of Privacy Simultaneously in EEG-Based Brain-Computer Interfaces. IEEE SMC.
- Zhibo Tian et al. 2025. BrainGuard: Privacy-Preserving Multisubject Image Reconstructions from Brain Activities. AAAI.
- Huabin Wang et al. 2025. ID-RemovalNet: Identity Removal Network for EEG Privacy Protection with Enhancing Decoding Tasks. IJCAI.
- Rafael Yuste et al. 2017. Four ethical priorities for neurotechnologies and AI. Nature.
- Marcello Ienca and Pim Haselager. 2016. Hacking the brain: brain–computer interfacing technology and the ethics of neurosecurity. Ethics and Information Technology.
- Baraka Maiseli et al. 2023. Brain–computer interface: trend, challenges, and threats. Brain Informatics.

## 14. 最终写作建议

论文主线应避免两个弱点：

1. 不要声称 “SFDA = privacy-preserving”。更严谨地说，SFDA reduces raw source data exposure。
2. 不要声称 “MU guarantees privacy”。除非采用 certified / exact unlearning，否则只能说 empirical privacy leakage mitigation。

最稳标题方向：

- **Source-Free and Forgettable EEG Adaptation for Privacy-Preserving Brain-Computer Interfaces**
- **Privacy-Aware Source-Free EEG Adaptation via User-Level Machine Unlearning**
- **Forgetting Source Users in Source-Free EEG Domain Adaptation for Brain-Computer Interfaces**

最稳贡献写法：

1. 提出 EEG SFDA 中的 residual source-model privacy leakage 问题。
2. 将 user/session/domain-level MU 引入 source-free EEG adaptation。
3. 设计 utility/privacy 双目标评测：target decoding performance + membership/identity leakage。
4. 讨论 DP 与 MU 的互补性，并用 DP-SGD 或 DP adaptation 作为 baseline。

## 15. 非 BCI 领域使用 SFDA 的领域化背景

这一节可以用于 related work 的第一段：说明 SFDA 不是 EEG 特有问题，而是多领域都面临的“源数据不可用 + 目标域漂移”问题。

| 领域 | 源数据为什么不可用 | 目标域漂移是什么 | SFDA 聚焦问题 | EEG/BCI 可借鉴点 |
| --- | --- | --- | --- | --- |
| 医学影像 | 患者隐私、医院数据孤岛、法规限制 | 不同医院、扫描仪、协议、病人群体 | 无源数据跨医院适配、伪标签噪声、模型校准 | 源 EEG 来自医院/实验室，目标端只能拿模型 |
| 自动驾驶 | 真实路测数据体量大且含个人/地理隐私 | 天气、城市、传感器、时间、车辆平台 | 语义分割/检测在新城市新天气适配 | BCI 跨设备、跨场景、跨日漂移类似 |
| 遥感 | 原始图像巨大且可能涉敏感地理区域 | 地区、季节、传感器、分辨率 | 大规模无源目标适配、开放类别 | EEG 不同采集设备/通道 montage |
| 工业检测 | 企业生产数据不可共享 | 产线、批次、设备老化、缺陷类型变化 | 小样本目标适配、异常/未知类 | 医疗 BCI 设备老化、电极阻抗变化 |
| 多媒体/推荐 | 用户行为数据和图像视频受隐私限制 | 平台、用户群体、时间趋势 | 黑盒模型/API 适配 | 商业 BCI 平台可能只开放 API |
| Foundation model | 大模型参数和预训练数据不可访问 | 下游目标域特定分布 | 冻结 backbone、prompt/proxy adaptation | EEG foundation model 只作为源模型交付 |

对论文写作的启发：

> SFDA 的“source-free”不是单纯技术设定，而是现实约束：源数据的隐私、合规、所有权和传输成本使得目标端只能拿到模型或预测器。EEG/BCI 是这一设定的强案例，因为源数据同时具有医学敏感性和个体可识别性。

## 16. MU 的任务类型：不要把所有 unlearning 混为一谈

MU 论文里“忘记”的对象并不总是同一种东西。写论文时必须说清楚 forget unit。

| MU 类型 | Forget unit | 典型例子 | 期望 forget 行为 | 期望 retain 行为 | 是否适合 EEG 隐私 |
| --- | --- | --- | --- | --- | --- |
| Sample-level | 单个训练样本 | 删除某张图、某条记录 | 该样本 MIA 接近非成员 | 总体 accuracy 不变 | 可用于删除某些 EEG trials，但意义较弱 |
| User-level | 某个用户全部数据 | 删除某用户贡献 | 无法判断该用户是否训练过 | 其他用户和目标用户性能保持 | 很适合 |
| Session-level | 某次采集会话 | 删除某天/某次实验 | 会话特征不再被模型记忆 | 同用户其他会话/其他用户性能保持 | 很适合 |
| Class-level | 某个类别 | 删除 `truck` 类 | 该类 accuracy 下降或被拒识 | 其他类 accuracy 保持 | EEG 隐私中不建议作为主设定 |
| Domain-level | 某个域/医院/设备 | 删除某医院或设备域 | 域特征不可识别 | 其他域和目标域性能保持 | 很适合 |
| Attribute-level | 某敏感属性 | 删除性别/年龄/疾病属性 | 属性预测下降 | 任务预测保持 | 很适合，但需要元数据 |

对 EEG/BCI 的关键选择：

```text
推荐 forget unit: subject / session / hospital / device / sensitive attribute
不推荐主设定: task class
```

原因：

- BCI 的任务类别通常是系统功能本身，例如运动想象、P300、SSVEP 或情绪状态。
- 如果删除任务类别，模型功能会缺失，不符合隐私保护目标。
- 隐私保护更应删除“谁贡献了数据”或“数据来自哪个敏感域”，而不是删除“任务语义”。

## 17. Source-Free MU + SFDA 的具体算法框架

### 17.1 问题定义

设源 EEG 数据为：

```text
D_s = {(x_i, y_i, u_i, e_i)}
```

其中：

- `x_i`：EEG epoch / trial。
- `y_i`：任务标签，例如 MI class、P300 target/non-target、emotion class。
- `u_i`：subject ID。
- `e_i`：session/device/hospital/domain metadata。

训练好的源模型：

```text
M_s = C_s(G_s(x))
```

删除请求：

```text
D_f = {(x_i, y_i, u_i, e_i): u_i = u_f}
```

source-free unlearning 约束：

```text
available: M_s, D_f, D_t
unavailable: D_r = D_s \ D_f
```

目标：

```text
M_u ≈ Train(D_s \ D_f)
M_t = SFDA(M_u, D_t)
```

### 17.2 三阶段流程

```text
Stage 1: Source training
  Train M_s on multi-subject source EEG D_s.

Stage 2: Source-free machine unlearning
  Given deletion request D_f, sanitize M_s -> M_u.
  D_r is unavailable.

Stage 3: Source-free / continual target adaptation
  Adapt M_u to unlabeled target EEG stream D_t.
```

### 17.3 可写成论文算法的伪代码

```text
Algorithm: Forget-and-Adapt for EEG SFDA

Input:
  Source model M_s = G_s + C_s
  Forget set D_f
  Unlabeled target EEG stream D_t
  Privacy attacker A_id or A_mia for evaluation

Output:
  Target-adapted sanitized model M_t

1. Initialize M_u <- M_s
2. For each unlearning step:
     a. Compute forget loss L_f on D_f
     b. Remove forget influence using one of:
          - negative gradient
          - random-label training
          - influence/Hessian update
          - representation erasure
     c. Preserve non-forget behavior using:
          - parameter regularization to M_s
          - surrogate samples / target samples
          - distillation on high-confidence target samples
3. Initialize M_t <- M_u
4. For each target adaptation step:
     a. Predict target pseudo-labels
     b. Minimize entropy or information-maximization loss
     c. Maintain diversity / avoid collapse
     d. Optionally update BN/statistics in CTTA manner
5. Evaluate:
     task utility on target
     retain utility on non-forget subjects
     privacy leakage on forget subject/session
```

### 17.4 可写的整体损失

```text
L_total = L_unlearn + λ_r L_retain_proxy + λ_t L_target_sfda + λ_p L_privacy
```

其中：

```text
L_unlearn:
  negative CE on D_f, random-label CE, or influence/Hessian correction

L_retain_proxy:
  parameter regularization, Fisher/Hessian regularization, or distillation

L_target_sfda:
  entropy minimization + diversity + pseudo-label CE

L_privacy:
  identity confusion / adversarial subject classifier / MIA risk penalty
```

关键写作点：

- `L_unlearn` 负责删除 forget influence。
- `L_retain_proxy` 防止模型崩坏。
- `L_target_sfda` 保证目标用户可用。
- `L_privacy` 用来减少身份/成员泄露。

## 18. 不同算法路线的优缺点

| 路线 | 是否需要 D_f | 是否需要 D_r | 优点 | 缺点 | 适合写成主方法吗 |
| --- | --- | --- | --- | --- | --- |
| Oracle retrain | 是 | 是 | gold standard，最可信 | 不 source-free，成本高 | 只做上界 |
| SISA | 是 | 局部 shard | 删除成本低，接近 exact | 训练前必须设计，模型多 | 可做背景 |
| Negative gradient | 是 | 否 | 简单，适合 baseline | 容易毁掉 retain utility | baseline |
| Random labels | 是 | 否 | 简单，forget class 常用 | 污染边界，缺理论 | baseline |
| Distillation | 是/否 | 通常需要 surrogate | 保留 utility 好 | surrogate 选择困难 | 可组合 |
| Hessian / influence | 是 | 传统需要，source-free 估计可不要 | 理论清楚，和 CVPR 2025 Source-Free MU 对齐 | 深度非凸模型近似强 | 适合主方法 |
| Representation erasure | 需要 privacy labels | 不一定 | 适合 subject/domain 隐私 | 需要 identity/domain metadata | 很适合 EEG |
| DP-SGD | 否 | 训练时全数据 | 有严格 DP 保证 | 不是 post-hoc deletion，utility 可能降 | baseline / complementary |

建议方法路线：

```text
主方法 = source-free influence/Hessian-style unlearning + identity representation erasure + SHOT-style SFDA
baseline = SFDA only, NegGrad+SFDA, RandomLabel+SFDA, DP-SGD+SFDA, OracleRetrain+SFDA
```

## 19. 评测矩阵：如何证明“忘掉了”且“还能用”

### 19.1 Utility metrics

| 指标 | 数据 | 说明 |
| --- | --- | --- |
| Target accuracy / balanced accuracy | `D_t` | 目标用户 BCI 解码是否可用 |
| Retain source accuracy | `D_r`，只在评测端可用 | 其他源用户性能是否保持 |
| Forget task accuracy | `D_f` | 如果 forget 是 user/session，不一定要求任务 accuracy 下降 |
| Calibration / ECE | `D_t` | 实时设备输出置信度是否可信 |
| CTTA stability | target stream | 连续适配是否漂移或崩塌 |
| Latency / memory | device side | 实时 BCI 是否可部署 |

### 19.2 Privacy metrics

| 指标 | 攻击目标 | 期望 |
| --- | --- | --- |
| Membership inference AUC | 判断 forget subject/session 是否在训练集中 | 接近 0.5 |
| Subject identity accuracy | 从 embedding 预测 subject ID | forget subject 下降；整体身份可分性下降 |
| Attribute inference accuracy | 年龄/性别/疾病/设备/医院 | 下降 |
| Model inversion / reconstruction | 从输出或 embedding 重建 EEG pattern | 质量下降 |
| Distance to retrained model | 输出分布或参数差异 | 接近 Oracle retrain |

### 19.3 关键解释：forget subject 的 task accuracy 是否应该下降

如果 forget unit 是 subject/session，**不一定要求 forget subject 的 task accuracy 降低**。原因：

- 隐私目标是删除该用户对模型的训练影响，而不是让模型故意不能识别该用户的任务。
- 如果该用户作为未来 target user 重新使用系统，模型仍应能通过无标签适配恢复任务性能。
- 更合理的目标是：成员关系、身份特征、会话特征不可被攻击者稳定识别。

因此论文中应写：

> In user-level EEG unlearning, forgetting does not necessarily mean misclassifying the user's BCI task labels. Instead, it means removing the user's contribution as training evidence while preserving general task semantics.

## 20. 为什么“forget class 要打标签”这个问题容易混淆

很多 MU 论文使用 CIFAR-10 / ImageNet，所以它们说 “forget class” 时通常指视觉类别，例如 airplane、truck、dog。这导致一个误解：MU 必须要 class label。

实际上：

```text
MU 需要的是 forget-set definition，不一定是 task class label。
```

三种情况：

1. **Class unlearning**  
   forget set 由 task class label 定义，所以需要 `y=c`。

2. **User unlearning**  
   forget set 由 user ID 定义，需要 `u=u_f`，task label `y` 只用于保持 utility。

3. **Domain unlearning**  
   forget set 由 domain ID 定义，需要 `e=e_f`，例如医院/设备/会话。

对 EEG 来说：

- `y` 是任务标签，应被保留。
- `u/e/a` 是隐私标签，应被去除或混淆。

可以画成论文图：

```text
EEG x
 ├── task factor y: should remain
 ├── identity factor u: should be forgotten
 ├── session/device factor e: should be adapted or removed
 └── noise/artifacts: should be suppressed
```

## 21. 和 CTTA 的结合点

用户之前提出了 CTTA，所以这里补上可用逻辑。

### 21.1 为什么 EEG 需要 CTTA

EEG 实时设备不是一次性 batch adaptation，而是持续流：

- 电极阻抗随时间变化。
- 用户疲劳、注意、情绪变化。
- session 内和 session 间都有 non-stationarity。
- 真实部署不能频繁要求用户重新标注。

所以 CTTA 适合放在 deployment stage：

```text
source-free MU 先净化源模型
SFDA 初始化目标用户
CTTA 在实时流上持续更新统计量或少量参数
```

### 21.2 CTTA 的隐私风险

CTTA 也可能带来新风险：

- 持续更新可能把目标用户敏感特征写进模型。
- 如果模型回传到云端，会形成新的隐私泄露通道。
- 连续伪标签错误可能造成 drift 和安全风险。

所以可以提出：

> Privacy-preserving CTTA should include update constraints, replay-free adaptation, confidence gating, and periodic unlearning or model reset.

## 22. 可直接写入论文的 Related Work 结构

建议 Related Work 分四段：

### 22.1 Source-Free Domain Adaptation

写法：

> Source-free domain adaptation addresses domain shifts when source data are inaccessible during adaptation. Existing methods such as SHOT, DINE, BETA, SF(DA)^2 and ProDe focus on pseudo-labeling, entropy/information maximization, black-box predictors, confirmation bias and proxy denoising. These works motivate our use of SFDA for EEG, where source EEG cannot be shared across users or institutions.

### 22.2 Machine Unlearning

写法：

> Machine unlearning aims to remove the influence of specified training data from a trained model. Prior work ranges from exact retraining and SISA to certified removal, influence-based updates, representation erasure and source-free unlearning. However, existing MU studies are mostly evaluated on vision or language models, while privacy-preserving EEG adaptation requires user/session-level forgetting under source-free constraints.

### 22.3 Privacy in EEG/BCI

写法：

> EEG signals can encode not only task-related neural responses but also user identity, familiar information, affective states and health-related attributes. Prior studies on BCI side channels, subliminal probing, EEG identity protection and neuroethics show that neural data leakage has privacy, safety and autonomy implications.

### 22.4 Differential Privacy and Privacy Attacks

写法：

> Differential privacy provides training-time protection by bounding the contribution of individual samples, while privacy attacks such as membership inference and model inversion reveal that trained models may leak information about their training data. Our work is orthogonal to DP: we target post-training deletion requests and residual source-model leakage in source-free EEG adaptation.

## 23. 可直接写入论文的 Introduction 大纲

### 第一段：BCI 价值与实时部署

BCI has shown promise in neurorehabilitation, assistive control, communication and affective computing. In real-time BCI systems, EEG decoding models must work across users, sessions and devices, but EEG signals are highly non-stationary and user-specific.

### 第二段：domain shift 与 SFDA

Traditional cross-subject adaptation assumes access to labeled source EEG. This assumption is unrealistic when EEG is collected by hospitals, laboratories or commercial devices under privacy and ownership constraints. SFDA provides a practical framework by adapting a source model to unlabeled target EEG without accessing raw source data.

### 第三段：SFDA 的隐私缺口

However, removing source data access does not remove source information from the model. The source model may encode subject identity, session-specific patterns or membership signals. This is particularly concerning for EEG, whose representations can reveal sensitive neural and health-related attributes.

### 第四段：MU 的必要性

Machine unlearning offers a post-training mechanism for removing the influence of specified data. In BCI, this corresponds to deletion requests from source users, sessions or institutions. Combining MU with SFDA allows a model to be sanitized before adapting to target EEG.

### 第五段：本文贡献

贡献可以写：

1. We formulate privacy-aware EEG source-free adaptation with user/session-level unlearning.
2. We propose a forget-and-adapt framework that removes source user influence without accessing retain source EEG.
3. We evaluate both decoding utility and privacy leakage under membership and identity inference attacks.
4. We analyze the complementarity between MU and DP in privacy-preserving BCI.

## 24. ACM Reference Format：非计算机/医学/神经伦理文献

下面条目偏 ACM Reference Format，正式投稿前需要用官方 BibTeX 补全页码、issue、DOI。

[NCS-01] Rafael Yuste, Sara Goering, Blaise Agüera y Arcas, Guoqiang Bi, Jose M. Carmena, Adrian Carter, Joseph J. Fins, Phoebe Friesen, Jack Gallant, Jane E. Huggins, Philipp Kellmeyer, Eran Klein, Torsten O. Kringe, Christine Mitchell, Partha Mitra, David B. Ozar, Graeme Rainey, Erich Schadt, and Mackenzie M. Specker Sullivan. 2017. Four ethical priorities for neurotechnologies and AI. *Nature* 551, 7679 (2017), 159–163.

[NCS-02] Liam Drew. 2019. The ethics of brain–computer interfaces. *Nature* 571 (2019), S19–S21.

[NCS-03] Marcello Ienca and Pim Haselager. 2016. Hacking the brain: brain–computer interfacing technology and the ethics of neurosecurity. *Ethics and Information Technology* 18 (2016), 117–129.

[NCS-04] Tamara Denning, Yoky Matsuoka, and Tadayoshi Kohno. 2009. Neurosecurity: security and privacy for neural devices. *Neurosurgical Focus* 27, 1 (2009), E7.

[NCS-05] Pim Haselager, Rutger Vlek, Jeremy Hill, and Femke Nijboer. 2009. A note on ethical aspects of BCI. *Neural Networks* 22, 9 (2009), 1352–1357.

[NCS-06] Femke Nijboer, Jens Clausen, Brendan Z. Allison, and Pim Haselager. 2013. The Asilomar Survey: Stakeholders' Opinions on Ethical Issues Related to Brain-Computer Interfacing. *Neuroethics* 6 (2013), 541–578.

[NCS-07] Steffen Steinert and Orsolya Friedrich. 2020. Wired Emotions: Ethical Issues of Affective Brain–Computer Interfaces. *Science and Engineering Ethics* 26 (2020), 351–367.

[NCS-08] Mark A. Attiah and Martha J. Farah. 2014. Minds, motherboards, and money: futurism and realism in the neuroethics of BCI technologies. *Frontiers in Systems Neuroscience* 8 (2014), Article 86.

[NCS-09] Baraka Maiseli, Abdussalam A. Abdalla, Luqman C. Massawe, et al. 2023. Brain–computer interface: trend, challenges, and threats. *Brain Informatics* 10 (2023), Article 20.

[NCS-10] Marcello Ienca and Roberto Andorno. 2017. Towards new human rights in the age of neuroscience and neurotechnology. *Life Sciences, Society and Policy* 13 (2017), Article 5.

[NCS-11] Jens Clausen. 2009. Man, machine and in between. *Nature* 457 (2009), 1080–1081.

[NCS-12] Ricardo Chavarriaga, Melanie Fried-Oken, Sarang Shaikhouni, and José del R. Millán. 2017. Heading for new shores! Overcoming pitfalls in BCI design. *Brain-Computer Interfaces* 4, 1–2 (2017), 60–73.

## 25. 最后给论文定位的强论点

如果只能保留一个最强论点，建议写：

> SFDA is necessary but insufficient for privacy-preserving EEG adaptation. It removes the need to transfer raw source EEG, but the source model can still encode sensitive source-user information. Machine unlearning complements SFDA by providing a post-training mechanism to remove the influence of specified users, sessions, or domains before target adaptation.

如果审稿人问“为什么不是 DP”，回答：

> DP protects training globally and prospectively; MU deletes specified influence post hoc. In BCI, user consent can change after training, and raw source EEG may no longer be accessible. Therefore, MU addresses a privacy requirement that DP alone does not directly satisfy.

如果审稿人问“忘记后 source class 性能怎么办”，回答：

> In EEG privacy unlearning, we do not aim to forget task classes. We aim to forget privacy-bearing units such as subjects or sessions. Thus, task decoding performance on retain and target users should be preserved, while membership/identity leakage for the forgotten user should be reduced.

## 26. 逐篇 SFDA 论文动机卡片

这一节回答“其他领域为什么使用 SFDA，他们的背景和聚焦问题是什么”。可以直接拆到 related work。

### 26.1 SHOT, ICML 2020

- **背景**：传统 UDA 假设源数据和目标数据可以同时访问。但真实场景中，源数据可能因为隐私、数据传输、存储和商业原因无法访问。
- **为什么 SFDA**：只利用训练好的 source model 和 unlabeled target data，避免目标适配时访问 source raw data。
- **聚焦问题**：如何在没有 source data 的情况下保留 source classifier 的决策结构，并把 target feature 对齐到该结构。
- **核心思想**：冻结 classifier，优化 target feature extractor；使用 information maximization 和 pseudo-labeling。
- **对 EEG 的启发**：源实验室/医院只交付训练好的 EEG decoder，目标用户端通过无标签 EEG 做 adaptation。

可写句子：

> SHOT established a practical source-free setting where source data are unavailable during adaptation, motivating the use of a fixed source hypothesis and target-side self-training.

### 26.2 DINE, CVPR 2022

- **背景**：更强约束下，目标端甚至拿不到源模型参数，只能访问一个或多个 black-box predictors。
- **为什么 SFDA/Black-box DA**：商业 API、医疗模型服务或第三方平台通常不会开放模型权重和源数据。
- **聚焦问题**：如何从黑盒 predictor 输出中学习目标模型；如何处理 single-source 和 multi-source。
- **核心思想**：从黑盒输出蒸馏目标模型，再利用目标数据结构优化。
- **对 EEG 的启发**：商业 BCI 或医院模型可能只开放推理 API；目标端只能拿 prediction confidence 或 labels。

可写句子：

> Black-box SFDA further relaxes source access by assuming that only predictions from the source model are available, which matches third-party medical or commercial BCI services.

### 26.3 BETA, ICLR 2023

- **背景**：黑盒预测器在目标域产生 noisy pseudo-labels，直接自训练会导致 confirmation bias。
- **为什么 SFDA**：source data/model parameters 不可用，只能根据 target data 和 predictor outputs 自我修正。
- **聚焦问题**：伪标签错误如何避免被反复强化。
- **核心思想**：把目标样本分成 easy/hard 子域，用互学习/双分支缓解错误标签传播。
- **对 EEG 的启发**：跨被试 EEG 初始伪标签很容易错，尤其是低信噪比 trial；需要 confidence gating 或 easy-to-hard adaptation。

可写句子：

> In EEG SFDA, confirmation bias is particularly harmful because noisy target pseudo-labels can quickly dominate adaptation under low signal-to-noise ratios.

### 26.4 RFC, AAAI 2024

- **背景**：黑盒 DA 中，某些类别可能在目标域中被忽略或被错误压制，造成 class imbalance 和 forgotten classes。
- **为什么相关**：源数据不可用时，模型难以判断哪些类别在目标域被低估。
- **聚焦问题**：black-box adaptation 下如何重新发现/审查被遗忘类别。
- **对 EEG 的启发**：BCI 中某些任务状态在目标用户上可能样本少或置信度低，容易被 adaptation 过程“抹掉”。

### 26.5 SF(DA)^2, ICLR 2024

- **背景**：SFDA 中缺少源数据，数据增强成为构造鲁棒目标表征的重要方式，但真实增强可能昂贵或不稳定。
- **为什么 SFDA**：需要在无源数据下利用 target 数据局部结构和 feature-level augmentation。
- **聚焦问题**：如何低成本增强目标域结构，避免过拟合单一目标分布。
- **对 EEG 的启发**：EEG 可用时间窗扰动、频带扰动、通道 dropout、时频 masking 或 feature-space augmentation。

### 26.6 Frozen Multimodal Foundation Model SFDA, CVPR 2024

- **背景**：foundation model 越来越常用，但模型很大，预训练数据不可见，权重不宜大规模更新。
- **为什么 SFDA**：下游目标域适配不能访问源训练数据，甚至希望冻结大模型。
- **聚焦问题**：如何借助多模态语义知识完成 source-free adaptation。
- **对 EEG 的启发**：EEG foundation model / brain foundation model 未来可能只作为 frozen backbone 提供；隐私适配要围绕 adapter、prompt、classifier 或 feature alignment 做。

### 26.7 LEAD / Universal SFDA, CVPR 2024

- **背景**：真实目标域类别空间可能和源域不一致，不一定是 closed-set。
- **为什么 SFDA**：无源数据时，判定 shared/private/unknown classes 更困难。
- **聚焦问题**：universal DA，识别目标私有类和源目标共享类。
- **对 EEG 的启发**：目标用户可能出现源数据中没有的状态，例如疲劳、伪迹、非任务状态；实时 BCI 应有 unknown/reject 机制。

### 26.8 ProDe, ICLR 2025

- **背景**：SFDA 的核心瓶颈是 proxy/pseudo-label 噪声。
- **为什么 SFDA**：无源数据下只能依赖目标样本结构和源模型输出构造 proxy。
- **聚焦问题**：denoise proxy，提升自训练稳定性。
- **对 EEG 的启发**：EEG target pseudo-label 的噪声比图像更严重，proxy denoising 可以作为目标适配模块。

## 27. 逐篇 MU 论文动机卡片

### 27.1 Making AI Forget You, NeurIPS 2019

- **背景**：数据删除请求和隐私法规要求模型响应“删除我的数据”。
- **为什么 MU**：模型一旦训练完成，简单删除数据库中的原始样本不能删除模型参数中的影响。
- **聚焦问题**：训练数据删除的算法化处理，而不是每次从头训练。
- **对 EEG 的启发**：用户撤回 EEG 使用授权后，原始 EEG 删除不等于 EEG decoder 删除了该用户影响。

### 27.2 Certified Data Removal, ICML 2020

- **背景**：需要比经验遗忘更严格的理论保证。
- **为什么 MU**：删除后模型应与“从未使用待删数据训练”的模型近似不可区分。
- **聚焦问题**：certified removal、参数不可区分、线性/凸模型下的理论界。
- **对 EEG 的启发**：如果论文要声称强隐私，需要引用 certified removal；如果只做深度 EEG empirical 方法，不能夸大为 certified privacy。

### 27.3 SISA / Machine Unlearning, IEEE S&P 2021

- **背景**：大模型重训成本高，删除请求可能频繁发生。
- **为什么 MU**：通过 shard/slice 训练结构减少重训范围。
- **聚焦问题**：工程化 exact/近似删除，降低删除成本。
- **对 EEG 的启发**：BCI 多用户模型可以按 subject shard 组织训练，方便 user-level deletion；但如果模型已经训练完且没有 shard 设计，SISA 不适用。

### 27.4 Amnesiac Machine Learning, AAAI 2021

- **背景**：模型训练过程中可记录更新信息，以后删除时撤销对应更新。
- **为什么 MU**：避免从头重训。
- **聚焦问题**：记录和逆转训练更新。
- **对 EEG 的启发**：如果实时 BCI 在线学习时记录每个用户/session 的 update log，之后可做更精确的删除；但会增加存储和隐私日志风险。

### 27.5 Remember What You Want to Forget, NeurIPS 2021

- **背景**：MU 需要理论化，不同问题设定下删除难度不同。
- **为什么 MU**：研究删除请求下的统计学习保证。
- **聚焦问题**：遗忘算法的理论复杂度和泛化。
- **对 EEG 的启发**：EEG 小样本和高噪声会让 unlearning 更不稳定，理论上 retain utility 与 forget quality 存在 trade-off。

### 27.6 Towards Unbounded Machine Unlearning, NeurIPS 2023

- **背景**：现有 MU benchmarks 不统一，很多方法只在有限 forget 请求下有效。
- **为什么 MU**：需要评估连续/大量删除请求下模型是否还能保持效用。
- **聚焦问题**：unbounded deletion、SCRUB、MIA-based privacy evaluation。
- **对 EEG 的启发**：实时 BCI 可能发生多次撤回授权或 session 删除，不能只评估一次 forget。

### 27.7 TOFU, ICLR 2024

- **背景**：LLM 中需要删除具体知识、人物、事实或版权内容。
- **为什么 MU**：删除知识不能简单靠从数据库移除文本。
- **聚焦问题**：生成模型遗忘 benchmark 和评估。
- **对 EEG 的启发**：虽然任务不同，但说明 MU 已从分类扩展到复杂模型；评估不能只看 accuracy，还要看泄露和保留能力。

### 27.8 Selective Unlearning via Representation Erasure, ICLR 2025

- **背景**：深度模型的隐私/偏见/域信息常藏在 representation 中，不一定只在输出层。
- **为什么 MU**：需要选择性擦除某种表征因素，而不破坏其他任务能力。
- **聚焦问题**：representation erasure、domain adversarial training。
- **对 EEG 的启发**：EEG 隐私最像 representation erasure：删除 subject/domain identity，保留 task-relevant neural feature。

### 27.9 Towards Source-Free Machine Unlearning, CVPR 2025

- **背景**：传统 MU 需要 retain data；但实际中 retain/source training data 可能因隐私、存储或法规不可访问。
- **为什么 MU**：只用 trained model 和 forget data 删除指定数据影响。
- **聚焦问题**：source-free setting、retain Hessian estimation、theoretical guarantee。
- **对 EEG 的启发**：这正是 EEG SFDA 的核心约束：目标端/模型拥有者没有 retain source EEG，但需要删除某个用户/会话影响。

### 27.10 Approximate Domain Unlearning for VLMs, NeurIPS 2025

- **背景**：大模型可能需要删除某个域、风格、数据源或敏感场景影响。
- **为什么 MU**：domain-level 删除比 sample-level 更接近真实治理需求。
- **聚焦问题**：domain unlearning、VLM utility preservation。
- **对 EEG 的启发**：BCI 中可忘记某医院、某设备、某采集 protocol 的域影响。

## 28. 论文中必须区分的四个概念

### 28.1 Source-free 不等于 data-free

SFDA 只是不访问 source data，但通常仍访问 target data。Source-free MU 通常仍访问 forget data。

严谨写法：

```text
source-free adaptation: no raw source data during target adaptation
source-free unlearning: no retain/source training data during unlearning, but forget data may be available
```

### 28.2 Forget data 不等于 retain data

- `D_f`：要删除的数据。
- `D_r`：要保留的数据。
- `D_t`：目标域无标签数据。

很多算法效果差，是因为删除 `D_f` 时没有保护 `D_r`。

### 28.3 Forget class 不等于 privacy class

视觉 MU 中 forget class 是语义类别；EEG 隐私中 forget target 应是用户/会话/域/属性。

### 28.4 Privacy mitigation 不等于 privacy guarantee

- 有 DP epsilon：可以说 differential privacy guarantee。
- 有 certified removal theorem：可以说 certified unlearning under assumptions。
- 只有 MIA/identity attack 下降：应说 empirical privacy leakage mitigation。

## 29. 审稿人可能质疑与回答

### Q1: SFDA 已经不访问源数据，为什么还需要 MU？

回答：

> SFDA removes raw source data access during adaptation, but the source model remains trained on sensitive source users. Model parameters and outputs may still encode membership, identity or domain-specific information. MU addresses this residual source-model leakage by removing specified source influence before adaptation.

### Q2: 为什么不用 DP？

回答：

> DP is a training-time global protection mechanism, while MU is a post-training selective deletion mechanism. If a source model has already been trained or a user revokes consent after training, DP cannot directly remove that user's learned influence. MU and DP are complementary rather than mutually exclusive.

### Q3: MU 后 forget data 的 task accuracy 是不是应该降到随机？

回答：

> Only for class-level unlearning. For user/session-level EEG privacy unlearning, the goal is not to make the model misclassify the user's task labels. The goal is to remove the user's contribution as training evidence and suppress membership/identity leakage while preserving task semantics.

### Q4: 没有 retain data 怎么保证 retain performance？

回答：

> In a strict source-free setting, retain performance cannot be directly optimized using raw retain data. Existing approaches use parameter regularization, source model behavior preservation, Hessian/influence approximation, surrogate data, target-domain high-confidence samples, or theoretical approximations of the retain Hessian.

### Q5: EEG 中 subject identity 和 task feature 能完全解耦吗？

回答：

> Not perfectly. EEG task and identity factors may be entangled. Therefore, the objective is a controlled trade-off: reduce identity/membership leakage while preserving sufficient task-discriminative information for BCI decoding.

### Q6: 如果 forget subject 之后又作为 target user 使用系统怎么办？

回答：

> The model should not retain that subject's historical source contribution, but it may still adapt to the user's newly provided unlabeled target EEG under a fresh consent boundary. This distinction is important for BCI systems where users may re-enter as target users.

### Q7: 为什么需要 forget set 标签？

回答：

> Unlearning requires a definition of what must be forgotten. In class unlearning this is a task label; in EEG privacy unlearning it is more naturally subject ID, session ID, device ID, hospital ID or consent metadata. Without such metadata, unlearning becomes weakly supervised and less reliable.

## 30. 方法图建议

可以画一张四块图：

```text
             Source Institution
      multi-subject EEG source data
                  |
             source training
                  v
          source EEG model M_s
                  |
      deletion request: subject/session/domain
                  |
                  v
       Source-Free Machine Unlearning
       input: M_s + D_f, no D_r
                  |
                  v
          sanitized model M_u
                  |
        unlabeled target EEG stream
                  |
                  v
        SFDA / CTTA target adaptation
                  |
                  v
        private and adaptive BCI decoder
```

图上标注：

- raw source EEG not transferred。
- retain source EEG unavailable。
- forget influence removed。
- target EEG processed locally。
- attacks evaluated on membership / identity / attributes。

## 31. 表格建议：论文实验结果应该怎么展示

即使现在不跑实验，也可以先设计表格。

### Table 1: Utility after unlearning and adaptation

| Method | Source access | Forget access | Target acc | Retain acc | Forget task acc | ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Source-only | no target adapt | no | - | - | - | - |
| UDA oracle | source + target | no | high | high | high | low |
| SFDA | no source | no | baseline | high | high | medium |
| NegGrad + SFDA | no retain | yes | ? | lower | maybe lower | high |
| RandomLabel + SFDA | no retain | yes | ? | lower | lower | high |
| DP-SGD + SFDA | trained with DP | no | lower? | lower? | high | medium |
| Ours MU + SFDA | no retain | yes | high | high | high if user-level | low |
| Oracle retrain + SFDA | retain only | yes | upper bound | high | high if user-level | low |

### Table 2: Privacy leakage after unlearning

| Method | MIA AUC ↓ | Subject ID acc ↓ | Attribute acc ↓ | Reconstruction corr ↓ |
| --- | ---: | ---: | ---: | ---: |
| SFDA | high | high | high | high |
| DP-SGD + SFDA | lower | lower | lower | lower |
| NegGrad + SFDA | lower | medium | medium | medium |
| Ours MU + SFDA | near random | low | low | low |
| Oracle retrain + SFDA | near random | low | low | low |

### Table 3: Ablation

| Variant | Target acc | MIA AUC | Subject ID acc | Comment |
| --- | ---: | ---: | ---: | --- |
| w/o MU | high | high | high | privacy leakage remains |
| w/o retain proxy | low | low | low | catastrophic forgetting |
| w/o privacy adversary | high | medium | high | identity remains |
| w/o SFDA | low target | low? | low? | adaptation missing |
| full method | high | low | low | best trade-off |

## 32. 最推荐的论文题目

如果论文标题需要强调 BCI：

1. **Source-Free and Forgettable EEG Adaptation for Privacy-Preserving Brain-Computer Interfaces**
2. **Forgetting Source Users in Source-Free EEG Adaptation for Privacy-Preserving BCIs**
3. **Privacy-Aware Source-Free EEG Domain Adaptation via User-Level Machine Unlearning**
4. **Can Source-Free EEG Adaptation Protect Neural Privacy? A Machine Unlearning Perspective**
5. **Forget-and-Adapt: Source-Free Machine Unlearning for Privacy-Preserving EEG-Based BCIs**

最稳的是第 1 个或第 5 个：

- 第 1 个更像正式论文。
- 第 5 个更像方法名明确的 AI 会议论文。

## 33. 可以直接给另一个 Codex 的任务说明

如果要让另一个 Codex 写双栏 LaTeX，可以给它这个 prompt：

```text
请基于 documents/research_notes/sfda_mu_dp_bci_privacy_motivation.md 和 documents/research_notes/eeg_sfda_mu_paper_brief.md 起草一篇双栏 AI 会议风格论文初稿。

论文题目暂定：Forget-and-Adapt: Source-Free Machine Unlearning for Privacy-Preserving EEG-Based BCIs。

要求：
1. 写 abstract, introduction, related work, problem formulation, method, experiment design, discussion, conclusion。
2. 不编造实验结果，只写实验设计和 expected evaluation protocol。
3. 强调 SFDA 减少 raw source EEG exposure，但不保证源模型隐私。
4. 强调 MU 用于 user/session/domain-level post-training deletion，不是 task-class deletion。
5. DP 作为 complementary baseline，不要说 MU 完全优于 DP。
6. 使用 ACM/IEEE 风格引用占位符，可从 documents/research_notes/top_ai_security_references_acm_1_55.md 和本文件第 24 节抽取。
7. 所有隐私结论写成 empirical privacy leakage mitigation，除非明确引用 certified/DP 方法。
```

## 34. 研究假设与可验证问题

可以把论文问题拆成 4 个 research questions。

### RQ1: SFDA 是否足以保护 EEG 源用户隐私？

假设：

```text
H1: SFDA reduces raw-source-data exposure but does not eliminate source-model privacy leakage.
```

验证方式：

- 训练源模型后不暴露源 EEG，只做 SFDA。
- 对源模型或 SFDA 后模型做 membership inference / subject identity inference。
- 如果攻击成功率显著高于随机，说明 SFDA 仍有 residual leakage。

### RQ2: User/session-level MU 是否能降低隐私泄露？

假设：

```text
H2: User/session-level MU reduces membership and identity leakage of forgotten source users.
```

验证方式：

- 选择一个 source subject/session 作为 `D_f`。
- 进行 MU 后再测 MIA AUC 和 subject ID accuracy。
- 与 SFDA-only、NegGrad、RandomLabel、DP-SGD 和 Oracle retrain 比较。

### RQ3: MU 是否会破坏目标用户 BCI 解码？

假设：

```text
H3: Proper retain-preserving MU maintains target adaptation utility better than naive forgetting baselines.
```

验证方式：

- 对 target subject 做 SFDA / CTTA。
- 比较 target accuracy、balanced accuracy、F1、ECE。
- 观察 naive NegGrad 是否出现 retain/target performance collapse。

### RQ4: DP 与 MU 是否互补？

假设：

```text
H4: DP and MU protect different privacy dimensions; MU is more suitable for post-hoc deletion, while DP offers training-time global protection.
```

验证方式：

- 加 `DP-SGD + SFDA` baseline。
- 讨论 DP 对 EEG utility 的影响。
- 不声称 MU 形式上强于 DP，只说明它满足不同需求。

## 35. 方法变体：从简单到复杂的三档方案

### 35.1 最小可写方案

适合初稿，不需要实现复杂 Hessian。

```text
Source model -> User-level NegGrad/RandomLabel unlearning -> SHOT-style SFDA -> privacy attack evaluation
```

优点：容易讲清楚和实现。缺点：创新弱，retain utility 可能差。

### 35.2 中等强度方案

适合作为正式论文设计。

```text
Source model
  -> source-free influence/Hessian-inspired unlearning
  -> identity adversarial representation erasure
  -> SHOT/ProDe-style target adaptation
```

特点：

- 用 CVPR 2025 Source-Free MU 做理论支撑。
- 用 representation erasure 贴合 EEG 隐私。
- 用 SFDA 解决目标域适配。

### 35.3 完整强方案

适合后续扩展。

```text
Source model
  -> certified/approx source-free MU
  -> user/session/domain privacy attacker minimization
  -> CTTA with confidence-gated updates
  -> periodic target-user unlearning/reset
```

特点：

- 同时考虑 source privacy 和 target streaming privacy。
- 适合强调 real-time BCI deployment。
- 成本和实现复杂度更高。

## 36. EEG 实验中忘记对象如何设置

### 36.1 Subject-level forgetting

```text
D_f = all trials from source subject k
```

适合证明：用户撤回授权。

评测：

- forget subject MIA。
- forget subject ID inference。
- retain subjects task accuracy。
- target subject SFDA accuracy。

### 36.2 Session-level forgetting

```text
D_f = all trials from session t of subject k
```

适合证明：某次采集会话含隐私/错误/异常数据，需要删除。

评测：

- session ID inference。
- same subject other sessions utility。
- cross-session robustness。

### 36.3 Domain-level forgetting

```text
D_f = all trials from hospital/device/domain d
```

适合证明：机构退出数据合作，或某设备采集协议不再授权。

评测：

- domain classifier accuracy。
- remaining domains utility。
- target domain adaptation。

### 36.4 Attribute-level forgetting

```text
D_f or privacy target = gender / age group / health condition / handedness
```

适合证明：敏感属性保护。

注意：需要数据集有属性标签；否则不能编造。

## 37. 传统 MU 中 source data class 的处理方式

用户问“如果使用了 MU，那么他们对于模型在 source data class 上的性能是如何处理的”。这里要分情况。

### 37.1 Class-level unlearning

视觉分类常见设定：忘记某个 source class。

处理方式：

- `forget class`：希望准确率下降，或者输出不再包含该类。
- `retain classes`：希望准确率保持。
- `test set`：如果测试集仍含 forget class，要单独报告 forget/retain accuracy。
- `oracle retrain`：用不含 forget class 的数据重训作为目标。

典型表述：

```text
A successful class unlearning method should erase the model's ability to recognize the forgotten class while preserving performance on retained classes.
```

### 37.2 Instance/user-level unlearning

如果忘记的是某些样本或用户，而不是任务类别：

- task class 不应被删除。
- forget samples 的 membership evidence 应下降。
- retain/test accuracy 应保持。
- forget samples 的分类 accuracy 不一定要下降，因为它们的语义仍属于正常任务类别。

典型表述：

```text
For instance- or user-level unlearning, forget accuracy alone is not a sufficient privacy metric; membership inference and retraining equivalence are more meaningful.
```

### 37.3 EEG 中应采用的解释

EEG 隐私保护最好采用 user-level/session-level：

```text
source data class = task labels should remain
privacy class = subject/session/domain should be removed
```

这样就避免了“为了隐私把左手/右手任务类删掉”的荒谬问题。

## 38. Forget influence 的几种定义

不同论文对“影响”定义不完全一样。

| 定义 | 数学/评估含义 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Retraining equivalence | `M_u` 接近 `Train(D_s \ D_f)` | 最直观 | 需要 oracle retrain 才能评估 |
| Parameter indistinguishability | unlearned 参数分布与 retrained 参数分布不可区分 | 理论强 | 通常只在特定模型/假设下成立 |
| Output indistinguishability | 两模型在测试点输出接近 | 容易评估 | 不能保证参数不泄露 |
| Membership privacy | MIA 无法判断 forget data 是否训练过 | 贴合隐私 | 依赖攻击强度 |
| Representation erasure | embedding 不含 subject/domain 信息 | 适合 EEG | 难以证明完全删除 |
| Performance removal | forget class accuracy 下降 | 简单 | 只适合 class unlearning |

对 EEG 论文建议组合：

```text
primary: membership privacy + subject identity erasure + target utility
secondary: output distance to oracle retrain if oracle available
```

## 39. 非计算机文献如何放进论文

### 39.1 Nature / neuroethics 文献放 introduction

用途：说明神经数据隐私不是单纯机器学习问题。

可写：

> Neurotechnology and AI raise ethical concerns around mental privacy, identity, agency and fairness, as emphasized by Yuste et al. in Nature. These concerns are amplified in BCI systems where neural data are directly acquired and interpreted by computational models.

### 39.2 Neurosurgical Focus / medical device security 放安全动机

用途：说明神经设备安全进入医疗安全范畴。

可写：

> Neural devices are increasingly networked and computationally mediated. Prior work on neurosecurity argues that security and privacy failures in neural devices can become patient-safety concerns rather than merely data-protection issues.

### 39.3 Ethics and Information Technology 放 neurosecurity

用途：说明 brain-hacking / neurocrime / unauthorized neural information access。

可写：

> Neurosecurity work has warned that BCI systems may expose users to unauthorized access to neural information or manipulation of neural interfaces, motivating privacy-aware model design.

### 39.4 Science and Engineering Ethics 放情绪识别 BCI

用途：如果论文任务选择 emotion recognition，引用 Wired Emotions。

可写：

> Affective BCIs create additional ethical concerns because inferred emotional states may be sensitive, context-dependent and vulnerable to misuse.

## 40. 最容易写错的表述

### 错误 1：SFDA protects privacy

更正：

```text
SFDA reduces raw source data exposure, but does not provide formal privacy guarantees.
```

### 错误 2：MU guarantees privacy

更正：

```text
Unless certified or exact unlearning is used, MU empirically mitigates privacy leakage rather than guaranteeing privacy.
```

### 错误 3：DP 和 MU 二选一

更正：

```text
DP and MU address complementary privacy requirements: training-time global protection vs post-training selective deletion.
```

### 错误 4：forget class 必须是任务类别

更正：

```text
The forget set can be defined by subject/session/domain metadata; in EEG privacy, this is usually more appropriate than task-class unlearning.
```

### 错误 5：forget subject 的 task accuracy 必须下降

更正：

```text
For user-level forgetting, task accuracy on forget user's samples is not the primary privacy metric. Membership and identity leakage are more appropriate.
```

## 41. 一页版总结

如果后续写论文只能记住一页内容，可以用下面这版。

### Problem

EEG/BCI models need cross-subject adaptation, but source EEG is sensitive and cannot be shared. SFDA solves source-data access during adaptation, but source models may still leak source-user information.

### Gap

Existing EEG SFDA methods focus on utility under source-free constraints. They rarely ask whether the source model still encodes subject identity, session signatures or membership information.

### Proposed idea

Before adapting to target EEG, perform source-free machine unlearning to remove the influence of specified source subjects/sessions/domains, then conduct SFDA/CTTA on unlabeled target EEG.

### Why MU instead of DP

DP is training-time global protection; MU is post-training selective deletion. User consent can change after model training, and source EEG may be unavailable. Therefore MU addresses a post-hoc privacy need that DP alone does not solve.

### What to forget

Forget users/sessions/domains, not task classes. The model should forget who contributed data while preserving what neural task patterns mean.

### How to evaluate

Report both utility and privacy:

- target EEG decoding accuracy
- retain subject accuracy
- membership inference AUC
- subject identity inference accuracy
- attribute inference accuracy
- distance to oracle retrain if possible

### Safe claim

This framework mitigates empirical privacy leakage in source-free EEG adaptation. It does not claim formal DP or certified removal unless those mechanisms are explicitly implemented.
