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
