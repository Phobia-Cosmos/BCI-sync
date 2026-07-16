# 非 BCI 领域数据撤回的 MU 处理方式与 EEG 跨任务/滤波影响

> 目标：回答两个问题：  
> 1. 非 BCI 领域（图像分类、分割、识别、大模型等）如果需要撤回隐私数据，传统 Machine Unlearning 如何处理？大量用户撤回时怎么办？如何降低对原模型性能的影响？  
> 2. EEG 数据可用于 MI、P300、ERN/ErrP、SSVEP、Natural Stimuli 等不同任务，这些任务之间是否有关联？不同范式的频段/预处理是否会影响 MI 性能？简单 LBP/低通/带通过滤是否会损害任务性能？

## 1. 总结结论

### 1.1 非 BCI 的数据撤回如何处理

传统 MU 的目标不是只从数据库删除原始数据，而是让模型接近：

```text
M_unlearned ≈ Train(D_train \ D_forget)
```

也就是模型表现应接近“从未使用撤回数据训练过”的模型。

非 BCI 场景一般有五类处理方式：

1. **从头重训**：最干净，但成本最高；通常作为 oracle upper bound。
2. **SISA / 分片训练**：训练前把数据分 shard/slice，某用户撤回时只重训受影响 shard。
3. **Influence / Hessian / gradient update**：估计撤回样本对参数的影响，然后做反向参数修正。
4. **Negative gradient / random labels / bad teacher**：降低模型对撤回数据的记忆，但容易损害 retain performance。
5. **Distillation / retain regularization / representation erasure**：在删除 forget influence 的同时保持 retain data 或通用任务能力。

大量用户撤回时，其他领域通常不会无限次做局部 patch，而会采用：

- 批量 unlearning。
- 周期性 retraining。
- SISA 式分片重训。
- 有删除预算的 unlearning；超过阈值后重训。
- 使用 DP / certified training 降低单个样本影响，但这不能替代 post-hoc deletion。

### 1.2 EEG 跨任务与滤波的结论

EEG 任务之间存在共享因素，也存在明显任务特异性：

- 共享因素：被试身份、头皮传导、电极接触、设备噪声、通道拓扑、基础频谱特征、伪迹。
- 任务特异性：MI 主要看 sensorimotor mu/beta ERD/ERS；P300/ERN 是事件相关电位 ERP；SSVEP 是外源视觉稳态频率响应。

因此：

- 一个范式的数据或预处理不能随便套到另一个范式。
- P300/ERN 低频 ERP 预处理如果过度低通，可能丢掉 MI 中重要 beta 成分。
- SSVEP 常用刺激频率和谐波滤波，不等同于 MI 的 8–30 Hz sensorimotor band。
- MI 常用 FBCSP/滤波器组正是因为固定单一频段可能不够鲁棒。
- 如果你说的 `LBP` 是 Local Binary Pattern 特征，它不是标准“滤波”；如果是 low-pass/band-pass filter，则 cutoff 选择会直接影响 MI 特征。

对论文的安全表述：

> EEG contains both task-relevant and task-agnostic factors. Cross-task transfer is possible at the representation level, but task-specific temporal-frequency structures must be preserved. A preprocessing pipeline designed for P300 or SSVEP should not be assumed optimal for MI without validation.

## 2. 非 BCI 领域：撤回隐私数据时传统 MU 如何做

### 2.1 标准问题定义

给定训练集：

```text
D = D_retain ∪ D_forget
```

撤回请求：

```text
D_forget = 用户、样本、类别、域、版权内容、敏感数据子集
```

目标：

```text
M_unlearned should behave like M_retrained = Train(D_retain)
```

同时要满足：

- forget data 的 membership / influence 下降。
- retain data 的 accuracy / mIoU / mAP / F1 尽量保持。
- 计算成本显著低于 full retraining。

### 2.2 图像分类中的处理

图像分类是 MU 最常见实验场景。撤回对象可能是：

| 撤回对象 | 例子 | 处理方式 | 性能评估 |
| --- | --- | --- | --- |
| 样本级 | 删除某些 ImageNet/CIFAR 图片 | influence update、gradient ascent、SISA、certified removal | retain/test accuracy，MIA，retrained distance |
| 用户级 | 删除某个用户上传的全部图片 | SISA、分组重训、user-level unlearning | 其他用户 accuracy，撤回用户 MIA |
| 类别级 | 删除 dog/truck/person 类 | class unlearning，negative gradient，random labels，输出层修改 | forget class accuracy 降，retain classes 保持 |
| 域级 | 删除某来源/风格/医院图像 | domain unlearning，representation erasure | domain classifier accuracy 降，任务 accuracy 保持 |

关键点：

- 如果是 **class unlearning**，forget class accuracy 下降是合理目标。
- 如果是 **sample/user unlearning**，不一定要求该样本的类别预测错；更重要是 membership/influence 下降。
- 如果撤回的是隐私数据，通常更像 sample/user/domain unlearning，不是 class unlearning。

### 2.3 图像分割/检测中的处理

分割/检测比分类更复杂，因为一个样本中包含多个对象、像素和上下文。

可能的撤回对象：

- 某些训练图像。
- 某个用户/城市/医院来源的数据。
- 某一类对象，例如 person/car。
- 图像中的某些 mask/区域。

处理难点：

- 删除一个图像可能影响 backbone 的通用表征。
- 删除一个类别会影响 shared feature 和背景建模。
- 分割指标是 mIoU，检测指标是 mAP，不能只看分类 accuracy。
- 隐私撤回常常不是删除语义类别，而是删除数据源或个体贡献。

合理策略：

- 如果撤回的是某些图像/用户：做 sample/user-level unlearning，同时保持 mIoU/mAP。
- 如果撤回的是某个类别：允许该类别 mIoU/mAP 下降，但保留其他类别。
- 如果撤回的是隐私属性/域：做 representation/domain erasure，而不是破坏语义任务能力。

### 2.4 大量用户撤回时怎么办

如果大部分用户都撤回数据，传统 MU 不能无限维持原性能。原因很简单：retain data 本身变少了，模型可学习信息减少，性能下降是不可避免的。

其他领域通常有以下处理：

#### 方案 A：批量 unlearning

把多个删除请求合并处理：

```text
D_forget = union of all pending deletion requests
```

优点：减少反复参数扰动。缺点：一次性删除规模大时 utility 会下降。

#### 方案 B：SISA / 分片重训

训练前把数据拆成 shard/slice：

```text
M = Aggregate(M_1, M_2, ..., M_k)
```

某个用户撤回时，只重训包含该用户的 shard。

优点：适合频繁撤回。缺点：训练前要设计；模型数量和存储成本增加。

#### 方案 C：删除预算 + 周期性重训

设置阈值：

```text
if deleted_fraction < threshold:
    approximate unlearning
else:
    retrain from remaining data
```

这是工程上最现实的策略。因为 approximate MU 的误差会累积。

#### 方案 D：DP / certified training 作为预防

DP 或 certified removal 可降低单个样本影响，但：

- DP 不能直接执行“指定用户 post-hoc 删除”。
- certified removal 需要强假设或训练时设计。
- 强 DP 可能降低模型性能。

#### 方案 E：重新收集/授权新数据

如果大量用户撤回，retain set 不足，模型性能无法凭算法凭空恢复。此时只能：

- 重新收集授权数据。
- 使用公开数据/合成数据/预训练模型。
- 降低模型能力范围。
- 提示系统进入低置信度/不可用状态。

可直接写进论文的结论：

> When deletion requests become large-scale, unlearning cannot preserve the original performance for free. Existing work typically treats retraining on retained data as the oracle, uses SISA or batched unlearning to reduce cost, and accepts a utility–privacy trade-off when retained data become insufficient.

## 3. 如何消除撤回对原模型性能的影响

严格说，不是“消除”，而是“缓解”。传统 MU 用以下约束维持 retain utility。

### 3.1 Retain data distillation

如果 `D_retain` 可访问：

```text
L_retain = KL(M_unlearned(x_r) || M_original(x_r))
```

或者直接在 retain data 上继续训练。

### 3.2 Parameter regularization

限制参数不要偏离原模型太多：

```text
L_reg = ||θ_unlearned - θ_original||^2
```

适合 source-free 或 retain data 不可访问场景，但保护能力有限。

### 3.3 Fisher / Hessian / influence correction

估计 forget data 对参数的影响：

```text
Δθ ≈ H_retain^{-1} ∇_forget
```

然后反向更新参数。

优点：理论清楚。缺点：深度网络中 Hessian 难估计，source-free 时更难。

### 3.4 Teacher-student / bad teacher

训练一个 student：

- 在 retain/surrogate 数据上模仿 good teacher。
- 在 forget 数据上远离 original model 或模仿 bad teacher。

优点：适合深度模型。缺点：需要 surrogate/retain 数据，否则容易不稳定。

### 3.5 Representation erasure

如果撤回目标是隐私属性/域/身份：

```text
feature should preserve task label
feature should not predict sensitive attribute
```

适合图像域隐私、医疗域隐私，也适合 EEG subject identity removal。

### 3.6 Oracle retrain

始终作为评估上界：

```text
Train(D_retain)
```

如果 MU 后性能接近 oracle retrain，说明删除和保留做得好。

## 4. 对 BCI 数据撤回的映射

BCI 中有两类需求，不能混淆。

### 4.1 真正撤回授权

用户说：不要再使用我的 EEG 数据。

应处理：

```text
删除原始 EEG + 删除模型中该用户/session 的训练影响
```

此时 forget unit 应是：

- subject-level。
- session-level。
- device/hospital-level。

评估：

- 该 subject/session 的 membership inference AUC 应下降。
- subject identity leakage 应下降。
- retain subjects 和 target users 的任务性能尽量保持。

### 4.2 只保护隐私特征

用户说：可以用我的数据提升任务性能，但不要泄露身份/年龄/性别/健康状态。

应处理：

```text
保留 task-relevant representation
移除 identity/attribute/domain representation
```

这更像：

- representation erasure。
- domain adversarial learning。
- attribute unlearning。
- privacy-preserving representation learning。

评估：

- task accuracy 保持。
- identity/attribute classifier accuracy 下降。

## 5. EEG 不同任务之间是否存在关联性

### 5.1 存在共享因素

不同 EEG 任务共享很多非任务因素：

- 被试身份：头皮传导、脑结构差异、个体频谱特征。
- 设备/电极：采样率、通道位置、阻抗、噪声。
- 会话状态：疲劳、注意、情绪、睡眠、药物。
- 通用脑电结构：delta/theta/alpha/beta/gamma 频段、空间拓扑、时序相关性。
- 伪迹：眼动、肌电、工频噪声、运动伪迹。

这就是为什么 EEG foundation model、跨任务预训练、跨被试迁移有意义。

### 5.2 也存在强任务特异性

| 任务 | 主要神经响应 | 常用频段/时间结构 | 主要通道区域 | 标签 |
| --- | --- | --- | --- | --- |
| MI | sensorimotor rhythm ERD/ERS | 常见 8–30 Hz，mu/beta；具体因人而异 | C3/C4/Cz 附近 | 左手/右手/脚/舌/rest |
| P300 | 事件相关正波 | 低频 ERP，约 250–500 ms | parietal/central 常见 | target/non-target |
| ERN/ErrP | 错误相关负波/正波 | 事件锁定低频 ERP，约 0–600 ms | fronto-central 常见 | error/correct |
| SSVEP | 稳态视觉诱发响应 | 刺激频率及谐波，如 8–15 Hz 及更高谐波 | occipital/parietal | 目标频率/相位 |
| NS | 自然图像/视频/语音诱发响应 | 取决于刺激与模态 | EEG/fMRI/MEG/ECoG 均可能 | 图像/语义/语音/视频等 |

因此跨任务关联是“共享底层噪声/身份/通用表征 + 任务特异神经响应”的关系。

## 6. P300/SSVEP/ERN 的频段是否会影响 MI

会。原因是不同任务依赖的时频结构不同。

### 6.1 MI 对频段很敏感

MI 主要利用 sensorimotor rhythm 的 ERD/ERS，常在 mu 和 beta 范围。FBCSP 使用多个滤波器组就是为了自动选择被试和任务相关频段。如果预处理把 beta 成分滤掉，MI 性能可能下降。

例如：

- P300/ERN 常用低频 ERP pipeline，可能低通到 20–30 Hz 或更低。
- 如果只保留很低频段，MI 的 beta 信息会受损。
- 如果过度平滑或低通，CSP/深度模型捕捉到的节律变化会减少。

### 6.2 P300/ERN 更依赖事件锁定低频 ERP

P300/ERN 的核心是 stimulus/response/feedback locked ERP。它们常需要：

- epoch 对齐。
- baseline correction。
- 低频保留。
- 通常不依赖高频谐波结构。

这类 pipeline 不一定适合 MI，因为 MI 是持续想象状态中的节律调制。

### 6.3 SSVEP 更依赖刺激频率和谐波

SSVEP 常用 CCA/FBCCA/TRCA 等方法，重点是：

- 刺激频率。
- 相位。
- 谐波。
- 枕区通道。

SSVEP 的滤波器组不等同于 MI 的 FBCSP。把 SSVEP 的频率滤波策略直接套到 MI，未必有效。

## 7. 简单 LBP/滤波是否会影响 MI 性能

这里先澄清：

- 如果你说的 `LBP` 是 **Local Binary Pattern**，它是纹理/局部模式特征，不是标准 EEG 滤波器。
- 如果你说的是 **low-pass / band-pass filtering**，那 cutoff 选择会显著影响 MI。

### 7.1 如果是 Local Binary Pattern

LBP 可用于把 EEG 时频图、拓扑图或特征图转成局部纹理特征。但它可能：

- 丢失相位和连续时序信息。
- 对 MI 的 ERD/ERS 频带动态表达不足。
- 对跨被试泛化不一定稳。

所以 LBP 可以作为 feature baseline，但不应作为唯一预处理来声称保留 MI 信息。

### 7.2 如果是低通/带通滤波

对 MI 来说：

```text
过低 low-pass cutoff -> 可能丢 beta rhythm
过窄 band-pass -> 可能错过个体差异频带
过宽 band-pass -> 可能引入噪声/伪迹
```

更稳的做法：

- 使用 task-specific bandpass。
- MI 用 filter bank 或 learnable temporal convolution。
- P300/ERN 用 ERP-friendly 低频 pipeline。
- SSVEP 用 harmonics-aware filter bank。
- 跨任务模型中保留较宽频段，再由模型学习任务相关频带。

### 7.3 对论文最稳的说法

> A single preprocessing filter should not be assumed task-agnostic. For MI, preserving mu/beta rhythms is critical; ERP-oriented low-pass filtering or SSVEP-oriented harmonic filtering may suppress or distort MI-relevant information. Therefore, task-aware filtering or learnable filter banks should be used when evaluating cross-task EEG adaptation.

## 8. 顶会/顶刊支撑文献

### 8.1 Machine Unlearning / 数据撤回

[MU-1] Ginart A, Guan M Y, Valiant G, et al. Making AI Forget You: Data Deletion in Machine Learning[C]//Advances in Neural Information Processing Systems. 2019.  
用途：数据删除请求/MU 的早期基础。

[MU-2] Guo C, Goldstein T, Hannun A, et al. Certified Data Removal from Machine Learning Models[C]//Proceedings of the 37th International Conference on Machine Learning. 2020.  
用途：certified removal、删除后接近未训练过该数据。

[MU-3] Bourtoule L, Chandrasekaran V, Choquette-Choo C A, et al. Machine Unlearning[C]//Proceedings of the IEEE Symposium on Security and Privacy. 2021.  
用途：SISA、频繁删除请求、分片重训。

[MU-4] Graves L, Nagisetty V, Ganesh V. Amnesiac Machine Learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021.  
用途：记录/撤销训练更新。

[MU-5] Sekhari A, Acharya J, Kamath G, et al. Remember What You Want to Forget: Algorithms for Machine Unlearning[C]//Advances in Neural Information Processing Systems. 2021.  
用途：MU 理论与 retain/forget trade-off。

[MU-6] Neel S, Roth A, Sharifi-Malvajerdi S. Descent-to-Delete: Gradient-Based Methods for Machine Unlearning[C]//Proceedings of Algorithmic Learning Theory. 2021.  
用途：gradient-based unlearning 理论。

[MU-7] Kurmanji M, Triantafillou P, Hayes J, et al. Towards Unbounded Machine Unlearning[C]//Advances in Neural Information Processing Systems. 2023.  
用途：大量/持续删除请求、保留性能和 privacy evaluation。

[MU-8] Maini P, et al. TOFU: A Task of Fictitious Unlearning for LLMs[C]//International Conference on Learning Representations. 2024.  
用途：复杂模型中 knowledge unlearning 和评估。

[MU-9] Sepahvand N M, Triantafillou E, Larochelle H, et al. Selective Unlearning via Representation Erasure Using Domain Adversarial Training[C]//International Conference on Learning Representations. 2025.  
用途：representation erasure，适合隐私属性/域/身份删除。

[MU-10] Ahmed S M, Basaran U Y, Raychaudhuri D S, et al. Towards Source-Free Machine Unlearning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025.  
用途：无 retain/source data 场景，贴近 EEG SFDA。

[MU-11] Kawamura K, Goto Y, Yanagi R, et al. Approximate Domain Unlearning for Vision-Language Models[C]//Advances in Neural Information Processing Systems. 2025.  
用途：domain-level unlearning，适合类比医院/设备/数据源撤回。

### 8.2 EEG 跨任务、范式和滤波

[EEG-1] Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces[J]. Journal of Neural Engineering, 2018, 15(5): 056013.  
用途：一个紧凑网络可覆盖 P300、ERN、MRCP、SMR 等多种 EEG BCI 范式，支撑“跨任务共享表征存在”。

[EEG-2] Ang K K, Chin Z Y, Wang C, et al. Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b[J]. Frontiers in Neuroscience, 2012, 6: 39.  
用途：MI 使用 filter bank 选择多频段，支撑 MI 对频段选择敏感。

[EEG-3] Pfurtscheller G, Lopes da Silva F H. Event-related EEG/MEG synchronization and desynchronization: basic principles[J]. Clinical Neurophysiology, 1999, 110(11): 1842-1857.  
用途：MI/运动相关 ERD/ERS 的基础神经生理机制。

[EEG-4] Farwell L A, Donchin E. Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials[J]. Electroencephalography and Clinical Neurophysiology, 1988, 70(6): 510-523.  
用途：P300 speller 经典文献，P300 是 ERP 范式。

[EEG-5] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain-computer interface[J]. Proceedings of the National Academy of Sciences, 2015, 112(44): E6058-E6067.  
用途：SSVEP 高频拼写系统，支撑 SSVEP 使用频率/相位/谐波信息。

[EEG-6] Nakanishi M, Wang Y, Chen X, et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis[J]. IEEE Transactions on Biomedical Engineering, 2018, 65(1): 104-112.  
用途：SSVEP 的 task-related component / 高频率目标检测。

[EEG-7] Wu D, Xu Y, Lu B L. Transfer Learning for EEG-Based Brain-Computer Interfaces: A Review of Progress Made Since 2016[J]. IEEE Transactions on Cognitive and Developmental Systems, 2022, 14(1): 4-19.  
用途：跨被试/跨会话/跨任务 EEG transfer learning 总览。

[EEG-8] He H, Wu D. Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach[J]. IEEE Transactions on Biomedical Engineering, 2020, 67(2): 399-410.  
用途：跨被试 EEG alignment，说明 EEG 分布偏移和适配必要性。

[EEG-9] Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization[J]. Human Brain Mapping, 2017, 38(11): 5391-5420.  
用途：深度模型学习 EEG 时频/空间特征，支撑可学习滤波和任务相关特征。

[EEG-10] Roy Y, Banville H, Albuquerque I, et al. Deep learning-based electroencephalography analysis: a systematic review[J]. Journal of Neural Engineering, 2019, 16(5): 051001.  
用途：EEG 深度学习系统综述，支撑不同任务、预处理、网络结构差异。

[EEG-11] Kostas D, Aroca-Ouellette S, Rudzicz F. BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data[J]. Frontiers in Human Neuroscience, 2021, 15: 653659.  
用途：大规模 EEG 自监督表征，支撑跨任务共享表征。

[EEG-12] Jiang W, Zhao L, Lu B L. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI[C]//International Conference on Learning Representations. 2024.  
用途：EEG foundation model / generic representation，支撑跨任务预训练和共享结构。

## 9. 可直接写进论文的两段话

### 9.1 非 BCI MU 撤回数据段

Machine unlearning in non-BCI domains is commonly motivated by post-training deletion requests, privacy regulations, data ownership changes and the right to be forgotten. In image classification, segmentation and recognition, the standard objective is not merely to remove raw files from storage, but to make the unlearned model behave similarly to a model retrained from scratch without the forgotten data. Existing methods reduce retraining cost through sharding-based training, influence/Hessian updates, gradient-based forgetting, distillation and representation erasure. However, when deletion requests become large-scale, retaining the original utility is not always possible because the effective training distribution changes; practical systems therefore combine batched unlearning, periodic retraining, deletion budgets and retain-performance regularization.

### 9.2 EEG 跨任务与滤波段

EEG tasks share subject-specific, device-specific and spectral-spatial structures, which explains why transfer learning and foundation models can learn reusable EEG representations. Nevertheless, BCI paradigms have strong task-specific temporal-frequency signatures: MI relies on sensorimotor mu/beta ERD/ERS, P300 and ErrP rely on event-locked low-frequency ERPs, and SSVEP relies on stimulus frequencies and harmonics. Therefore, preprocessing designed for one paradigm should not be assumed optimal for another. In particular, ERP-oriented low-pass filtering or SSVEP-oriented harmonic filtering may suppress MI-relevant mu/beta information, while a single handcrafted feature such as LBP may discard temporal and phase information important for MI decoding. Task-aware filtering or learnable filter banks are necessary when evaluating cross-task EEG adaptation.
