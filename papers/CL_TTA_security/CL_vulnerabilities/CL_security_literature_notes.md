# Continual Learning 安全与防护文献整理

本文档整理前面对 Continual Learning (CL) 安全问题的讨论，重点面向后续研究：

> EEG 场景中 CL 过程可能面临的 data poisoning / backdoor / targeted forgetting 攻击，以及 replay-based 和 regularization-based CL 中可迁移的防御策略。

---

## 1. CL 的主要类型

CL 的常见方法不只 replay 和 regularization 两类，但这两类是安全研究中最常被攻击和分析的对象。

### 1.1 Replay-based CL

Replay-based CL 的核心思想是：训练新任务时，同时重放旧任务信息。

训练目标可简化为：

```text
L = L(current task data) + L(replayed old data)
```

旧任务信息的来源包括：

| 类型 | 旧知识来源 | 代表方法 |
|---|---|---|
| Exemplar replay / memory replay | 存储少量真实旧样本 | ER, GEM, A-GEM, iCaRL, ER-ACE |
| Generative replay | 用生成模型产生旧样本 | DGR, cGAN/cVAE replay |
| Feature / prototype replay | 存储旧任务特征、logits、prototype | prototype replay, distillation replay |

Replay-based CL 的设计原因是：模型只训练当前任务会快速遗忘旧任务；replay 通过近似联合训练来缓解遗忘。

主要安全风险：

- buffer 中的污染样本会被反复 replay，攻击效果被放大。
- replay sampler 被操控会导致旧任务分布失衡。
- generative replay 中的 generator 一旦被污染，会持续生成错误旧样本。
- 后门样本可能通过 replay 在后续任务中长期保留。

### 1.2 Memory-based CL

Memory-based CL 通常是 replay-based CL 的一种具体实现，也叫 exemplar replay 或 experience replay。

基本流程：

```text
新任务数据到来
-> 选择部分样本进入 memory / replay buffer
-> 训练时混合 current batch 和 memory batch
-> buffer 满后根据策略替换旧样本
```

常见 memory 管理策略：

| 策略 | 原理 |
|---|---|
| Reservoir sampling | 流式随机保留样本，近似均匀采样 |
| Herding / exemplar selection | 选择最接近类别中心的样本 |
| Diversity sampling | 保证 buffer 覆盖不同模式 |
| Class-balanced memory | 每类保留相近数量样本 |
| Gradient-based memory | 选择对梯度或决策边界影响大的样本 |
| Loss/uncertainty-based selection | 根据损失、不确定性或难度选择样本 |

Memory-based CL 的安全关键点是：样本一旦进入长期 memory，它会在很多后续任务中被重复使用。因此，memory 写入阶段是一个非常重要的防御点。

### 1.3 Regularization-based CL

Regularization-based CL 不存储旧数据，而是通过正则项限制模型参数或输出的变化。

典型目标：

```text
L = L(current task data) + lambda * penalty(theta, old_theta, importance)
```

代表方法：

| 方法 | 核心思想 |
|---|---|
| EWC | 用 Fisher information 估计旧任务重要参数，限制这些参数变化 |
| Online EWC | EWC 的在线版本，累积旧任务重要性 |
| SI | 根据训练轨迹估计参数对旧任务的贡献 |
| MAS | 根据输出敏感性估计参数重要性 |
| LwF | 用旧模型输出作为软标签进行 knowledge distillation |

主要安全风险：

- 投毒样本在当前任务 loss 中产生异常梯度，regularization 无法完全阻止。
- 攻击者可以让模型把错误知识或后门当成需要保护的“重要知识”。
- 如果 Fisher/SI/MAS 的重要性估计被污染，模型会错误保护或错误释放部分参数。

### 1.4 Replay 与 Regularization 的区别和共同点

| 维度 | Replay-based CL | Regularization-based CL |
---|---|---|
| 旧知识保存方式 | 存旧样本、旧特征、prototype 或 generator | 存参数重要性、旧模型输出或正则约束 |
| 是否依赖旧数据 | 通常依赖 | 通常不依赖 |
| 存储成本 | 较高 | 较低 |
| 隐私风险 | 高，尤其保存原始样本时 | 较低 |
| 对 data poisoning 的敏感点 | buffer、sampler、generator、exemplar labels | 当前任务数据、importance estimation、task update direction |
| 典型攻击后果 | 污染样本被长期 replay，导致遗忘或后门持久化 | 错误知识被 regularizer 固化，或旧知识保护失败 |

共同点：

- 都要在稳定性和可塑性之间折中。
- 都依赖任务顺序和流式数据质量。
- 都可能在训练阶段被 data poisoning、label flipping、clean-label attack、backdoor attack 影响。
- 都可以通过 task-level verification、数据过滤、异常梯度检测、少量可信 anchor data 来增强鲁棒性。

---

## 2. CL 中常见攻击类型

### 2.1 Data Poisoning

攻击者在训练数据流中加入污染样本，使模型在学习新任务时遗忘旧任务、学习错误决策边界，或后续出现定向错误。

常见形式：

- Dirty-label poisoning：修改样本标签。
- Clean-label poisoning：标签保持正确，但样本被优化成有害样本。
- Task-level poisoning：污染某一个 task、session、client 或 data supplier。
- Buffer poisoning：污染样本进入 replay buffer 后被长期重放。

### 2.2 Backdoor Attack

攻击者在训练阶段插入带 trigger 的样本，使模型在正常输入上表现正常，但遇到 trigger 时输出攻击者指定类别。

在 CL 中，backdoor 的特殊性是：

- trigger 可以跨任务传播。
- replay 可能帮助后门保持。
- regularization 可能把后门当成旧知识保护。
- prompt-based CL 中 trigger 可能控制 prompt selection。

### 2.3 Targeted Forgetting / False Memory

攻击目标不是让模型在 trigger 上错误，而是让模型忘记某个旧任务或旧类别。

典型机制：

- 后续任务中插入少量 poison。
- 利用稳定性机制，让模型保护错误知识。
- 利用新任务梯度破坏旧任务边界。

### 2.4 Replay Buffer Attack

针对 replay-based CL 的攻击，目标是 replay buffer 或 replay 过程。

攻击方式包括：

- 污染进入 buffer 的样本。
- 修改 buffer 中样本或标签。
- 控制 replay sampler 选哪些样本。
- 使 generator 学到错误分布。
- 制造类别不平衡的 replay batch。

### 2.5 Generative Replay Poisoning

如果 CL 用 generator 产生旧任务样本，攻击者可以污染当前任务，使 generator 之后生成错误旧样本，从而导致模型持续遗忘旧任务。

### 2.6 Prompt-Based CL Backdoor

在 prompt-based CL 中，攻击目标可能不是分类头，而是 prompt pool、key-query matching 或 prompt selection。

例如，trigger 可以让样本被路由到攻击者希望的 prompt，从而诱导目标类别输出。

---

## 3. Replay-Based CL 的攻击面、防御和论文

### 3.1 攻击面

| 攻击目标 | 攻击方式 | 后果 |
|---|---|---|
| Incoming stream | 污染当前任务数据 | 污染样本可能进入 buffer |
| Replay buffer | 修改或污染 stored exemplars | 污染长期保留并被反复 replay |
| Replay sampler | 控制 replay index set | 类别或任务重放比例失衡，引起遗忘 |
| Exemplar labels | label flipping 或 clean-label poison | 旧任务边界被破坏 |
| Generator | 污染 generative replay 模型 | 未来生成错误旧样本 |
| Replay training loss | 让 replay 梯度偏离旧任务保护方向 | 加剧 forgetting |

### 3.2 代表攻击论文

| 论文 | 年份/会议 | 攻击对象 | 说明 |
|---|---:|---|---|
| Amnesia: A Stealthy Replay Attack on Continual Learning Dreams | 2026 arXiv | replay sampler / replay index set | 不改数据和模型，只操控 replay 时从 buffer 取哪些样本 |
| Are Exemplar-Based Class Incremental Learning Models Victim of Black-Box Poison Attacks? | WACV 2025 | exemplar set | 黑盒攻击 exemplar-based CIL |
| BrainWash: A Poisoning Attack to Forget in Continual Learning | CVPR 2024 | current task data, replay CL | 通过 poison noise 促使模型忘记旧任务，也评估 ER/ER-ACE |
| Poisoning Generative Replay in Continual Learning to Promote Forgetting | ICML 2023 | generative replayer | 污染 generator/replayer，使其生成有害旧样本 |
| PACOL: Poisoning Attacks Against Continual Learners | 2023 arXiv | regularization 和 DGR | 含 DGR/generative replay 实验 |
| Backdoor Attacks Against Incremental Learners | 2023 arXiv | 多种 replay learners | 评估 DGR, ER, A-GEM, iCaRL 等 |
| Persistent Backdoor Attacks in Continual Learning | USENIX Security 2025 | 多种 CL 方法 | 后门在 CL 过程中持续存在，包括 replay learners |

### 3.3 对应防御思路

| 防御点 | 防御方法 | 对应论文 |
|---|---|---|
| 数据进入 buffer 前 | outlier detection、feature kNN、loss filtering、clean-label poison detection | Deep k-NN Defense against Clean-Label Data Poisoning Attacks；Spectral Signatures；Activation Clustering |
| buffer 写入阶段 | purified buffer、diversity + purity sampling、delayed insertion | Self-Purified Replay；PuriDivER；NLOCL；Alternate Replay |
| buffer 使用阶段 | replay sampler audit、class-balanced replay、task-balanced replay | Amnesia 提醒需要检查 replay index distribution |
| replay batch 训练 | robust loss、sample reweighting、semi-supervised relabeling | Co-teaching；DivideMix；SELFIE；ELR |
| generative replay | generated sample filtering、generator validation、anchor memory 校验 | Poisoning Generative Replay 中评估 ν-SVM filtering |
| backdoor persistence | trigger inversion、entropy detection、fine-pruning、anti-backdoor training | Neural Cleanse；STRIP；Fine-Pruning；Adversary Aware Continual Learning |

### 3.4 Replay-Based CL 防御论文

| 论文 | 年份/会议 | 防御对象 | 核心贡献 |
|---|---:|---|---|
| Continual Learning on Noisy Data Streams via Self-Purified Replay | ICCV 2021 | noisy stream + replay buffer | Self-Replay + Self-Centered Filter，维护 purified replay buffer |
| Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries | CVPR 2022 | online blurry CL + corrupted labels | PuriDivER，memory sampling 同时考虑 purity 和 diversity |
| NLOCL: Noise-Labeled Online Continual Learning | Electronics 2024 | online CL + noisy labels | 分离 clean/noisy 样本，结合 replay 和 semi-supervised fine-tuning |
| May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels | BMVC 2024 | replay under noisy labels | AER + ABS，利用 forgetting behavior 过滤 noisy samples |
| Noise-Tolerant Coreset-Based Class Incremental Continual Learning | 2025 arXiv | CIL + noisy data | Continual CRUST / Continual Cosine-CRUST |
| Class-incremental SAR ATR in Noisy and Adversarial Environments | SPIE 2025 | SAR class-incremental CL | 用 noise-tolerant coreset/replay buffer 抵抗 noisy/adversarial data |
| Adversary Aware Continual Learning | 2023 arXiv / IEEE Access | CL backdoor | 用 defender-controlled perceptible pattern 对抗 imperceptible trigger |

---

## 4. Regularization-Based CL 的攻击面、防御和论文

### 4.1 攻击面

| 攻击目标 | 攻击方式 | 后果 |
|---|---|---|
| 当前任务数据 | 加入少量 poison | 新任务梯度破坏旧任务知识 |
| Importance estimation | 污染 Fisher/SI/MAS 重要性估计 | 错误保护参数或释放关键参数 |
| Regularization loss | 让 poison 梯度绕过 penalty | regularization 无法保护旧任务 |
| False memory | 将后门或错误标签变成“需要保护的知识” | 后续任务继续保留错误知识 |
| Task update direction | 污染某个 task 的整体更新方向 | 一个 task 就能造成长期退化 |

### 4.2 代表攻击论文

| 论文 | 年份/会议 | 攻击对象 | 说明 |
|---|---:|---|---|
| Targeted Forgetting and False Memory Formation in Continual Learners | IJCNN 2020 | EWC | 后续任务注入后门样本，让模型忘记目标旧任务或形成 false memory |
| Targeted Data Poisoning Attacks Against Continual Learning Neural Networks | IJCNN 2022 | EWC, Online EWC, SI | 白盒 target poisoning，破坏 regularization-based CL |
| Data Poisoning Attack Aiming the Vulnerability of Continual Learning | ICIP 2023 | EWC, SI 等 | 利用 CL 脆弱性进行数据投毒 |
| PACOL: Poisoning Attacks Against Continual Learners | 2023 arXiv | EWC, Online EWC, SI, DGR | clean-label/adversarial poison，攻击旧任务保持能力 |
| BrainWash: A Poisoning Attack to Forget in Continual Learning | CVPR 2024 | 多种 regularization methods | 使模型定向遗忘旧类，评估 EWC, MAS, RWALK, AFEC, ANCL |
| Single-Task Data Poisoning in Exemplar-Free Continual Learning | 2025 arXiv / CoLLAs | exemplar-free CL | 一个 task 被污染即可造成长期影响 |

### 4.3 对应防御思路

| 防御点 | 防御方法 | 对应论文 |
|---|---|---|
| 当前 task 数据 | 输入过滤、feature anomaly detection、label consistency check | Deep k-NN Defense；Spectral Signatures；Activation Clustering |
| 参数重要性估计 | 用 trusted anchor data 估计 importance；robust Fisher/SI/MAS；异常任务剔除 | Theory of CL Against Data Poisoning Attacks |
| task update | task-vector anomaly detection；task-to-task verification；rollback | Single-Task Data Poisoning in Exemplar-Free CL；Theory of CL Against Data Poisoning Attacks |
| 后门 false memory | task 后进行 trigger inversion / pruning / unlearning | Neural Cleanse；STRIP；Fine-Pruning；I-BAU |
| regularization 失效 | gradient monitoring；限制异常梯度；旧任务验证集检查 | PACOL 和 BrainWash 揭示问题，现有防御多为迁移式 |

### 4.4 Regularization-Based CL 防御论文

| 论文 | 年份/会议 | 防御对象 | 核心贡献 |
|---|---:|---|---|
| Theory of Continual Learning Against Data Poisoning Attacks | 2026 arXiv | regularization-based CL | task-to-task verification；robust feature defense；理论分析可防边界 |
| Addressing the Devastating Effects of Single-Task Data Poisoning in Exemplar-Free Continual Learning | 2025 arXiv / CoLLAs | exemplar-free CL | task vector detection，发现 poisoned task |
| Temporal Robustness against Data Poisoning | NeurIPS 2023 | 持续数据收集和周期更新 | temporal aggregation，抵抗短时间窗口投毒 |

说明：相比 replay-based CL，regularization-based CL 的专门防御论文更少。很多防御需要从通用 data poisoning/backdoor defense 中迁移。

---

## 5. 通用 Data Poisoning / Backdoor 防御论文

这些论文不是专门为 CL 提出的，但可以迁移到 CL 的数据进入阶段、buffer 写入阶段、task 后安全检查阶段。

### 5.1 Data Poisoning / Clean-Label Poisoning 防御

| 防御方法 | 对应论文 | 可迁移到 CL 的位置 |
|---|---|---|
| Certified outlier removal | Certified Defenses for Data Poisoning Attacks, NeurIPS 2017 | 每个 task 训练前的数据过滤 |
| Deep feature kNN | Deep k-NN Defense against Clean-Label Data Poisoning Attacks, ECCV Workshop 2020 | 检查样本与同类邻居是否一致 |
| Spectral detection | Spectral Signatures in Backdoor Attacks, NeurIPS 2018 | task/class 内部找异常 feature direction |
| Activation clustering | Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering, 2018 | 检查同类样本是否分裂成 clean/backdoor 两簇 |

### 5.2 Noisy-Label Robust Learning

| 防御方法 | 对应论文 | 可迁移到 CL 的位置 |
|---|---|---|
| Small-loss selection | Co-teaching, NeurIPS 2018 | 两个 learner 互相选 clean samples |
| Semi-supervised relabeling | DivideMix, ICLR 2020 | 把可疑样本当作 unlabeled 处理 |
| Sample refurbishing | SELFIE, ICML 2019 | 对疑似错误标签样本 relabel/refurbish |
| Early-learning regularization | ELR, NeurIPS 2020 | 防止模型后期记住噪声标签 |

### 5.3 Backdoor Detection / Removal

| 防御方法 | 对应论文 | 可迁移到 CL 的位置 |
|---|---|---|
| Trigger inversion | Neural Cleanse, IEEE S&P 2019 | 每个 task 后检查是否存在异常小 trigger |
| Entropy-based test detection | STRIP, ACSAC 2019 | 推理阶段检测 trigger 输入 |
| Pruning + fine-tuning | Fine-Pruning, RAID 2018 | task 后模型净化 |
| Backdoor unlearning | I-BAU, ICLR 2022 | 训练后反向寻找并 unlearn 后门 |
| Physical/local trigger saliency | SentiNet, IEEE S&P DL Security Workshop 2020 | 对图像局部 trigger 有用，EEG 需改造成时窗/频带 saliency |

---

## 6. Prompt / Architecture / Federated CL 相关攻击与防御

虽然本文重点是 replay 和 regularization，但以下方向也与 CL 安全有关。

| 类型 | 攻击目标 | 代表论文 | 防御启发 |
|---|---|---|---|
| Prompt-based CL | prompt pool, key-query matching, prompt selection | Backdoor Attack in Prompt-Based Continual Learning, AAAI 2025 | 检查 trigger 是否控制 prompt selection |
| Architecture-based CL | task-specific subnetworks, masks, prompts | Backdoor Attacks Against Incremental Learners, 2023 | 检查不同任务子网络是否共享后门路径 |
| Federated CL | client update, data supplier, aggregation | Towards a Defense against Backdoor Attacks in Continual Federated Learning | 适合多数据供应方/多用户 EEG 场景 |
| Continual adversarial defense | 连续出现的新攻击 | Defense without Forgetting, CVPR 2024；Adversarial Robust Memory-Based Continual Learner, ICCV 2025 | 学新防御时不忘旧防御 |

---

## 7. 本地目录论文的数据集与模型

以下表格对应本地 CL 相关目录中已经讨论过的论文。

### 7.1 CL 防御 / 鲁棒类

| 论文 | 使用数据集 | 使用模型 / Backbone | 方法类型 |
|---|---|---|---|
| 2021 ICCV Self-Purified Replay | MNIST, CIFAR-10, CIFAR-100, WebVision | MNIST 用 2-layer MLP；CIFAR/WebVision 用 ResNet-18 | Replay buffer purification |
| 2022 CVPR PuriDivER | CIFAR-10, CIFAR-100, mini-WebVision, Food-101N | CIFAR 用 ResNet-18；Food-101N 用 ResNet-32；WebVision 用 ResNet-34 | Online blurry CL + contaminated stream |
| 2023 NeurIPS Temporal Robustness | News Category Dataset | RoBERTa/BERT-base feature extractor + linear classifier | Temporal aggregation defense |
| 2025 AAAI DiffAdapt | Office, OfficeHome, miniDomainNet | ResNet-50；另测 VGG16, ViT-Base, ConvNeXt, MobileNet, ResNet-101 | Unlabeled adaptation trojan defense |
| 2026 arXiv Theory of CL Against Data Poisoning | CIFAR-100, CIFAR-10 | CIFAR-100: pretrained ViT + linear layer；CIFAR-10: CNN | Theory + regularization-based CL defense |

### 7.2 CL 攻击 / 漏洞类

| 论文 | 使用数据集 | 使用模型 / Backbone | 攻击对象 |
|---|---|---|---|
| 2020 IJCNN Targeted Forgetting and False Memory | Permuted MNIST, Split MNIST | 2-hidden-layer MLP | EWC |
| 2022 IJCNN Targeted Data Poisoning Against CLNN | Rotated MNIST, MNIST + Fashion-MNIST + KMNIST | 2-hidden-layer MLP | EWC, Online EWC, SI |
| 2023 arXiv Backdoor Attacks Against Incremental Learners | Permuted MNIST；Split MNIST/CIFAR-10/CIFAR-100 | benchmark 默认网络 | XdG, EWC, SI, LwF, DGR, ER, A-GEM, iCaRL 等 |
| 2023 arXiv PACOL | Rotated MNIST, Split MNIST, Split SVHN, Split CIFAR-10 | MLP, CNN, ResNet-20 | EWC, Online EWC, SI, DGR |
| 2023 ICML Poisoning Generative Replay | Split-MNIST, Split-CIFAR-10, FashionMNIST-MNIST, Permuted-MNIST, Split-EMNIST | SpinalVGG, ResNet, cWGAN-GP, cVAE | Generative replay |
| 2024 CVPR BrainWash | Split CIFAR-100, miniImageNet, tinyImageNet | ResNet-18；另测 RegNetX | EWC, MAS, RWALK, AFEC, ANCL, ER, ER-ACE |
| 2025 AAAI Backdoor Attack in Prompt-Based CL | ImageNet-R, CUB200；surrogate: TinyImageNet, CIFAR-100 | ViT-B/16 | L2P, DualPrompt, HiDe-Prompt, CODA-Prompt |
| 2025 USENIX Persistent Backdoor Attacks in CL | SplitMNIST, PermutedMNIST, SplitCIFAR10 | 5-layer CNN, ResNet-18 | SI, EWC, XdG, LwF, DGR, A-GEM |

---

## 8. 面向 EEG-CL 的研究切入点

### 8.1 如果 EEG-CL 使用 replay / memory

可能攻击链：

```text
污染 EEG trial / session
-> 污染样本进入 replay buffer
-> 后续任务反复 replay
-> 旧 subject/session/task 性能下降
```

推荐研究方向：

1. EEG replay buffer poisoning detection。
2. 基于时频特征、Riemannian covariance、通道拓扑特征的 buffer filtering。
3. task-balanced / subject-balanced replay sampler audit。
4. generative EEG replay 的样本一致性检查。
5. replay batch 对旧任务性能影响的 online verification。

对应可参考论文：

- Self-Purified Replay
- PuriDivER
- Amnesia
- Poisoning Generative Replay
- Are Exemplar-Based CIL Models Victim of Black-Box Poison Attacks?
- NLOCL
- Alternate Replay
- Continual CRUST

### 8.2 如果 EEG-CL 使用 regularization

可能攻击链：

```text
污染某个 session / subject / task
-> 当前任务梯度异常
-> EWC/SI/MAS 重要性估计或参数更新被误导
-> 错误知识被保护，旧知识被破坏
```

推荐研究方向：

1. task/session-level update anomaly detection。
2. robust Fisher / robust importance estimation。
3. 旧 subject / old session validation based rollback。
4. task vector 检测 poisoned session。
5. trusted anchor EEG trials 用于更新前后验证。

对应可参考论文：

- Targeted Forgetting and False Memory Formation
- Targeted Data Poisoning Attacks Against CLNNs
- PACOL
- BrainWash
- Single-Task Data Poisoning in Exemplar-Free CL
- Theory of CL Against Data Poisoning Attacks

### 8.3 EEG 场景中特别需要注意的问题

EEG 和图像不同，正常数据本身就有很强的 subject/session/domain shift。因此直接套用图像 outlier detection 容易误伤正常个体差异。

更合理的检测粒度可能是：

```text
trial-level detection
+ session-level detection
+ subject-level detection
+ task update-level detection
```

可能的 EEG trigger 形式：

- 特定时间窗扰动。
- 特定频带能量增强或抑制。
- 特定通道组合异常。
- 相位模式扰动。
- 跨 trial 的统计偏移。
- 对时频图或 covariance matrix 的隐式 perturbation。

因此，EEG-CL 防御最好不要只在原始 signal 上做检测，而应同时检查：

- raw waveform。
- time-frequency representation。
- spatial covariance / Riemannian feature。
- deep EEG encoder embedding。
- task update vector。
- replay buffer distribution。

---

## 9. 可以形成的论文研究问题

### 9.1 Replay-Based EEG-CL 防御问题

可提出的问题：

> 在 EEG continual learning 中，攻击者污染少量 trial/session 后，这些样本一旦进入 replay buffer，会被后续任务反复 replay，从而导致旧 subject/task 性能持续下降。如何在无标签或弱标签条件下检测并阻止 poisoned EEG samples 进入 buffer？

可能方案：

```text
EEG feature consistency
+ buffer purity/diversity selection
+ subject-balanced replay sampler
+ old-task validation rollback
```

### 9.2 Regularization-Based EEG-CL 防御问题

可提出的问题：

> 在 regularization-based EEG-CL 中，单个污染 session 可能改变当前任务更新方向，使 EWC/SI/MAS 的旧知识保护失效。如何检测异常 task update 并进行安全更新？

可能方案：

```text
task-vector anomaly detection
+ robust importance estimation
+ trusted anchor trials
+ update rejection / rollback
```

### 9.3 无标签 EEG-CL 防御问题

如果 EEG-CL 没有标签，可以考虑：

```text
pseudo-label confidence
+ temporal consistency
+ augmentation consistency
+ subject/session distribution monitoring
+ entropy / feature drift detection
```

可借鉴方向：

- DiffAdapt
- TTA robust adaptation methods
- Temporal Robustness against Data Poisoning
- semi-supervised noisy-label learning

---

## 10. 推荐阅读顺序

如果目标是尽快进入 EEG-CL data poisoning 防御研究，建议顺序如下：

1. 先读 CL 攻击基础：
   - Targeted Forgetting and False Memory Formation
   - Targeted Data Poisoning Attacks Against CLNNs
   - PACOL
   - BrainWash

2. 再读 replay/memory 攻击：
   - Poisoning Generative Replay
   - Amnesia
   - Are Exemplar-Based CIL Models Victim of Black-Box Poison Attacks?

3. 再读 CL 防御：
   - Self-Purified Replay
   - PuriDivER
   - NLOCL
   - Alternate Replay
   - Theory of CL Against Data Poisoning
   - Single-Task Data Poisoning in Exemplar-Free CL

4. 最后读通用防御：
   - Deep k-NN Defense
   - Spectral Signatures
   - Activation Clustering
   - Neural Cleanse
   - STRIP
   - Fine-Pruning
   - I-BAU

