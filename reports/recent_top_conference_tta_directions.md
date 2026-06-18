# 近年顶会 Test-Time Adaptation / Continual Test-Time Adaptation 研究方向梳理

> 范围：主要关注 CVPR、ICCV、ECCV、ICLR、ICML、NeurIPS、AAAI 等 AI 顶会中的 Test-Time Training/Adaptation (TTT/TTA)、Online TTA、Continual TTA、Test-Time Prompt Tuning、Source-free/Medical/Multimodal TTA。结论不局限隐私安全，而是总结近几年 TTA 领域主要在做什么。

## 1. 一句话总结

近年 TTA 的主线已经从“测试时熵最小化让分类更准”扩展为更完整的部署问题：**如何在没有源数据、没有目标标签、目标分布持续变化、测试流可能非独立同分布、模型可能遗忘/崩溃、目标数据可能隐私敏感、基础模型/视觉语言模型越来越大**的情况下，让模型在真实部署中持续适配。

可以概括为八个方向：

1. **基础 TTA/TTT 方法**：测试时自监督、熵最小化、BN/统计量更新。
2. **稳定性与防崩溃**：避免错误伪标签、确认偏差、灾难遗忘、熵坍塌。
3. **Continual / Online TTA**：目标域随时间连续变化，模型要持续适配。
4. **样本选择与可靠性估计**：只用可靠样本更新，过滤异常/高风险样本。
5. **参数高效/轻量 TTA**：只调 BN、adapter、LoRA、prompt 或少量参数。
6. **视觉语言/基础模型 TTA**：CLIP/SAM/VLM 的 test-time prompt tuning 和跨域泛化。
7. **任务扩展**：从图像分类扩展到语义分割、检测、3D、视频、医学、EEG/BCI。
8. **隐私/联邦/源自由 TTA**：不访问源数据、本地适配、DP 或 federated TTA。

## 2. 方向 A：基础 TTA/TTT，从测试时自监督到熵最小化

### 2.1 Test-Time Training with Self-Supervision, ICML 2020

- 论文：Sun Y, Wang X, Liu Z, Miller J, Efros A, Hardt M. *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*.
- 会议：ICML 2020.
- 核心：训练时加入自监督辅助任务，测试时用单个测试样本的自监督损失更新模型。
- 意义：把“测试时更新模型”正式推到主流视野。

### 2.2 Tent, ICLR 2021

- 论文：Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. *Tent: Fully Test-Time Adaptation by Entropy Minimization*.
- 会议：ICLR 2021.
- 核心：测试时最小化预测熵，只更新 BN affine parameters。
- 意义：成为后续 TTA/CTTA 大量工作的基础 baseline。

### 2.3 MEMO, NeurIPS 2022

- 论文：Zhang M M, Levine S, Finn C. *MEMO: Test Time Robustness via Adaptation and Augmentation*.
- 会议：NeurIPS 2022.
- 核心：对单个测试样本做多增强，最小化增强后一致预测熵。
- 意义：强调 single-sample TTA，不依赖 batch statistics，更适合在线场景。

**趋势判断**：早期 TTA 主要证明“测试时更新能抗 domain shift”。但这类方法容易出错，因为没有标签监督，错误更新会被不断放大。

## 3. 方向 B：稳定性、防崩溃、防遗忘

### 3.1 CoTTA, CVPR 2022

- 论文：Wang Q, Fink O, Van Gool L, Dai D. *Continual Test-Time Domain Adaptation*.
- 会议：CVPR 2022.
- 核心：面向连续变化目标域，用 teacher-student、随机恢复和增强平均减少 error accumulation。
- 解决问题：目标域不是一次性变化，而是连续漂移；直接熵最小化会逐步崩。

### 3.2 EATA, ICML 2022

- 论文：Niu S, Wu J, Zhang Y, Chen Y, Zheng S, Zhao P, Tan M. *Efficient Test-Time Model Adaptation without Forgetting*.
- 会议：ICML 2022.
- 核心：选择可靠且非冗余样本更新，并用 Fisher regularization 保持源知识。
- 解决问题：避免无用样本拖慢更新，避免适配时遗忘源域能力。

### 3.3 SAR, ICLR 2023

- 论文：Niu S, Wu J, Zhang Y, Wen Z, Chen Y, Zhao P, Tan M. *Towards Stable Test-Time Adaptation in Dynamic Wild World*.
- 会议：ICLR 2023.
- 核心：用 sharpness-aware/reliable adaptation，避免 noisy samples 和 entropy collapse。
- 解决问题：真实世界测试流中样本质量不稳定，直接熵最小化可能使模型过度自信和崩溃。

### 3.4 RoTTA, CVPR 2023

- 论文：Yuan L, Xie B, Li S. *Robust Test-Time Adaptation in Dynamic Scenarios*.
- 会议：CVPR 2023.
- 核心：维护类别平衡记忆库和时间衰减统计，提升动态场景下的鲁棒性。
- 解决问题：test stream 存在时间相关和类别不平衡，不能假设每个 batch 都代表目标分布。

**趋势判断**：这是 TTA 最核心的一条线。顶会越来越关注“不要适配坏了”，包括 error accumulation、catastrophic forgetting、unstable entropy minimization、class imbalance、non-i.i.d. stream。

## 4. 方向 C：Continual / Online TTA，应对持续变化的目标域

### 4.1 Continual Test-Time Domain Adaptation, CVPR 2022

CoTTA 标志着 TTA 从 single target domain 走向 continual shift。现实部署中目标域可能按天气、地点、设备、用户状态、时间不断变化，因此不能只做一次性 adaptation。

### 4.2 Learning to Adapt to Online Streams with Distribution Shifts, NeurIPS 2023

- 论文：Boudiaf M, Mueller R, Ayed I B, Bertinetto L. *Learning to Adapt to Online Streams with Distribution Shifts*.
- 会议：NeurIPS 2023.
- 核心：把 TTA 放入 online stream 设定，强调数据流中的分布变化和在线适配策略。

### 4.3 ActMAD / 动态 classifier 适配类工作, ICLR/CVPR 2024 附近

- 核心思想：不是大规模更新整个模型，而是在测试时动态调整 classifier、prototype 或少量模块。
- 解决问题：连续适配中大模型更新代价高，且更容易遗忘。

**趋势判断**：CTTA 更贴近真实部署。对 BCI/EEG 来说，用户疲劳、注意、session、电极阻抗变化天然就是 continual stream，因此 CTTA 比传统离线 DA 更贴近场景。

## 5. 方向 D：样本选择、伪标签可靠性与 uncertainty-aware TTA

很多 TTA 方法的问题来自“错误样本也参与更新”。因此近年大量工作关注：哪些样本可以更新，哪些应该跳过。

代表思想：

- 低熵样本更可靠，但过低熵也可能是错误自信；
- 多增强一致性可衡量稳定性；
- 类别平衡可防止模型偏向高频类；
- Fisher/源模型约束可防止远离源知识；
- OOD/outlier 样本不应参与适配。

代表论文：

- EATA, ICML 2022：可靠且非冗余样本选择。
- SAR, ICLR 2023：sharpness-aware 过滤不可靠更新。
- RoTTA, CVPR 2023：类别平衡记忆和时间衰减。
- DeYO, ICLR 2024 附近：面向可靠目标样本筛选与去伪相关区域。

**趋势判断**：TTA 的关键已经从“怎么更新”变成“何时更新、用谁更新、更新多少”。这对 EEG 很重要，因为 EEG 试次噪声大，错误伪标签会快速污染模型。

## 6. 方向 E：参数高效、轻量化、边缘设备 TTA

TTA 发生在部署端，通常受算力、延迟、电池、隐私和稳定性限制。因此近年很多工作只更新少量参数。

常见做法：

- 只更新 BN affine parameters 或统计量；
- 只更新 classifier/head；
- adapter / LoRA / prompt tuning；
- 不反向传播的 test-time transformation；
- prototype/statistics-level adaptation。

代表论文：

- Tent, ICLR 2021：只更新 BN 参数。
- EATA, ICML 2022：高效样本选择，减少更新成本。
- ViDA / visual domain adapter 类工作，ICLR/CVPR 2024 附近：用 adapter 做参数高效视觉域适配。
- Backpropagation-free TTA for lightweight EEG-based BCIs, 2026：直接面向轻量 EEG-BCI，避免反向传播。

**趋势判断**：TTA 与 PEFT 越来越结合。大模型时代不可能在测试端全量更新，adapter/LoRA/prompt/prototype 会是主流。

## 7. 方向 F：视觉语言模型与基础模型的 Test-Time Prompt Tuning

随着 CLIP、SAM、VLM 成为通用 backbone，TTA 也从“更新 CNN 参数”扩展到“测试时调 prompt”。

### 7.1 TPT, NeurIPS 2022

- 论文：Shu M, Nie W, Huang D A, Yu Z, Goldstein T, Anandkumar A, Xiao C. *Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models*.
- 会议：NeurIPS 2022.
- 核心：测试时为 CLIP 优化 prompt，使增强视图预测一致。
- 意义：把 TTA 引入 vision-language foundation model。

### 7.2 DiffTPT, ICCV/CVPR 2023-2024 附近

- 核心：利用 diffusion 生成多样增强或视图，提升 test-time prompt tuning 的鲁棒性。
- 解决问题：普通增强覆盖不足，prompt tuning 需要更丰富的测试时视图。

### 7.3 VLM/FMs 上的 TTA 新问题

- prompt 是否会过拟合单个测试样本；
- 类别文本先验是否可靠；
- zero-shot 和 test-time adaptation 如何平衡；
- 大模型不可全量更新，必须 prompt/adapter 化。

**趋势判断**：这是近两年 TTA 很热的方向。BCI 若使用 CLIP 对齐脑信号、EEG-to-image/video 或语义解码，也可借鉴 test-time prompt/adapter adaptation。

## 8. 方向 G：从分类扩展到分割、检测、3D、视频和自动驾驶

早期 TTA 多在 CIFAR-C/ImageNet-C 分类上验证，近年顶会开始把 TTA 推到更复杂任务。

### 8.1 语义分割

研究问题：城市街景、医学图像、自动驾驶场景中，测试域可能来自新城市、新天气、新相机、新医院。

常见策略：

- 测试时归一化统计更新；
- pseudo-label self-training；
- entropy minimization；
- memory/prototype alignment；
- source-free segmentation adaptation。

### 8.2 目标检测与开放世界识别

研究问题：检测任务有局部框、类别、背景和置信度，错误伪标签更复杂。TTA 需要同时处理分类和定位不确定性。

### 8.3 3D 点云/LiDAR

研究问题：不同传感器、天气、密度、扫描线数量导致 domain shift。TTA 可用于 point cloud classification/segmentation 和自动驾驶 LiDAR。

### 8.4 视频/时序任务

研究问题：视频帧相关性强，不能按 i.i.d. batch 处理；需要利用时间一致性，同时防止错误随时间累积。

**趋势判断**：TTA 正在从 classification benchmark 走向真实系统任务。这与 EEG/BCI 一致：BCI 是时序流，不是静态图像分类。

## 9. 方向 H：医疗、BCI、生理信号中的 TTA

### 9.1 医疗影像 TTA

- 目标：不同医院、扫描仪、协议、患者群体造成 domain shift；不能共享患者数据。
- 代表方向：source-free/TTA segmentation、federated TTA、day-night adaptation、test-time normalization。
- 隐私意义：医疗场景天然不能访问源数据，TTA/SFDA 更符合部署。

### 9.2 EEG/BCI TTA

- 目标：跨 subject、跨 session、跨天、跨电极佩戴、跨疲劳/注意状态适配。
- 代表论文：Calibration-free Online Test-Time Adaptation for EEG Motor Imagery Decoding, 2023；Backpropagation-free TTA for Lightweight EEG-based BCIs, 2026。
- 关键挑战：EEG 低 SNR、小样本、非平稳、标签不可得、在线延迟限制、用户隐私。

### 9.3 睡眠/情绪/可穿戴生理信号

- 目标：个体差异和长期漂移强；每个用户重新标注不现实。
- 可用方法：online BN adaptation、pseudo-label refinement、source-free individual adaptation、personalized TTA。

**趋势判断**：医疗和 BCI 是 TTA 最有现实必要性的场景之一，因为 source data/privacy/calibration burden 同时存在。

## 10. 方向 I：隐私、联邦、源自由 TTA

虽然你这次问不局限隐私，但这个方向与我们课题相关，需要单独保留。

代表论文：

- Private and Stable Test-Time Adaptation with Differential Privacy, 2026：用 DP 改造 Tent/EATA/SAR/DeYO/COME。
- Federated Test-Time Adaptive Face Presentation Attack Detection, FG 2021：训练用 FL，测试用 TTA。
- FedCTTA, IJCNN 2025：联邦 CTTA，避免 feature sharing。
- Day-Night Adaptation for Medical Image Segmentation, 2024：源数据不可访问的医疗 TTA/SFDA。

核心问题：

- test-time 用户数据会不会被参数/统计量/memory bank 记住；
- adapted model 是否能被共享；
- 多用户 CTTA 如何避免互相泄露；
- 用户撤权后如何删除适配影响。

**趋势判断**：这是正在形成的新方向。对 EEG/BCI 来说，privacy-preserving CTTA + MU 是合理切入点。

## 11. 对我们课题最有用的研究缺口

如果要从 EEG + SFDA + MU + CTTA 写论文，可以把 TTA 顶会趋势转成下面几个 gap：

1. **稳定性 gap**：TTA 容易因伪标签错误、噪声样本和连续漂移崩溃；EEG 低 SNR 会放大问题。
2. **个性化 gap**：disabled/BCI 用户需要长期个性化，TTA 正好解决 target user/session drift。
3. **隐私 gap**：TTA 会把测试用户数据写入参数、BN 或 memory，现有 BCI TTA 多只说 source-free，不处理 target-time privacy。
4. **撤权 gap**：现有 TTA 关注 adaptation performance，很少考虑用户撤回某段 stream 后如何 unlearn。
5. **轻量 gap**：真实 BCI 设备算力有限，不能全量反向传播更新大模型。
6. **时序 gap**：BCI 是连续流，目标分布和标签分布会随疲劳、注意、康复变化，不能用静态 batch TTA 假设。

适合汇报的一句话：

> 近年顶会 TTA 的重点已经从“测试时熵最小化提升鲁棒性”转向“真实部署中的持续适配”：包括 CTTA 稳定性、防遗忘、样本可靠性、参数高效、基础模型 prompt tuning、医疗/联邦/隐私场景。BCI 与这些趋势高度一致，因为 EEG 部署天然是无标签、在线、非平稳、用户特异且隐私敏感的 test-time stream。

## 12. 代表论文清单

[1] Sun Y, Wang X, Liu Z, Miller J, Efros A A, Hardt M. Test-time training with self-supervision for generalization under distribution shifts[C]//International Conference on Machine Learning. 2020.

[2] Wang D, Shelhamer E, Liu S, Olshausen B A, Darrell T. Tent: Fully test-time adaptation by entropy minimization[C]//International Conference on Learning Representations. 2021.

[3] Zhang M M, Levine S, Finn C. MEMO: Test time robustness via adaptation and augmentation[C]//Advances in Neural Information Processing Systems. 2022.

[4] Wang Q, Fink O, Van Gool L, Dai D. Continual test-time domain adaptation[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[5] Niu S, Wu J, Zhang Y, Chen Y, Zheng S, Zhao P, Tan M. Efficient test-time model adaptation without forgetting[C]//International Conference on Machine Learning. 2022.

[6] Niu S, Wu J, Zhang Y, Wen Z, Chen Y, Zhao P, Tan M. Towards stable test-time adaptation in dynamic wild world[C]//International Conference on Learning Representations. 2023.

[7] Yuan L, Xie B, Li S. Robust test-time adaptation in dynamic scenarios[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[8] Shu M, Nie W, Huang D A, Yu Z, Goldstein T, Anandkumar A, Xiao C. Test-time prompt tuning for zero-shot generalization in vision-language models[C]//Advances in Neural Information Processing Systems. 2022.

[9] Feng C, Zhong Y, Jie Z, Chu X, Ren H, Wei X, Xie W, Ma L. Prompt distribution learning[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[10] Boudiaf M, Mueller R, Ayed I B, Bertinetto L. Parameter-free online test-time adaptation[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[11] Boudiaf M, Mueller R, Ayed I B, Bertinetto L. Learning to adapt to online streams with distribution shifts[C]//Advances in Neural Information Processing Systems. 2023.

[12] Döbler M, Marsden R A, Yang B. Robust mean teacher for continual and gradual test-time adaptation[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[13] Wimpff M, Döbler M, Yang B. Calibration-free online test-time adaptation for electroencephalography motor imagery decoding[J]. arXiv preprint arXiv:2311.18520, 2023.

[14] Li Z, Tang Q, Lecuyer M, Shelhamer E. Private and stable test-time adaptation with differential privacy[J]. arXiv preprint arXiv:2606.01908, 2026.

[15] Li S, Ouyang J, Cui Z, Wang Z, Jia T, Wan F, Wu D. Backpropagation-free test-time adaptation for lightweight EEG-based brain-computer interfaces[J]. arXiv preprint arXiv:2601.07556, 2026.
