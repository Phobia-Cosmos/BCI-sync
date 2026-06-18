# 近年 TTA/CTTA 隐私保护论文与 Nature 研究角度

> 结论：TTA 隐私保护是很新的交叉方向。严格意义上“直接把 TTA 更新做成隐私保护”的论文还不多；更多工作是从 source-free、federated、local/on-device、medical data governance、model update leakage 等角度间接处理隐私。Nature 系列目前很少直接使用“privacy-preserving test-time adaptation”这个术语，主要从医疗 AI 的隐私保护协作学习、联邦评测、数据分布漂移、神经技术伦理等角度研究。

## 1. 为什么 TTA 本身会产生隐私问题

TTA/Test-Time Adaptation 的核心是在推理时用目标域测试样本更新模型。它的隐私风险在于：

- 更新后的模型参数依赖历史测试样本；
- 连续 TTA/CTTA 会把一个用户或机构的长期数据分布写入模型；
- 如果共享 adapted model、BN statistics、feature statistics、memory bank 或 prototypes，可能泄露目标域信息；
- 在医疗、BCI、移动端和联邦场景中，测试数据往往就是最敏感的真实用户数据。

因此，TTA 的隐私问题不只是“训练数据隐私”，而是 **test-time/user-time data privacy**：模型部署后接触到的用户数据是否会被参数、统计量或记忆模块带走。

## 2. 直接相关论文

### 2.1 Private and Stable Test-Time Adaptation with Differential Privacy, 2026

- 作者：Li Z, Tang Q, Lecuyer M, Shelhamer E.
- 形式：arXiv:2606.01908, 2026.
- 链接：https://arxiv.org/abs/2606.01908
- 关键词：TTA, Differential Privacy, Tent, EATA, SAR, DeYO, COME, continual setting.

这是目前最直接的“隐私保护 TTA”论文之一。论文指出 TTA 在测试时更新模型，模型参数会依赖所有过去测试输入，因此产生对测试数据的隐私风险。作者把 Tent、EATA、SAR、DeYO、COME 等 TTA 方法改造成 DP 版本，用 per-sample gradient clipping 和 Gaussian noise 控制隐私泄露。

对我们课题的启发：

- 传统 TTA 默认测试数据可被用来更新模型，但没有保护测试用户隐私；
- BCI/EEG 中每个 test-time stream 可能对应一个 disabled 用户，隐私风险更强；
- DP-TTA 可以作为 baseline 或对比，但 DP 可能牺牲 EEG 小样本/低 SNR 性能，也不能处理用户撤权后的模型残留，因此仍可引出 MU。

### 2.2 Federated Test-Time Adaptive Face Presentation Attack Detection with Dual-Phase Privacy Preservation, FG 2021

- 作者：Shao R, Zhang B, Yuen P C, Patel V M.
- 会议：FG 2021.
- 链接：https://arxiv.org/abs/2110.12613
- 关键词：Federated learning, test-time adaptation, face presentation attack detection, privacy.

论文面向 face presentation attack detection。由于真实/攻击人脸图像不能在数据源之间直接共享，训练阶段用 FL 聚合模型更新，测试阶段用 entropy minimization 做 TTA 来缓解 unseen attack/domain gap。

对我们课题的启发：

- 这是“训练阶段隐私 + 测试阶段适配”的早期组合；
- 隐私保护主要来自 FL 不共享原始数据，TTA 用于部署时域适配；
- 可类比 BCI：源 EEG 不共享，目标用户在线适配。

### 2.3 Calibration-free Online Test-Time Adaptation for EEG Motor Imagery Decoding, 2023

- 作者：Wimpff M, Döbler M, Yang B.
- 形式：arXiv:2311.18520, 2023.
- 链接：https://arxiv.org/abs/2311.18520
- 关键词：EEG, BCI, online TTA, motor imagery, source-free, calibration-free.

论文把 online TTA 用于 EEG motor imagery decoding，强调不访问 source data，因此保留隐私；同时不需要 session/subject-specific calibration data。

对我们课题的启发：

- 这是和 BCI 最直接相关的 TTA 论文之一；
- 它把隐私主要定义为“不访问源数据”；
- 但没有系统处理 target stream 在 TTA 后被模型记住的问题，也没有处理用户撤权/MU。

### 2.4 Day-Night Adaptation for Medical Image Segmentation, 2024/2025

- 作者：Chen Z, Ye Y, Pan Y, Zhang J, Zhang Y, Xia Y.
- 形式：arXiv:2410.13472, 2024.
- 链接：https://arxiv.org/abs/2410.13472
- 关键词：medical image segmentation, source-free adaptation, TTA, privacy, clinical deployment.

论文指出 medical centres 之间共享数据会带来隐私风险，因此 SFDA 和 TTA 依赖 target data、无需 source data，是保护隐私的部署范式。它提出 day-night adaptation：白天对每个测试样本做 prompt adaptation，夜间复用当天测试数据做更稳定的模型更新。

对我们课题的启发：

- 医疗场景中 TTA 被视为源数据不可访问下的隐私友好适配；
- 但它也暴露出新问题：如果夜间复用测试数据/记忆库，target user privacy 如何保证？
- 这和 BCI 的持续在线数据流非常接近。

### 2.5 FedCTTA: A Collaborative Approach to Continual Test-Time Adaptation in Federated Learning, IJCNN 2025

- 作者：Rajib R H, Iftee M A R, Hossain M S, et al.
- 会议：IJCNN 2025.
- 链接：https://arxiv.org/abs/2505.13643
- 关键词：federated learning, continual TTA, privacy-preserving, feature sharing risk.

论文指出 FL 适合隐私敏感应用，但 FL 模型部署后仍会遇到 distribution shift；现有 federated TTA 可能因为 feature sharing 带来隐私风险。FedCTTA 避免直接交换 features，改用随机噪声样本上的输出分布做 similarity-aware aggregation。

对我们课题的启发：

- CTTA 中的 memory/statistics/prototypes 是潜在隐私泄露点；
- 可以借鉴“不要共享 feature，改共享更弱的输出统计”的思想；
- 对 BCI 可扩展为不共享原始 EEG、不共享特征、不共享用户特异 BN 统计。

### 2.6 pFedBBN: Personalized Federated Test-Time Adaptation with Balanced Batch Normalization, 2025

- 作者：Iftee M A R, Hasan S M A, Hossain M S, et al.
- 形式：arXiv:2511.18066, 2025.
- 链接：https://arxiv.org/abs/2511.18066
- 关键词：personalized federated TTA, class imbalance, balanced BN, privacy.

论文面向 federated test-time adaptation 中的 class imbalance 和 domain shift，强调无需 labeled/raw client data，实现 personalized inference without compromising privacy。

对我们课题的启发：

- 个性化 TTA 与隐私天然冲突：越个性化，越可能编码用户分布；
- EEG/BCI 中 target user class distribution 也可能不均衡，例如 MI 类别、疲劳状态、错误反馈事件。

### 2.7 Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based BCIs, 2026

- 作者：Li S, Ouyang J, Cui Z, Wang Z, Jia T, Wan F, Wu D.
- 形式：arXiv:2601.07556, 2026.
- 链接：https://arxiv.org/abs/2601.07556
- 关键词：EEG, BCI, TTA, backpropagation-free, privacy risk, resource-constrained device.

论文指出 EEG-BCI 中现有 TTA 往往依赖反向传播更新参数，会带来计算开销、隐私风险和噪声敏感性；因此提出无需反向传播的 test-time transformation/aggregation。

对我们课题的启发：

- BCI 设备上 TTA 不仅要适配，还要轻量、稳定、隐私友好；
- 不更新参数可以降低模型携带用户信息的风险，但不等于提供形式化隐私保证；
- 可以作为“无参数更新 TTA”方向的对比。

## 3. 与传统 TTA/CTTA 的关系

这些隐私工作通常建立在经典 TTA/CTTA 之上：

- Tent, ICLR 2021：测试时最小化预测熵，更新 BN affine parameters。
- CoTTA, CVPR 2022：面向连续变化目标域，处理 error accumulation 和 catastrophic forgetting。
- EATA, ICML 2022：选择可靠样本并用 Fisher regularization 缓解遗忘。
- SAR, ICLR 2023：用 sharpness-aware/reliable adaptation 提升鲁棒性。
- DeYO, COME 等：后续被 DP-TTA 论文纳入隐私化改造对象。

隐私保护 TTA 的核心不是重新发明适配，而是回答：**在测试数据用于更新模型之后，如何防止测试用户信息被模型参数、统计量、记忆库或共享更新泄露？**

## 4. Nature 系列目前从哪些角度研究相关问题

### 4.1 直接结论

截至目前，Nature Portfolio 中很少有论文直接以“privacy-preserving test-time adaptation / private TTA / CTTA privacy”为中心。Nature 上更常见的是相邻问题：

1. 医疗数据不能集中共享，因此需要 federated learning / secure AI；
2. 医疗 AI 部署后会遭遇自然数据分布漂移，需要外部验证和持续监测；
3. 模型要到真实机构本地运行或评测，而不是把患者数据拿出来；
4. 神经技术/BCI 必须保护 privacy、identity、agency、equality。

这些角度可以支撑我们写 TTA 隐私背景，但不能写成“Nature 已经系统研究 privacy-preserving TTA”。

### 4.2 Nature 角度 A：医疗 AI 数据隐私与联邦学习

- Rieke et al., *The future of digital health with federated learning*, npj Digital Medicine, 2020.
- 链接：https://www.nature.com/articles/s41746-020-00323-1

该文指出医疗数据存在 silos，privacy concerns restrict access；FL 通过不交换原始数据来训练模型。文章还指出，即使匿名化也不总能保护隐私，患者/机构可以保留数据控制和撤销访问权。

对 TTA 的意义：

- TTA/SFDA 与 FL 共享现实约束：source data 或 hospital data 不能随意访问；
- 如果部署阶段还要适配，只能把模型带到数据旁边，而不是把数据集中起来。

### 4.3 Nature 角度 B：安全、隐私保护和攻击向量

- Kaissis et al., *Secure, privacy-preserving and federated machine learning in medical imaging*, Nature Machine Intelligence, 2020.
- 链接：https://www.nature.com/articles/s42256-020-0186-1

该文总结医疗影像中的 secure/federated/privacy-preserving AI，并明确讨论 attack vectors 和隐私保护技术，如 FL、DP、encrypted data 等。

对 TTA 的意义：

- TTA 是部署后模型更新；它同样会产生 model inversion、membership inference、gradient/parameter leakage 等问题；
- Nature 的角度是“医疗 AI 必须同时考虑数据利用和数据保护”，可以迁移到 TTA。

### 4.4 Nature 角度 C：临床部署中的自然数据漂移

- Zhang et al., *Shifting machine learning for healthcare from development to deployment and from models to data*, Nature Biomedical Engineering, 2022.
- 链接：https://www.nature.com/articles/s41551-022-00898-y

该综述强调 healthcare ML 从开发到部署时，data pipeline 和 natural data shifts 会导致性能下降，需要部署层面的数据与模型治理。

对 TTA 的意义：

- Nature 关注的是“部署后数据分布变化”，这正是 TTA/CTTA 要解决的问题；
- 但 Nature 文章通常没有把解决方案限定为 TTA，而是从数据治理、外部验证、监测和联邦学习等角度展开。

### 4.5 Nature 角度 D：联邦评测/模型到数据，而不是数据到模型

- Karargyris et al., *Federated benchmarking of medical artificial intelligence with MedPerf*, Nature Machine Intelligence, 2023.
- 链接：https://www.nature.com/articles/s42256-023-00652-2

MedPerf 提出 federated evaluation：把模型发到不同医疗机构本地评测，只上传聚合指标，避免提取患者数据到中心。论文强调 federated evaluation 可以量化模型跨机构泛化，同时保护 data privacy 和 model IP。

对 TTA 的意义：

- 这与 TTA 非常接近：模型部署到本地后面对真实目标域数据；
- 下一步自然问题是：如果模型不仅本地评测，还要本地适配，适配后的参数/统计量如何保护？

### 4.6 Nature 角度 E：神经技术与 BCI 的 privacy/identity/agency

- Yuste et al., *Four ethical priorities for neurotechnologies and AI*, Nature, 2017.
- 链接：https://www.nature.com/articles/551159a

该文明确提出 AI 和 BCI 必须尊重和保护 privacy、identity、agency、equality。

对 BCI-TTA 的意义：

- EEG/BCI 的 TTA 是对用户神经数据流的在线适配，直接涉及 neural privacy；
- disabled-aided BCI 中，TTA 会持续接触最真实的使用数据，因此隐私风险比离线 benchmark 更强。

## 5. 对我们 EEG + SFDA + MU + CTTA 课题的启发

可以形成如下研究缺口：

1. TTA/CTTA 能解决 EEG/BCI 部署时的跨 session、跨状态漂移；
2. 但 TTA 会把测试用户数据写入参数、BN 统计、memory bank 或 prototypes；
3. 现有 privacy TTA 主要有三类：DP-TTA、federated TTA、source-free/local TTA；
4. 这些方法多关注“不访问源数据/不共享原始数据”，但较少处理“适配后模型如何忘记某个用户、某段 stream、某类隐私属性”；
5. 因此 MU 可以补足 TTA/SFDA 的撤权问题：当 disabled 用户撤回数据、某个 session/domain 失效或敏感属性需要移除时，模型不仅要继续适配，还要删除残留影响。

适合写入 introduction 的中文论点：

> 近年来 TTA/CTTA 被用于解决模型部署后的目标域漂移，但其隐私问题开始受到关注。TTA 在推理阶段利用目标用户数据更新模型，因此模型参数、BN 统计、memory bank 或共享特征可能携带测试用户信息。最新 DP-TTA 工作已明确指出，TTA 参数依赖过去测试输入，会产生 test-time data privacy 风险；federated TTA 和 medical TTA 则从不共享原始数据、不访问源数据的角度降低隐私泄露。Nature 系列虽然尚未系统研究 privacy-preserving TTA，但在医疗 AI 中已经从 federated learning、federated evaluation、数据漂移和 secure/privacy-preserving AI 等角度提出相同的部署约束：真实敏感数据不能集中共享，模型必须在本地数据上验证、适配并受治理。因此，在 EEG/BCI 中研究 privacy-preserving CTTA/SFDA/MU 具有明确现实动机。

## 6. 参考文献草稿

[1] Li Z, Tang Q, Lecuyer M, Shelhamer E. Private and stable test-time adaptation with differential privacy[J]. arXiv preprint arXiv:2606.01908, 2026.

[2] Shao R, Zhang B, Yuen P C, Patel V M. Federated test-time adaptive face presentation attack detection with dual-phase privacy preservation[C]//IEEE International Conference on Automatic Face and Gesture Recognition. 2021.

[3] Wimpff M, Döbler M, Yang B. Calibration-free online test-time adaptation for electroencephalography motor imagery decoding[J]. arXiv preprint arXiv:2311.18520, 2023.

[4] Chen Z, Ye Y, Pan Y, Zhang J, Zhang Y, Xia Y. Day-night adaptation: An innovative source-free adaptation framework for medical image segmentation[J]. arXiv preprint arXiv:2410.13472, 2024.

[5] Rajib R H, Iftee M A R, Hossain M S, et al. FedCTTA: A collaborative approach to continual test-time adaptation in federated learning[C]//International Joint Conference on Neural Networks. 2025.

[6] Iftee M A R, Hasan S M A, Hossain M S, et al. pFedBBN: A personalized federated test-time adaptation with balanced batch normalization for class-imbalanced data[J]. arXiv preprint arXiv:2511.18066, 2025.

[7] Li S, Ouyang J, Cui Z, Wang Z, Jia T, Wan F, Wu D. Backpropagation-free test-time adaptation for lightweight EEG-based brain-computer interfaces[J]. arXiv preprint arXiv:2601.07556, 2026.

[8] Wang D, Shelhamer E, Liu S, Olshausen B, Darrell T. Tent: Fully test-time adaptation by entropy minimization[C]//International Conference on Learning Representations. 2021.

[9] Wang Q, Fink O, Van Gool L, Dai D. Continual test-time domain adaptation[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[10] Niu S, Wu J, Zhang Y, et al. Efficient test-time model adaptation without forgetting[C]//International Conference on Machine Learning. 2022.

[11] Rieke N, Hancox J, Li W, et al. The future of digital health with federated learning[J]. npj Digital Medicine, 2020, 3: 119. DOI: 10.1038/s41746-020-00323-1.

[12] Kaissis G A, Makowski M R, Rückert D, Braren R F. Secure, privacy-preserving and federated machine learning in medical imaging[J]. Nature Machine Intelligence, 2020, 2: 305-311. DOI: 10.1038/s42256-020-0186-1.

[13] Dayan I, Roth H R, Zhong A, et al. Federated learning for predicting clinical outcomes in patients with COVID-19[J]. Nature Medicine, 2021, 27: 1735-1743. DOI: 10.1038/s41591-021-01506-3.

[14] Zhang A, Xing L, Zou J, Wu J C. Shifting machine learning for healthcare from development to deployment and from models to data[J]. Nature Biomedical Engineering, 2022, 6: 1330-1345. DOI: 10.1038/s41551-022-00898-y.

[15] Karargyris A, Umeton R, Sheller M J, et al. Federated benchmarking of medical artificial intelligence with MedPerf[J]. Nature Machine Intelligence, 2023, 5: 799-810. DOI: 10.1038/s42256-023-00652-2.

[16] Yuste R, Goering S, Agüera y Arcas B, et al. Four ethical priorities for neurotechnologies and AI[J]. Nature, 2017, 551: 159-163. DOI: 10.1038/551159a.
