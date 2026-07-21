# PACOL 论文分析与 TODO 回答

论文：*PACOL: Poisoning Attacks Against Continual Learners*

作者：Huayu Li, Gregory Ditzler。说明：arXiv 元数据写作 Huayu Li；本地 Markdown 与 arXiv TeX 正文写作 Huayu Liu，存在作者姓氏元数据不一致，本报告以 arXiv 官方元数据为主。

会议/期刊：arXiv 预印本

年份：2023

源码/项目：论文未发布明确标注为 PACOL 的完整官方代码。作者的 [2022 前驱仓库](https://github.com/HuayuLiArizona/Data-Poison-Incremental-Learning) 含核心梯度匹配实现，但不等同于 2023 PACOL 全量复现。

本报告依据：用户提供的 [本地 Markdown](</home/undefined/Desktop/IPhone/论文/2023arXiv-PACOL - Poisoning Attacks Against Continual Learners.md>)、[arXiv:2311.10919](https://arxiv.org/abs/2311.10919) 及其 TeX 源文件。

论文速览：

| Item | Content |
| --- | --- |
| 研究对象 | 通过未来任务中的少量训练样本，使持续学习模型定向遗忘一个旧任务 |
| 核心问题 | 能否把强但显眼的旧任务 label-flipping 攻击，伪装成未来任务中的 clean-label 小扰动 |
| 方法摘要 | 计算旧任务错误标签梯度，再优化未来任务输入，使其正确标签训练梯度与该恶意梯度匹配 |
| 任务/场景 | 监督式 domain/task incremental learning |
| 数据集 | Rotation-MNIST、Split-MNIST、Split-SVHN、Split-CIFAR10 |
| 受害方法 | EWC、Online EWC、SI、DGR |
| 主要指标 | 各任务最终准确率；防御过滤到的投毒样本比例 |
| 关键结论 | 1%-5% 的 clean-label 投毒可增加目标旧任务错误率，通常比 label flipping 更隐蔽但攻击更弱 |

## TODO 快速回答

### 核心问题是什么？

PACOL 问的是：模型已经学会旧任务后，攻击者能否只在未来任务里插入少量保持正确标签的扰动样本，让模型选择性忘记某个旧任务，同时避免旧图像配错误标签所带来的明显异常。

### 方法相比已有工作解决了什么弱点？

- 相比 backdoor/false memory，它不依赖测试时 trigger。
- 相比 label flipping，最终注入样本不改标签，也不直接注入旧任务图像。
- 相比只研究 i.i.d. 数据流的在线投毒，它显式利用任务之间的非平稳分布和 CL 遗忘机制。
- 它给出 white-box、gray-box 和 surrogate black-box 三种知识条件，并测试正则化与生成重放。

但梯度匹配本身并非全新技术；主要创新是把“旧任务错误标签梯度”选作 CL 遗忘的参考方向。

### 关键公式如何一步步推导？

\[
\text{最大化旧任务损失}
\rightarrow \text{label flipping 给出已知恶意更新}
\rightarrow \text{一次 SGD 的影响由梯度决定}
\rightarrow \text{让未来 clean-label 样本产生相同梯度}
\rightarrow \text{PGD 优化输入扰动}.
\]

详细推导见第四部分。

### 实验设置和结论是否可信？

作为漏洞证明，结论基本可信：四个数据集、四类 CL 受害者、三种模型知识、多个投毒比例，并有十次运行与简单防御实验。但外部有效性和复现性有限：自然图像上的效果较弱，算法伪代码存在更新符号冲突，缺少官方完整代码、标准 CL 遗忘指标、关键消融和现代防御。

## 第一部分 研究背景

持续学习模型依次学习多个任务。攻击者不必在目标旧任务出现时立即行动，因为未来任务训练仍会继续修改共享参数。只要未来任务中的少量样本产生了专门破坏旧决策边界的更新，模型就可能在其他任务看似正常的同时忘记目标任务。

最直接的攻击是把旧任务样本复制到未来训练流，并把标签翻错。其训练梯度会明确要求模型“反向学习”旧任务，因此攻击很强；但它需要旧样本、标签与内容明显矛盾，也容易被人工或离群检测发现。PACOL 的目标是保留这种恶意更新方向，同时把外观和标签伪装成未来任务的正常样本。

## 第二部分 历史发展

| 时间 | 代表工作/阶段 | 核心思想 | 尚未解决的问题 | 与本文关系 |
| --- | --- | --- | --- | --- |
| 2017-2020 | EWC、SI、DGR 等 CL | 正则化或重放缓解自然遗忘 | 默认数据流可信 | PACOL 攻击其训练数据假设 |
| 2020 | False Memory Formation | 在未来任务注入带 trigger 的错误记忆 | 需要 trigger，通常只在触发输入上生效 | PACOL 追求无 trigger 的旧任务遗忘 |
| 2022 | Targeted Data Poisoning Against CL | 在未来任务中注入旧任务错误标签样本 | 强但显眼，依赖旧数据 | PACOL 的直接前驱 |
| 2021-2023 | Clean-label gradient matching | 让投毒样本梯度匹配目标恶意梯度 | 主要研究离线模型或单任务目标 | PACOL 将其移植到 CL 遗忘 |
| 2023 | PACOL | 用未来任务 clean-label 扰动模仿旧任务 label-flip 梯度 | 仍需旧任务真实/辅助分布，算法细节不完整 | 本文 |
| 2024 | BrainWash | 模型反演旧任务，再直接双层优化全局遗忘 | 权限仍为白盒，成本高 | 去掉真实旧数据访问的同期后续方向 |
| 2025-2026 | STP、理论防御、sampler 攻击 | 弱权限 task poison、可防边界、replay 调度攻击 | 统一 benchmark 尚缺 | 将 PACOL 问题扩展到更现实威胁模型 |

## 第三部分 本文创新

### Novelty

1. 把 label flipping 解释为一个可复制的“旧知识破坏梯度”。
2. 让未来任务的正确标签样本通过输入扰动产生近似相同的参数梯度。
3. 在多个模拟训练状态上反复优化投毒样本，使其不只对单个 checkpoint 有效。
4. 给出白盒、灰盒和不同架构代理模型的黑盒迁移设定。

### Contributions

- 证明少量未来任务投毒可以定向损伤已学任务。
- 同时评估 EWC/Online EWC/SI 与生成重放 DGR。
- 将 label flipping 的攻击强度与 PACOL 的隐蔽性进行比较。
- 测试若干原始空间、深层特征和 t-SNE 空间的离群检测方法。

### Core Insight

如果两个训练样本集合在当前模型上产生近似相同的参数梯度，那么一次 SGD 后造成的参数位移也近似相同；因此可以把“旧任务错误标签”的破坏作用隐藏到“未来任务正确标签”的输入扰动中。

## 第四部分 方法详解

### 4.1 攻击目标与术语

- 目标任务 \(\mathcal D_\tau\)：攻击者希望模型遗忘的已学任务。
- 非目标任务 \(\mathcal D_{\tau+n}\)：未来到来的任务，投毒样本实际注入这里。
- Label-flipping poison：旧任务输入配错误标签。
- PACOL poison：未来任务输入加小扰动，标签保持正确。

整体目标写成：

\[
\max_{\mathcal D^{adv}_{\tau+n}}
\mathcal L_{\theta^*}(\mathcal D_\tau),
\qquad
\theta^*\in\arg\min_\theta
\sum_{n=1}^{T-\tau}
\mathcal L_\theta
(\mathcal D_{\tau+n}\cup\mathcal D^{adv}_{\tau+n}).
\]

外层希望旧任务损失变大，内层仍是模型在未来训练流上的正常经验风险最小化。

### 4.2 Label-flipping 如何产生遗忘方向

二分类时可令 \(Y_\tau^{adv}=-Y_\tau\)；多分类时论文使用随机偏移：

\[
Y_\tau^{adv}=(Y_\tau+z)\bmod(n+1).
\]

一次 SGD 更新可写为：

\[
\theta'
=\theta
-\eta\underbrace{\nabla_\theta
\mathcal L(f(X_{\tau+1};\theta),Y_{\tau+1})}_{g_{\mathrm{current}}}
-\eta\underbrace{\nabla_\theta
\mathcal L(f(X_\tau;\theta),Y_\tau^{adv})}_{g_{\mathrm{lf}}}.
\]

交叉熵对 logits 的梯度为 \(p-e_y\)。将真实标签 \(y\) 换成错误标签 \(y'\) 后，更新会提高 \(y'\) 的 logit，并相对压低原类别 \(y\)，因此破坏旧任务分类边界。

### 4.3 PACOL 如何伪装这个方向

从未来任务选出少量正常样本，初始化：

\[
X_{\tau+n}^{adv}=X_{\tau+n},
\qquad Y_{\tau+n}^{adv}=Y_{\tau+n}.
\]

最终标签不变。投毒样本产生的梯度是：

\[
g_{\mathrm{adv}}
=\nabla_\theta
\mathcal L(f(X_{\tau+n}^{adv};\theta),Y_{\tau+n}).
\]

希望满足：

\[
g_{\mathrm{adv}}\approx g_{\mathrm{lf}},
\]

于是优化：

\[
X_{\tau+n}^{adv}
=\arg\min_{\|X^{adv}-X\|_\infty\le\epsilon}
d(g_{\mathrm{adv}},g_{\mathrm{lf}}).
\]

最终注入的是“未来任务图像 + 小扰动 + 原始正确标签”。错误标签旧样本只用于生成参考梯度，不会交给受害者。

### 4.4 距离函数

平方欧氏距离：

\[
d_{\ell_2}(p,q)=\|p-q\|_2^2.
\]

负余弦相似度：

\[
d_{\cos}(p,q)
=-\frac{p^\top q}{\|p\|_2\|q\|_2}.
\]

\(\ell_2\) 同时匹配方向与幅值；负余弦主要匹配更新方向，对梯度尺度更不敏感。论文在 MNIST 系列使用 \(\ell_2\)，在 SVHN/CIFAR 使用负余弦。

论文的余弦展开式漏写了范数中的平方；上式是正确形式。

### 4.5 为什么优化输入需要二阶导

距离目标依赖参数梯度：

\[
H(X^{adv})=d(
\nabla_\theta L(X^{adv},Y),
g_{\mathrm{lf}}).
\]

更新输入需要：

\[
\nabla_{X^{adv}}H,
\]

其中包含 \(\nabla^2_{X,\theta}L\) 的混合二阶导。因而 PACOL 的每个 PGD 步比普通输入对抗样本更昂贵。

### 4.6 多模型状态优化

如果只在初始 \(\theta_0\) 上匹配梯度，模型训练几步后参数变化，投毒方向可能失效。PACOL 因此交替执行：

```text
在 theta_k 上计算旧任务错误标签梯度
-> 对输入做 S 次梯度匹配 PGD
-> 用生成的投毒样本更新一次模型，得到 theta_{k+1}
-> 在新参数状态重新匹配
-> 共循环 K 次
```

这相当于让 poison 覆盖一段模拟训练轨迹。

### 4.7 伪代码中的关键符号冲突

原始 TeX 把目标定义为：

\[
\mathcal H=\operatorname{dist}
(\Delta_\theta^{lf},\Delta_\theta^{adv}),
\]

正文要求最小化 \(\mathcal H\)，但算法写成：

\[
X^{adv}\leftarrow
\operatorname{clip}_{X,\epsilon}
\left(X^{adv}
+\alpha\operatorname{sign}(\nabla_X\mathcal H)\right),
\]

即对距离做梯度上升。若 \(\mathcal H\) 是平方 \(\ell_2\) 或负余弦距离，按正文应使用减号；若想用加号，则应将 \(\mathcal H\) 定义为待最大化的正相似度。由于没有 PACOL 完整官方代码，这一不一致会直接影响复现。

其他未说明清楚的细节：

- 正则化实验步长写作 \(2\epsilon/T\)，\(T\) 究竟指任务数还是 PGD 步数不清楚。
- DGR 实验没有报告投毒 PGD 步长。
- 1%-5% poison 是追加还是替换、如何抽样没有明确说明。
- 没有报告实际生成开销。

### 4.8 攻击者知识

| 设置 | 已知信息 | 不知道/不拥有 | 实际性质 |
| --- | --- | --- | --- |
| White-box | victim 参数、未来任务数据、目标旧任务真实数据 | EWC/SI importance、DGR 生成器细节 | 权限最强，直接在 victim 上生成 |
| Gray-box | victim 架构、未来任务数据、目标分布辅助数据 | victim 参数、真实目标训练集 | 训练同架构 surrogate |
| Black-box | 未来任务数据、目标分布辅助数据 | victim 参数和架构 | 用不同架构 surrogate 做迁移 |

论文所谓 black-box 不是查询式黑盒；它仍需要目标任务同分布辅助数据。

### 4.9 对正则化和 DGR 的关系

PACOL 推导时暂时忽略 CL 正则项与 replay，以便把一次更新简化为梯度匹配。攻击并不是通过直接操纵 Fisher、SI importance 或 DGR buffer 构造的，而是希望恶意梯度跨这些保护机制仍然保留足够影响。

对 DGR，当前投毒数据还会参与生成模型学习，因此可能进一步污染未来生成的记忆；论文实验展示了脆弱性，但没有分离“分类器参数被直接攻击”和“generator 被间接污染”各自的贡献。

## 第五部分 实验

| Item | Content |
| --- | --- |
| Datasets | Rotation-MNIST、Split-MNIST、Split-SVHN、Split-CIFAR10 |
| Target | 固定 Task 1 为目标旧任务；主要向 Tasks 4、5 注入 |
| Poison Ratios | 1%、3%、5% |
| Regularization Victims | EWC、Online EWC、SI，仅在 R-MNIST 上 |
| Replay Victim | DGR，在四个数据集上 |
| Knowledge | White-box、gray-box、black-box surrogate |
| Defenses | \(\ell_2\) centroid、one-class SVM、Isolation Forest、LOF |
| Metrics | 最终任务准确率、过滤出的 poison 比例 |

### 5.1 训练配置

正则化实验：两层、每层 400 单元的 MLP；每任务 5000 iterations；batch size 128；Adam，学习率 \(10^{-4}\)。PACOL 使用 \(K=10\)、\(S=40\)、\(\epsilon=25.5/255\)。

DGR：MNIST 用 CNN，自然图像用 ResNet20，black-box surrogate 用不同 CNN 或 VGG；WGAN-GP 负责生成重放。PACOL 使用 \(K=15\)、\(S=40\)、\(\epsilon=16/255\)。

### 5.2 5% white-box PACOL 的代表结果

| 受害者/数据 | Clean Task 1 | PACOL Task 1 | 下降 |
| --- | ---: | ---: | ---: |
| EWC / R-MNIST | 63.20 | 55.73 | 7.47 |
| Online EWC / R-MNIST | 64.80 | 52.34 | 12.46 |
| SI / R-MNIST | 54.46 | 42.41 | 12.05 |
| DGR / R-MNIST | 93.25 | 83.40 | 9.85 |
| DGR / S-MNIST | 87.12 | 65.66 | 21.46 |
| DGR / S-SVHN | 70.98 | 59.80 | 11.18 |
| DGR / S-CIFAR | 78.26 | 75.57 | 2.69 |

PACOL 在 MNIST/SVHN 上能产生明显定向遗忘，但在 S-CIFAR 上效果较弱。Label flipping 通常更强，例如 DGR/R-MNIST 5% 时 Task 1 从 93.25 降至 16.53，但其旧图像与错误标签更容易检测。

### 5.3 防御结果如何解读

论文显示简单离群检测常难发现 PACOL。例如 R-MNIST 子集上，防御允许删除全数据 5% 时，原始空间 \(\ell_2\) 检测可找出接近全部 label-flip poison，却只能找出少量 PACOL poison。

但“PACOL 不可检测”并不成立：S-CIFAR 的 deep-feature one-class SVM 在部分设置能过滤超过一半 PACOL。论文只报告命中 poison 的比例，没有同时报告误删干净样本、过滤后准确率或 ROC，因此不能完整评估防御实用性。

### 5.4 实验可信度判断

可信之处：

- 四个 benchmark 覆盖简单数字和自然图像。
- 同时测试正则化 CL 与生成式 replay。
- 给出三种攻击知识条件和多个投毒比例。
- 正则化实验报告十次运行及 95% 区间。
- 与 label flipping 的攻击强度/隐蔽性形成直观对照。

限制：

- 正则化方法只在 R-MNIST 上测试，不能说明自然图像中的普遍性。
- S-CIFAR 上 PACOL 仅降低约 1-4 个百分点，主张明显依赖数据集。
- 表格中并非所有 white-box 都最强，攻击随投毒比例也并非严格单调。
- 没有 random bounded noise、通用 clean-label poison 等关键基线。
- 没有 \(K\)、\(S\)、距离函数、\(\epsilon\)、目标任务选择的系统消融。
- 只报告最终每任务准确率，没有 ACC、BWT、forgetting measure 等统一 CL 指标。
- “不可感知”主要靠样例图，没有 PSNR/SSIM、用户研究或物理约束。
- 伪代码更新方向、步长和数据混合方式不清楚，且没有完整代码。

综合判断：内部有效性中等；外部有效性偏低；复现性偏低到中等。

## 第六部分 与已有工作的比较

| Work | Key Idea | Difference From PACOL | Limitation |
| --- | --- | --- | --- |
| False Memory / Backdoor | trigger 与错误标签建立关联 | PACOL 无测试时 trigger，直接诱导旧任务遗忘 | 后门行为通常只在 trigger 输入上激活 |
| Label Flipping | 旧任务输入配错误标签后注入未来任务 | PACOL 只把它作为参考梯度，最终 poison 保持未来任务标签 | 强但显眼，需旧任务数据 |
| Witches' Brew 类 gradient matching | 匹配目标梯度构造 clean-label poison | PACOL 把目标梯度设为旧任务 label-flip 梯度 | 原方法通常不是 CL 跨任务遗忘 |
| Poisoning Generative Replay | 定向污染 replay generator | PACOL 同时攻击正则化与 DGR，构造不依赖 generator 细节 | PACOL 对 DGR 的具体污染路径未分离 |
| BrainWash | 模型反演 + 双层最大化所有旧任务损失 | PACOL 可做 surrogate gray/black-box，并定向一个旧任务 | PACOL 仍需真实旧数据或目标分布辅助数据 |

## 第七部分 局限性

### 作者提到的局限

- 实验只覆盖特定形式的 concept forgetting。
- 需要进一步研究数据、算法和架构层面的稳健 CL。
- CL 攻击缺乏统一 benchmark。

### 作者没有充分展开但值得注意的局限

1. White-box 需要完整目标旧任务数据；gray/black-box 仍需同分布辅助数据。
2. “黑盒”没有查询或真实部署约束，本质是 surrogate transfer。
3. 目标旧任务固定为 Task 1，未来 Tasks 4/5 被污染，任务位置泛化不足。
4. 梯度匹配推导忽略 CL 正则与 replay，缺乏攻击仍成立的理论保证。
5. 输入优化需要二阶导，计算成本和可扩展性未量化。
6. 目标梯度来自随机 label flip，未研究更优 target label 或 class-specific 策略。
7. 基本离群防御不足以代表现代 poison detection。
8. 算法内部存在减号/加号矛盾和余弦公式排版错误。

### 未来仍未解决的问题

- 不接触旧任务或辅助旧分布的定向梯度攻击。
- 更现实的 query-only 或模型 API 黑盒。
- 对 replay buffer、生成器和分类器贡献的因果分解。
- EEG、语音和时序传感器中的可感知/生理约束。
- 攻击检测后的模型恢复，而不只是丢弃可疑样本。

## 第八部分 最新发展

截至日期：2026-07-20

| 方向/工作 | 最新状态 | 与本文关系 | 证据来源 |
| --- | --- | --- | --- |
| PACOL 代码 | 未发现论文或作者主页明确链接的 PACOL 完整仓库 | 复现仍依赖论文细节与前驱实现 | [arXiv](https://arxiv.org/abs/2311.10919)、[作者主页](https://gditzler.github.io/publications/) |
| 2022 前驱实现 | 作者仓库公开梯度匹配、错误标签参考梯度和 PGD | 可验证 PACOL 核心思想，但不含全部 2023 实验 | [GitHub](https://github.com/HuayuLiArizona/Data-Poison-Incremental-Learning) |
| BrainWash | CVPR 2024，引入模型反演和双层全局遗忘 | 去掉 PACOL 对真实旧任务数据的依赖，但只给白盒攻击 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.html) |
| STP | CoLLAs 2025，只访问当前 task，提出 task-vector 检测 | 进一步降低攻击权限 | [arXiv:2507.04106](https://arxiv.org/abs/2507.04106) |
| CL poisoning theory | 2026，分析正则化 CL 的攻击/防御边界 | 将 PACOL 的经验发现推进到理论防御 | [arXiv:2606.29841](https://arxiv.org/abs/2606.29841) |

工业/开源采用情况：没有发现 PACOL 被主流工业系统公开采用；研究用途主要是安全评测和防御设计。

Open Problems：弱权限定向遗忘、统一 benchmark、时序/物理约束、训练轨迹可扩展优化、可信 task 验证和攻击后恢复。

## 第九部分 科研价值

PACOL 最有价值的洞见是把“恶意遗忘”转化为梯度几何问题：攻击者不一定需要让 poison 在输入空间看起来像目标旧任务，只要它在参数空间产生相同的更新方向即可。这一视角适合迁移到 BCI，因为跨受试者、跨 session 的输入分布可以完全不同，但共享网络参数仍可能受到同方向梯度影响。

如果继续写下一篇论文：

| 方向 | 具体问题 | 为什么有价值 | 难度/资源 | 可能做法 |
| --- | --- | --- | --- | --- |
| EEG clean-label PACOL | 新受试者 EEG 能否匹配旧受试者错误伪标签梯度 | 直接对应无标签/弱标签 BCI 适应安全 | 中 | 用 teacher pseudo-label 代替真实标签，加入 PSD/协方差约束 |
| Past-data-free PACOL | 不访问旧数据时如何得到恶意参考梯度 | 放宽最弱环节 | 中高 | 模型反演、prototype inversion、生成先验 |
| Low-rank gradient matching | 只匹配关键层或子空间 | 降低二阶导成本 | 中 | 最后层、Fisher top-k、随机投影、K-FAC 近似 |
| Task-selective attack | 只遗忘某一受试者/类别且保持其他任务 | 比全局 availability attack 更有研究价值 | 高 | class-conditioned gradient target、多目标约束 |
| Defense | 检测未来任务梯度是否接近旧任务破坏方向 | 与攻击目标直接对应 | 中 | trusted anchor gradient、task-vector angle、influence score |

### EEG 中不应直接照搬像素约束

可把 \(\ell_\infty\) 替换或补充为：

- 每通道幅值、SNR 和设备量程约束。
- 频带功率与 PSD 距离。
- 时间平滑、带通范围和瞬态伪迹约束。
- 通道协方差/SPD 几何距离。
- 电极空间图 Laplacian 平滑。
- 保持临床/BCI 事件相关特征的语义一致性。

## 第十部分 Roadmap

### 推荐阅读顺序

| Order | Paper/Topic | Why Read It |
| ---: | --- | --- |
| 1 | EWC、SI、DGR | 理解 PACOL 的受害机制 |
| 2 | False Memory Formation | 理解早期 CL 恶意记忆/遗忘 |
| 3 | 2022 Targeted Data Poisoning | 理解 label flipping 前驱 |
| 4 | Witches' Brew / gradient matching | 理解 PACOL 的优化工具 |
| 5 | PACOL | 理解 clean-label 定向遗忘 |
| 6 | BrainWash | 理解无旧数据代理与全局双层目标 |
| 7 | STP 与 2026 theory | 理解更现实权限和防御边界 |

### 知识树

```text
PACOL
├── Attack target
│   └── Forget one old task
├── Reference direction
│   └── Old-task label-flip gradient
├── Poison carrier
│   └── Future-task clean-label samples
├── Optimizer
│   ├── L2 / cosine gradient matching
│   ├── PGD in input space
│   └── Multiple simulated model states
├── Threat models
│   ├── White-box
│   ├── Gray-box surrogate
│   └── Different-architecture surrogate
└── Victims
    ├── EWC / Online EWC / SI
    └── DGR
```

### 时间线

```text
旧样本错误标签
-> 抽象出恶意参数梯度
-> 用未来 clean-label 样本匹配该梯度
-> 模型反演去除旧数据需求
-> task-level 弱权限攻击
-> task verification 与理论防御
```

### 后续论文/主题

- PACOL 伪代码与前驱实现的严格复现核验。
- 对 BrainUICL/EEG UICL 的伪标签梯度攻击。
- 低成本、低秩、关键层 gradient matching。
- 目标任务梯度方向的在线监测与可信 anchor 防御。
- generator、buffer 和 classifier 三类攻击路径的消融。

## Sources

- 本地论文：[用户 Markdown](</home/undefined/Desktop/IPhone/论文/2023arXiv-PACOL - Poisoning Attacks Against Continual Learners.md>)；[工作区 PDF](</home/undefined/Desktop/bci/papers/CL_Attacks/2023arXiv-PACOL - Poisoning Attacks Against Continual Learners.pdf>)
- 官方论文页：[arXiv:2311.10919](https://arxiv.org/abs/2311.10919)、[PDF](https://arxiv.org/pdf/2311.10919)
- 官方代码/项目：未找到 PACOL 完整官方仓库；[作者 2022 前驱代码](https://github.com/HuayuLiArizona/Data-Poison-Incremental-Learning)、[核心文件](https://github.com/HuayuLiArizona/Data-Poison-Incremental-Learning/blob/main/AdvAttack/poison_attack.py)
- 其他来源：[BrainWash](https://openaccess.thecvf.com/content/CVPR2024/html/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.html)、[STP](https://arxiv.org/abs/2507.04106)、[2026 theory](https://arxiv.org/abs/2606.29841)
