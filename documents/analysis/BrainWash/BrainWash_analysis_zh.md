# BrainWash 论文分析与 TODO 回答

论文：*BrainWash: A Poisoning Attack to Forget in Continual Learning*

作者：Ali Abbasi, Parsa Nooralinejad, Hamed Pirsiavash, Soheil Kolouri

会议/期刊：CVPR 2024

年份：2024（arXiv 首次提交于 2023-11-20）

源码/项目：[官方代码](https://github.com/mint-vu/Brainwash)；[官方数据、checkpoint 与反演样本](https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main)

本报告依据：用户提供的 [本地 Markdown](</home/undefined/Desktop/IPhone/论文/2024CVPR-BrainWash A Poisoning Attack to Forget in Continual Learning.md>)、工作区内 CVPR 正式版 PDF、官方代码与公开资源。需要特别注意：本地 Markdown 对应较早版本，缺少 CVPR 定稿中的 ER/ER-ACE、额外投毒基线和 RegNetX 实验。

论文速览：

| Item | Content |
| --- | --- |
| 研究对象 | 持续学习模型在训练阶段遭受 clean-label 数据投毒后产生的恶意遗忘 |
| 核心问题 | 能否只污染最新任务的数据，使模型同时忘掉多个旧任务，而攻击者不持有旧任务数据 |
| 方法摘要 | 模型反演恢复旧任务代理集，再通过截断双层优化学习最新任务的逐样本扰动 |
| 任务/场景 | 监督式图像分类；multi-head task-incremental 为主，正式版补充 single-head replay |
| 数据集 | 10-Split CIFAR-100、10-Split miniImageNet、20-Split tinyImageNet |
| 主要指标 | Backward Transfer（BWT）、最新任务干净测试准确率 |
| 关键结论 | 新任务中的有界扰动可以显著放大旧任务遗忘；模型反演数据可接近真实旧数据的攻击效果 |
| 开放资源 | 官方代码、数据、checkpoint、反演样本；但代码未完整覆盖论文全部方法 |

## TODO 快速回答

### 核心问题是什么？

BrainWash 研究的是：攻击者能否只控制即将到来的最新任务训练数据，在不知道旧任务数据、受害者具体 CL 算法和超参数的情况下，诱使模型在学习最新任务时大规模忘记旧任务。

### 相比已有工作解决了什么弱点？

此前的定向遗忘、label flipping 或 gradient matching 攻击往往需要旧任务真实样本或同分布辅助数据。BrainWash 用模型反演从白盒模型中构造旧任务代理集，再把“训练后旧任务损失最大”直接写成双层目标，因此去掉了真实旧数据访问要求，并可同时攻击多个旧任务。Cautious 版本还显式兼顾最新任务的干净准确率。

### 关键公式如何一步步推导？

逻辑链是：

\[
\text{没有旧数据}
\rightarrow \text{模型反演得到旧任务代理集}
\rightarrow \text{模拟在投毒新任务上更新模型}
\rightarrow \text{最大化更新后旧任务代理损失}
\rightarrow \text{将元梯度反传到新任务扰动}.
\]

详细推导见第四部分。

### 实验设置和结论是否可信？

作为“该攻击面确实存在”的 proof-of-concept，证据较强：覆盖多个规模的数据集、五种正则化方法、ER/ER-ACE、两种攻击模式、多个噪声预算和多项消融。但对现实隐蔽性、跨模态泛化和统计稳定性的证据不足：主表没有均值方差，\(\epsilon=0.3\) 较大，Cautious 模式仍经常严重损伤最新任务准确率，且攻击成本和白盒权限都很高。

### 论文到底讲什么？攻击基于正则化吗？针对无标签数据吗？

- 论文讲的是如何把“破坏旧知识的参数更新方向”编码进最新任务的输入扰动。
- 攻击不是依据受害者的 EWC/MAS 等正则项构造。其内层只模拟普通 fine-tuning，甚至明确不知道受害者使用的 CL 正则。
- 模型反演中使用 TV、\(\ell_2\)、BN 统计等正则，但这是输入先验，不是 CL 正则。
- 原论文是监督分类攻击，攻击者需要最新任务的标签；最终投毒样本保留正确标签。它不是无标签攻击。
- BrainUICL 的无标签个体流需要把交叉熵内层改成伪标签、自监督或一致性损失，不能直接照搬原方法。

## 第一部分 研究背景

持续学习模型按任务或数据流顺序更新。学习新任务时，当前数据产生的梯度可能覆盖旧任务所依赖的参数，形成灾难性遗忘。CL 方法通常通过 replay、参数正则化或结构隔离来维持稳定性，但这些机制默认新任务数据是可信的。

BrainWash 反过来利用稳定性-可塑性矛盾：只要攻击者能控制一次新任务训练数据，就可以把新任务梯度设计成对旧知识特别有害。模型为了保持可塑性仍必须吸收一部分新数据更新；正则或 replay 虽然减弱这种更新，却不保证其方向是安全的。

一个典型 CL 训练流程是：

```text
已完成任务 1...t-1
-> 保存旧模型、参数重要性或 replay memory
-> 任务 t 数据到来
-> 用当前任务损失 + CL 保护机制训练
-> 更新模型与保护状态
-> 进入任务 t+1
```

BrainWash 插入在“任务 \(T\) 数据到来”与“受害者训练任务 \(T\)”之间。它不修改旧 checkpoint，而是修改最新任务的训练输入。

## 第二部分 历史发展

| 时间 | 代表工作/阶段 | 核心思想 | 尚未解决的问题 | 与本文关系 |
| --- | --- | --- | --- | --- |
| 2017-2019 | EWC、SI、MAS、ER、DGR | 用参数约束或重放缓解正常遗忘 | 默认训练流可信 | BrainWash 攻击这些稳定机制依赖的可信数据假设 |
| 2020 | Targeted Forgetting / False Memory | 在未来任务插入后门或错误记忆 | 常依赖 trigger，目标行为与正常输入分离 | 提出“恶意遗忘”攻击目标 |
| 2022 | Targeted Data Poisoning Against CL | 用旧任务数据、错误标签或梯度构造定向遗忘 | 需要旧任务信息，隐蔽性有限 | BrainWash 引用的直接前驱之一 |
| 2023 | Poisoning Generative Replay | 污染生成式重放机制 | 集中于 generator/replay 特定结构 | 展示 replay 同样存在遗忘攻击面 |
| 2023 | PACOL | 让新任务 clean-label 样本梯度匹配旧任务错误标签梯度 | 仍需要真实旧数据或旧分布辅助数据 | 与 BrainWash 同期，攻击构造思路不同 |
| 2024 | BrainWash | 模型反演 + 双层元投毒，直接最大化旧任务损失 | 白盒、计算昂贵、视觉场景为主 | 本文 |
| 2025 | Single-Task Poisoning（STP） | 只访问当前任务，用标准 corruption 破坏后续训练 | 攻击更现实但控制目标较粗 | 进一步削弱攻击者权限 |
| 2026 | CL poisoning theory、Amnesia | 理论化可防边界；攻击 replay sampler 而非像素 | 完整统一威胁模型仍缺失 | 将研究推向理论防御和更窄权限攻击 |

PACOL 于 2023-11-18 提交，BrainWash 于 2023-11-20 提交。BrainWash 未引用 PACOL 预印本，但引用了同一研究线的 2022 IJCNN 前驱，因此不应简单描述为“BrainWash 改进 PACOL”。

## 第三部分 本文创新

### Novelty

1. 用模型反演代替真实旧任务数据，构造攻击者内部的旧知识代理集。
2. 将“模型学完最新任务后，所有旧任务损失上升”写成 clean-label 双层投毒目标。
3. 提出 Reckless 与 Cautious 两种外层目标，后者显式维持最新任务干净性能。
4. 攻击者的内层模型无需复现受害者的 EWC、MAS、ER 等真实训练目标，仍能迁移到多种 CL 方法。

### Contributions

- 在三个视觉 CL benchmark 上攻击五种正则化方法。
- CVPR 正式版补充 ER、ER-ACE 单头 replay 实验。
- 消融模型反演、正则系数、注入率、扰动幅度、任务长度与骨干架构。
- 发布代码、checkpoint、反演样本和数据资源。

### Core Insight

只要最新任务样本产生的训练梯度经过一次模型更新后会增大旧任务损失，这些样本就是遗忘型投毒；旧任务真实数据并非必需，只要有足够反映旧决策边界的代理样本即可。

## 第四部分 方法详解

### 4.1 Multi-head、single-head 与 BrainUICL

论文的主设定把模型写成共享 backbone \(f(x;\theta)\) 和每个任务独立分类头 \(h_t(\cdot;\psi_t)\)：

\[
z_t=h_t(f(x;\theta);\psi_t)\in\mathbb R^{K_t}.
\]

“第 \(t\) 个 head 的 logits”就是该任务分类器输出的 \(K_t\) 个未归一化分数。Multi-head 通常对应 task-incremental learning，推理时知道 task ID，所以选择对应 head。Single-head 则让所有任务共享一个输出层，常用于 class-incremental 或 domain-incremental learning。

BrainUICL 把新受试者视为连续到来的 domain，语义类别并不随受试者增加，而且论文强调无需修改模型结构，因此通常沿用同一分类器；这就是它没有 BrainWash 主设定中的多个任务 head。BrainWash CVPR 定稿也在 ER/ER-ACE 上补充了 single-head 实验，所以多 head 不是攻击原理的必要条件，但会让旧任务类别和任务身份更容易从模型中读取。

### 4.2 威胁模型

| 条件 | BrainWash 假设 |
| --- | --- |
| 攻击时机 | 受害者已经完成任务 \(1,\ldots,T-1\)，即将学习任务 \(T\) |
| 模型权限 | 白盒访问结构与参数 |
| 数据权限 | 拥有最新任务 \(D_T\) 的干净、有标签训练数据 |
| 不拥有 | 旧任务真实数据、受害者具体 CL 算法和超参数 |
| 操作能力 | 对最新任务样本添加范数受限、逐样本扰动 |
| 最终标签 | 保持 \(y_T^i\) 不变，属于 clean-label poisoning |
| 目标 | 让共享模型参数更新后破坏旧任务知识 |

Reckless 攻击者不关心最新任务性能；Cautious 攻击者假设受害者会监控最新任务验证准确率，因此在攻击强度与当前性能间折中。

### 4.3 Model Inversion 如何生成旧任务代理集

对旧任务 \(t\)，攻击者从 head 输出维度得知类别数 \(K_t\)，随机抽取目标 one-hot 标签 \(\hat y_t^i\)，并在冻结的白盒模型上优化随机输入：

\[
\{\hat x_t^i\}_{i=1}^{M}
=\arg\min_{\{x^i\}}
\sum_{i=1}^{M}
\operatorname{CE}\!\left(
h_t(f(x^i;\theta^*_{1:T-1});\psi_t^*),
\hat y_t^i
\right)
+\sum_{i=1}^{M}R_{\mathrm{prior}}(x^i)
+\alpha_fR_{\mathrm{feat}}(\{x^i\},\theta^*_{1:T-1}).
\]

输入先验为：

\[
R_{\mathrm{prior}}(x)
=\alpha_{\mathrm{TV}}R_{\mathrm{TV}}(x)
+\alpha_{\ell_2}\|x\|_2.
\]

BN 特征统计约束为：

\[
R_{\mathrm{feat}}
=\sum_l\|\mu_l(X)-m_l\|_2^2
+\sum_l\|\sigma_l^2(X)-v_l\|_2^2.
\]

各项作用：

| 项 | Why | How | Failure Mode |
| --- | --- | --- | --- |
| 分类交叉熵 | 让代理样本承载指定旧类知识 | 提高目标类 logit | 可生成模型高置信但不自然的模式 |
| TV | 抑制像素级高频震荡 | 邻域差分惩罚 | 不适用于所有模态 |
| \(\ell_2\) | 控制输入能量 | 限制整体幅值 | 不能保证语义真实性 |
| BN 统计 | 贴近训练时内部特征分布 | 匹配每层 running mean/variance | 无 BN 或统计混合多个任务时信息变弱 |

“随机目标标签”不是随机错误标签。生成前随机选一个旧类；一旦选定，优化就要求对应 head 把合成输入判为这个特定类。官方代码从 `torch.rand` 图像开始，默认每个旧任务反演 128 个样本并优化 10000 步。

这些代理样本只供攻击者估计旧任务损失，不会加入受害者训练集，也不会写入 replay buffer。

### 4.4 Reckless 双层目标

定义最新任务投毒数据：

\[
\widetilde D_T(\delta)
=\{(x_T^i+\delta^i,y_T^i)\}_{i=1}^{N_T},
\qquad \|\delta^i\|_\infty\le\epsilon.
\]

内层模拟受害者学完投毒任务后的参数：

\[
(\tilde\theta(\delta),\tilde\psi_T(\delta))
=\arg\min_{\theta,\psi_T}
\sum_i
\mathcal L(x_T^i+\delta^i,y_T^i;\theta,\psi_T).
\]

外层最大化旧代理集损失：

\[
\delta^*
=\arg\max_{\|\delta^i\|_\infty\le\epsilon}
\sum_{t=1}^{T-1}\sum_{j=1}^{M}
\mathcal L(
\hat x_t^j,\hat y_t^j;
\tilde\theta(\delta),\psi_t^*
).
\]

旧 head \(\psi_t^*\) 固定；攻击主要通过改变共享 backbone \(\theta\) 破坏旧表征。

### 4.5 Cautious 双层目标

Cautious 模式增加最新任务干净损失的负项：

\[
J_{\mathrm{cautious}}(\delta)
=L_{\mathrm{old}}(\tilde\theta(\delta))
-\eta L_{T}^{\mathrm{clean}}
(\tilde\theta(\delta),\tilde\psi_T(\delta)).
\]

最大化该目标等价于：增大旧任务损失，同时减小最新任务在未扰动数据上的损失。\(\eta\) 越大，越偏向保持当前任务性能，但攻击强度通常下降。

### 4.6 一步截断展开与元梯度

完整反传全部 SGD 训练轨迹代价过高。令 \(w=(\theta,\psi_T)\)，只展开一步：

\[
w^{(1)}(\delta)
=w^{(0)}-\alpha\nabla_wL_{\mathrm{in}}(w^{(0)},\delta).
\]

外层目标 \(J(w^{(1)}(\delta))\) 对扰动的梯度为：

\[
\nabla_\delta J
=-\alpha
\left(\nabla_{\delta,w}^{2}L_{\mathrm{in}}\right)^{\!\top}
\nabla_{w^{(1)}}J.
\]

这说明攻击不是直接追求“当前样本被误分类”，而是在求一种输入，使其训练梯度经过参数更新后最大程度损坏旧任务。论文把方法称作 first-order approximation，准确说是 \(k=1\) 的截断双层展开；自动微分仍需计算混合二阶导或 Hessian-vector product。

扰动可按投影梯度上升更新：

\[
\delta\leftarrow
\Pi_{[-\epsilon,\epsilon]}
\left(\delta+\beta\operatorname{sign}(\nabla_\delta J)\right).
\]

官方代码等价地最小化负的旧任务损失，再对扰动做梯度下降。

### 4.7 实际攻击流程

```text
输入：训练到 T-1 的模型、最新任务有标签数据
1. 对每个旧任务/旧类执行模型反演，得到代理集
2. 为最新任务每个样本初始化独立扰动
3. 从旧 checkpoint 和随机新 head 开始
4. 在投毒当前 batch 上做一次可微参数伪更新
5. 在更新后的模型上计算旧代理集损失
6. Cautious 模式再加入最新任务干净损失
7. 将外层梯度反传到扰动并投影到范数球
8. 重复优化，最后把带原标签的投毒数据交给受害者
9. 受害者用自己的 CL 算法正常训练任务 T
```

### 4.8 受害 CL 算法的常规流程

正则化式 CL 的统一形式为：

\[
\theta^*_{1:T}
=\arg\min_\theta
L_T(\theta)
+\lambda R_{\mathrm{CL}}(\theta,\theta^*_{1:T-1}).
\]

| 方法 | 旧知识如何表示 | 训练新任务时如何保护 | 主要特点 |
| --- | --- | --- | --- |
| EWC | 对角 Fisher \(F_i\) | \(\frac\lambda2\sum_iF_i(\theta_i-\theta_i^*)^2\) | 任务结束点估计重要性，经典但近似较粗 |
| MAS | 输出对参数的敏感度 \(\Omega_i\) | 对高敏感参数加二次约束 | 重要性可用无标签输入估计 |
| RWALK | Fisher + 训练路径贡献 | 在线累积重要性并约束参数移动 | 同时考虑 forgetting 与 intransigence |
| AFEC | 旧模型重要性 + 当前任务经验最优模型 | 在保旧知识与吸收当前知识间双向约束 | 主动放弃产生负迁移的旧知识 |
| ANCL | 旧模型 + 自由学习新任务的 auxiliary network | 主模型同时靠近旧模型与辅助网络 | 用辅助网络调节稳定性-可塑性 |
| ER | 真实旧样本 buffer | 当前 batch 与 memory batch 联合训练 | 简单有效，但 buffer 是额外攻击面 |
| ER-ACE | ER buffer + asymmetric CE | 当前流只与当前类别竞争，replay 用完整 CE | 减少新类训练对旧类 logits 的压制 |

BrainWash 并不读取这些正则项来生成攻击。内层只做普通 fine-tuning，因此攻击对不同 CL 算法的迁移是实验现象，而非精确复制受害者优化器的结果。

### 4.9 Replay 版本中的 memory 权限

CVPR 正式版用 single-head ER/ER-ACE 和大小为 1000 的 memory 评估两种情况：

- 无 memory 访问：攻击者继续使用模型反演得到旧代理集。
- 有只读 memory 访问：直接用 buffer 中旧样本估计旧任务目标，攻击更强。
- 不考虑写 memory；作者认为写权限会形成更简单但不同的威胁模型。

在论文设定中，攻击直接改变的是模型参数，不是 buffer。若系统把被污染的新任务样本随后写入 buffer，攻击可能进一步持久化，但这不是本文正式评估的机制。

## 第五部分 实验

| Item | Content |
| --- | --- |
| Datasets | 10-Split CIFAR-100、10-Split miniImageNet、20-Split tinyImageNet |
| Backbone | 主实验 ResNet-18；补充 RegNetX-1.6GF |
| CL Methods | EWC、MAS、RWALK、AFEC、ANCL；正式版另含 ER、ER-ACE |
| Metrics | BWT 与最新任务干净准确率 |
| Main Budgets | \(\epsilon\in\{0.1,0.3\}\)，输入缩放到 \([0,1]\) |
| Baselines | Clean、Uniform noise；正式版另含 Unlearnable Examples、DeepConfuse、MetaPoison |
| Ablations | \(\lambda\)、注入率、噪声幅值、模型反演、任务长度、骨干架构 |
| Reproducibility | 有官方代码和公开资产，但未覆盖 ANCL、ER、ER-ACE 等全部正式版实验 |

### 5.1 代表结果

以 CIFAR-100/EWC 为例：

| 设置 | BWT | 最新任务准确率 |
| --- | ---: | ---: |
| Clean | -5.2 | 68.3 |
| Cautious, \(\epsilon=0.3\) | -24.7 | 42.2 |
| Reckless, \(\epsilon=0.3\) | -29.1 | 25.5 |

BWT 越负表示旧任务遗忘越严重。该结果说明攻击有效，但也显示 Cautious 并未真正保持接近 clean 的最新任务性能。

模型反演消融：

| 旧任务目标数据 | Forgetting |
| --- | ---: |
| Clean training | 5.2 |
| 无先验的反演数据 | 23.6 |
| 带先验的反演数据 | 28.8 |
| 真实旧数据 | 28.16 |

带先验反演数据与真实旧数据效果接近，支持“代理集只需反映旧决策边界，不必视觉上完美重建”的核心主张。

### 5.2 Replay 结果

| 方法 | Clean BWT/Acc | BrainWash（MI） | BrainWash（读 memory） |
| --- | ---: | ---: | ---: |
| ER | -18.9 / 71.2 | -23.1 / 53.8 | -27.7 / 49.5 |
| ER-ACE | -9.8 / 80.1 | -16.6 / 55.1 | -17.0 / 55.9 |

只读 memory 能增强攻击，但即使没有 memory 权限，模型反演仍能攻击 replay learner。

### 5.3 与普通投毒基线

在 CIFAR-100/EWC、\(\epsilon=0.3\) 上：

| 方法 | BWT / Acc |
| --- | ---: |
| Clean | -5.2 / 68.3 |
| Unlearnable Examples | -21.4 / 3.7 |
| DeepConfuse | -19.42 / 10.0 |
| MetaPoison | -22.94 / 35.3 |
| BrainWash | -29.1 / 25.5 |

BrainWash 的 BWT 最差，说明更聚焦旧任务遗忘；但最新任务性能同样严重下降。普通 availability poison 更倾向于直接使当前任务不可学。

### 5.4 实验可信度判断

可信之处：

- 数据集规模从 CIFAR-100 扩展到 mini/tinyImageNet。
- 覆盖正则化和 exemplar replay 两类 CL。
- 同时报告旧任务遗忘与当前任务性能，避免只展示单一攻击指标。
- 模型反演、噪声预算、注入比例、正则强度和架构消融与论文主张直接相关。
- 官方代码与反演资产提高了可检查性。

限制：

- 主表未报告多随机种子的均值、标准差或显著性检验。
- \(\epsilon=0.3\) 在 \([0,1]\) 图像上相当于最高约 \(76.5/255\) 的逐像素变化，不能自然等同于“不可感知”。
- Cautious 模式的最新任务准确率经常比 clean 低 20-30 个百分点，现实监控下仍可能暴露。
- 主实验常污染整个最终任务；低注入率只在消融中展示。
- 每个旧任务反演 10000 步、投毒优化默认 5000 个 epoch，并为每个当前样本维护独立扰动，成本高。
- 官方仓库 README 主要覆盖 EWC、RWalk、MAS、AFEC-EWC，没有正式版 ANCL、ER、ER-ACE 的完整代码路径。
- 未评估现代 poison detection、可信验证集、梯度异常检测或 task rollback。
- 视觉分类结果不能直接外推到 EEG、LLM 或无标签在线适应。

综合判断：内部有效性中等偏强；现实隐蔽性、外部有效性和完整复现性中等或偏低。

## 第六部分 与已有工作的比较

| Work | Key Idea | Difference From BrainWash | Limitation |
| --- | --- | --- | --- |
| Label Flipping CL Attack | 把旧任务样本改成错误标签后注入未来任务 | BrainWash 最终不改标签，也不需要旧数据 | 易检测，需旧任务样本 |
| PACOL | 匹配旧任务错误标签梯度 | BrainWash 直接优化训练后的旧任务代理损失，并默认攻击全部旧任务 | PACOL 仍需旧数据或辅助旧分布 |
| Poisoning Generative Replay | 污染 generator/replay 产生的旧样本 | BrainWash 直接污染最新任务输入，可攻击正则化与 ER | 依赖生成式 replay 结构 |
| MetaPoison | 截断双层 clean-label poisoning | BrainWash 把外层目标改为 CL 旧任务遗忘，并加入模型反演 | 原方法不针对跨任务遗忘 |
| Unlearnable Examples | 让当前数据难以学习 | BrainWash 重点破坏旧知识，不只降低当前任务可学性 | 常显著损害当前任务，易被监控 |
| STP 2025 | 仅访问当前任务，用普通 corruption 攻击整个 CL 过程 | 权限比 BrainWash 更弱，无需白盒 | 目标控制较粗，未直接重建旧任务 |

## 第七部分 局限性

### 作者提到的局限

- 攻击强度、注入比例和可见性之间存在折中。
- replay 场景依赖攻击者能否访问 memory，威胁模型更加多样。
- 较高 CL 正则强度会略微降低攻击，但同时损害模型可塑性，不能作为理想防御。

### 作者没有充分展开但值得注意的局限

1. 白盒模型与即将到来的完整有标签任务数据并非轻量权限。
2. Multi-head 模型暴露了任务和类别结构，模型反演在 single-head、未知任务边界下更困难。
3. 逐样本扰动依赖固定训练数据，难以直接迁移到攻击者尚未看到的未来样本。
4. 模型反演代理集可能主要捕获模型高置信模式，而非真实数据分布；其成功不等同于隐私意义上的精确重建。
5. 一步内层训练忽略真实 CL 正则、优化器状态、数据增强和完整 epoch，攻击迁移缺少理论保证。
6. 正式版结论扩展到 memory-based CL，但 replay 实验仅 ER/ER-ACE、单数据集和一个 buffer 大小。
7. 没有系统比较攻击生成时间、显存、代理样本量与攻击收益。

### 未来仍未解决的问题

- 无标签、半监督、自监督和 test-time adaptation 中如何定义可微的“当前任务学习”内层。
- 不读取模型参数的黑盒 BrainWash。
- 对 task boundary、head 结构和旧类别数均未知的攻击。
- 用可信 anchor、task vector、梯度几何或参数回滚检测恶意遗忘。
- 对 EEG 等物理信号采用语义和生理约束，而不是直接照搬像素 \(\ell_\infty\)。

## 第八部分 最新发展

截至日期：2026-07-20

| 方向/工作 | 最新状态 | 与本文关系 | 证据来源 |
| --- | --- | --- | --- |
| BrainWash 官方实现 | 代码与数据仍公开；代码覆盖少于正式论文 | 可复现核心 MI + 双层攻击 | [GitHub](https://github.com/mint-vu/Brainwash)、[Hugging Face](https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main) |
| Single-Task Poisoning | CoLLAs 2025，提出无模型、无过去/未来任务访问的更弱权限攻击及 task-vector 检测 | 挑战 BrainWash 的高权限假设 | [arXiv:2507.04106](https://arxiv.org/abs/2507.04106) |
| CL poisoning theory | 2026 预印本，给出正则化 CL 中攻击/防御边界和 task-to-task verification | 为 BrainWash 暴露的经验风险提供理论框架 | [arXiv:2606.29841](https://arxiv.org/abs/2606.29841) |
| Amnesia | 2026 预印本，只操控 replay index selection | 从像素投毒扩展到 replay 调度攻击 | [arXiv:2606.12655](https://arxiv.org/abs/2606.12655) |

工业/开源采用情况：没有发现主流工业 CL 系统公开采用 BrainWash；其主要价值仍是安全评测与防御研究。

Open Problems：统一攻击 benchmark、现实权限分级、任务级检测、跨模态扰动约束、攻击后的可恢复性和可信回滚。

## 第九部分 科研价值

BrainWash 的持久价值不在于某个具体噪声，而在于把“遗忘”从 CL 的自然故障转化为可优化的攻击目标。它还把模型反演、双层投毒和 CL 稳定性机制连接起来，适合作为 BCI 持续适应安全研究的起点。

### 迁移到 EEG/BCI 时的合理先验

图像 TV 与像素范数不能直接代表 EEG 合法性。可考虑：

\[
R_{\mathrm{EEG}}(x)=
\lambda_aR_{\mathrm{amp}}
+\lambda_t\|\Delta_t x\|_2^2
+\lambda_f\|\log\mathrm{PSD}(x)-\bar p\|_2^2
+\lambda_c d_{\mathrm{SPD}}(\mathrm{Cov}(x),\bar\Sigma)^2
+\lambda_g\sum_t x_{:,t}^{\top}L_Gx_{:,t}.
\]

| 约束 | 含义 |
| --- | --- |
| \(R_{\mathrm{amp}}\) | 每通道幅值、标准差、峰峰值和设备量程 |
| 时间平滑 | 限制不自然的高频突变 |
| PSD/频带 | 匹配 delta/theta/alpha/beta/gamma 等频带结构 |
| 协方差 SPD 距离 | 保持跨通道空间统计 |
| 电极图 Laplacian | 保持邻近电极的空间平滑与头皮拓扑 |

这些是迁移建议，不是 BrainWash 原文结论。对 BrainUICL，还需解决无标签内层：可以使用高置信伪标签、teacher consistency、entropy minimization 或其真实自监督适应损失，但必须同时评估伪标签误差是否已经在无攻击时引起遗忘。

如果继续写下一篇论文：

| 方向 | 具体问题 | 为什么有价值 | 难度/资源 | 可能做法 |
| --- | --- | --- | --- | --- |
| EEG-UICL BrainWash | 无标签个体流中能否诱导跨受试者遗忘 | 与实际 BCI 适应直接相关 | 中 | 对 BrainUICL 的伪标签/一致性损失做一阶双层攻击 |
| 生理可行扰动 | \(\ell_\infty\) 是否对应真实 EEG 攻击 | 提高物理与临床可信度 | 中 | SNR、PSD、协方差、电极拓扑联合约束 |
| Task-level Detection | 单个恶意受试者/session 能否被识别 | 可形成防御论文 | 中 | task vector、Fisher drift、BN 统计变化、可信 anchor |
| 黑盒迁移 | 不访问 victim 参数时能否攻击 | 权限更现实 | 高 | 多架构代理、ensemble gradient、查询受限估计 |
| 恢复与回滚 | 检测后如何恢复旧知识 | 比单纯检测更实用 | 高 | checkpoint selection、parameter delta removal、trusted replay |

## 第十部分 Roadmap

### 推荐阅读顺序

| Order | Paper/Topic | Why Read It |
| ---: | --- | --- |
| 1 | EWC、MAS、ER/ER-ACE | 理解受害者如何保护旧知识 |
| 2 | Targeted Forgetting / False Memory 2020 | 理解 CL 恶意遗忘问题的早期形式 |
| 3 | 2022 Targeted Data Poisoning | 理解旧数据和错误标签攻击 |
| 4 | PACOL | 理解 gradient matching clean-label 构造 |
| 5 | BrainWash | 理解 MI + 双层全局遗忘 |
| 6 | STP 2025 | 理解更弱权限的 task-level 攻击与检测 |
| 7 | Theory of CL Against Data Poisoning 2026 | 理解可防边界与理论化防御 |

### 知识树

```text
Continual Learning Security
├── Victim mechanisms
│   ├── Regularization: EWC / MAS / RWALK / AFEC / ANCL
│   └── Replay: ER / ER-ACE / DGR
├── Forgetting attacks
│   ├── Trigger / false memory
│   ├── Label flipping
│   ├── Gradient matching: PACOL
│   ├── Bilevel + inversion: BrainWash
│   └── Task/sampler attacks: STP / Amnesia
└── Defenses
    ├── Sample filtering
    ├── Gradient/task-vector verification
    ├── Trusted anchors
    └── Rollback and recovery
```

### 时间线

```text
自然遗忘防护
-> 恶意后门/错误记忆
-> 定向数据投毒
-> clean-label 梯度匹配
-> 模型反演 + 双层遗忘
-> 弱权限 task/sampler 攻击
-> 理论防御与恢复
```

### 后续论文/主题

- BCI 中的无标签持续适应投毒。
- replay buffer 的写入、采样与可信审计。
- 基于 task vector/Fisher/BN 统计的 poisoned-session 检测。
- 生理约束下的 EEG model inversion 与 clean-label poison。
- CL 攻击后的 unlearning、rollback 和知识恢复。

## Sources

- 本地论文：[用户 Markdown](</home/undefined/Desktop/IPhone/论文/2024CVPR-BrainWash A Poisoning Attack to Forget in Continual Learning.md>)；[工作区 CVPR PDF](</home/undefined/Desktop/bci/papers/CL_Attacks/2024CVPR-BrainWash - A Poisoning Attack to Forget in Continual Learning.pdf>)
- 官方论文页：[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.html)、[arXiv:2311.11995](https://arxiv.org/abs/2311.11995)、[DOI](https://doi.org/10.1109/CVPR52733.2024.02271)
- 官方代码/项目：[GitHub](https://github.com/mint-vu/Brainwash)、[Hugging Face assets](https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main)
- 其他来源：[PACOL](https://arxiv.org/abs/2311.10919)、[STP](https://arxiv.org/abs/2507.04106)、[2026 theory](https://arxiv.org/abs/2606.29841)、[Amnesia](https://arxiv.org/abs/2606.12655)
