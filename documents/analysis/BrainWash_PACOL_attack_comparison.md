# BrainWash 与 PACOL 攻击构造对比

## Papers

| Paper | Year | Venue | Main Problem | Core Idea |
| --- | --- | --- | --- | --- |
| PACOL: Poisoning Attacks Against Continual Learners | 2023 | arXiv | 在未来任务中注入少量 clean-label 样本，定向遗忘一个旧任务 | 让未来任务样本梯度匹配旧任务错误标签梯度 |
| BrainWash: A Poisoning Attack to Forget in Continual Learning | 2024 | CVPR | 不访问旧数据，只污染最新任务，使模型广泛遗忘旧任务 | 模型反演旧知识，再用双层优化直接最大化训练后的旧任务损失 |

最简洁的区别是：

- PACOL 先找一个已知有害的参考梯度，再让新任务毒样本模仿它。
- BrainWash 先重建旧任务代理数据，再直接询问“哪种新任务扰动会让模型更新后在旧数据上最差”。

两者最终注入的样本都保持当前/未来任务原标签，不是把错误标签直接交给受害者。

## Evolution

```text
Label flipping
旧任务图像 + 错误标签，攻击强但显眼
        |
        v
PACOL
只把错误标签用于生成恶意参考梯度；真正注入未来任务 clean-label 样本
        |
        +-------------------------+
                                  v
BrainWash                      更弱权限方向
模型反演替代旧数据，           STP：只访问当前 task
双层目标直接优化全局遗忘       Amnesia：只控制 replay sampler
```

PACOL 与 BrainWash 是同期工作：PACOL 于 2023-11-18 提交，BrainWash 于 2023-11-20 提交。BrainWash 引用了 PACOL 作者的 2022 IJCNN 前驱，但没有引用两天前提交的 PACOL 预印本，因此更准确的说法是“相邻研究线”，而不是明确的直接改进关系。

## 共同数学视角

设模型当前参数为 \(\theta\)，新任务毒样本产生训练梯度 \(g_{\mathrm{new}}(\delta)\)，一次 SGD 后：

\[
\theta'=\theta-\alpha g_{\mathrm{new}}(\delta).
\]

旧任务损失的一阶展开是：

\[
L_{\mathrm{old}}(\theta')
\approx
L_{\mathrm{old}}(\theta)
-\alpha
\nabla_\theta L_{\mathrm{old}}(\theta)^\top
g_{\mathrm{new}}(\delta).
\]

若想让旧任务损失增加，就要使：

\[
\nabla_\theta L_{\mathrm{old}}^\top
g_{\mathrm{new}}(\delta)<0.
\]

这说明两篇论文做的是同一件抽象事情：把一个破坏旧知识的更新方向编码进新任务样本。

- PACOL 用旧任务错误标签梯度 \(g_{\mathrm{lf}}\) 作为这个方向的代理，并显式要求 \(g_{\mathrm{new}}\approx g_{\mathrm{lf}}\)。
- BrainWash 不预先规定参考梯度，而是通过“更新后旧损失最大”的元目标自动寻找该方向。

## PACOL 如何构造攻击

### 输入条件

White-box PACOL 需要：受害模型参数、未来任务数据、目标旧任务真实数据。Gray/black-box 不拿 victim 参数，但仍需要目标旧任务同分布辅助数据来训练 surrogate。

### 第一步：构造 label-flip 参考梯度

对希望遗忘的旧任务 \(\mathcal D_\tau=(X_\tau,Y_\tau)\)，生成错误标签 \(Y_\tau^{adv}\)，计算：

\[
g_{\mathrm{lf}}
=\nabla_\theta
\mathcal L(f(X_\tau;\theta_k),Y_\tau^{adv}).
\]

这代表“如果模型真的在旧任务错误标签上训练，会朝哪个方向破坏旧知识”。

### 第二步：选择未来任务载体

从未来非目标任务选择少量正常样本：

\[
(X_{\tau+n},Y_{\tau+n}).
\]

标签始终保持 \(Y_{\tau+n}\)，只优化输入：

\[
X_{\tau+n}^{adv}=X_{\tau+n}+\delta,
\qquad \|\delta\|_\infty\le\epsilon.
\]

### 第三步：匹配参数梯度

投毒样本当前产生：

\[
g_{\mathrm{adv}}
=\nabla_\theta
\mathcal L(f(X_{\tau+n}^{adv};\theta_k),Y_{\tau+n}).
\]

优化目标：

\[
\min_{X_{\tau+n}^{adv}}
d(g_{\mathrm{adv}},g_{\mathrm{lf}}),
\]

其中 \(d\) 是平方 \(\ell_2\) 或负余弦相似度。对输入求导需要 \(\nabla^2_{X,\theta}\mathcal L\) 的混合二阶导。

### 第四步：覆盖训练轨迹

PACOL 在 \(\theta_k\) 上对输入做 \(S\) 次 PGD，然后用 poison 更新一次模型得到 \(\theta_{k+1}\)，共进行 \(K\) 轮。这样生成的 poison 不只适用于初始 checkpoint，也尽量适用于后续训练状态。

### 第五步：注入

最终向未来任务注入约 1%-5% 的：

\[
(X_{\tau+n}^{adv},Y_{\tau+n}).
\]

旧任务错误标签从不进入受害训练流。

### PACOL 复现警告

正文要求最小化梯度距离，但原始伪代码使用：

\[
X^{adv}\leftarrow
X^{adv}+\alpha\operatorname{sign}(\nabla_X d),
\]

即梯度上升。按正文通常应为减号；同时余弦展开式漏写平方。由于缺少完整 PACOL 官方代码，复现时必须先明确优化符号。

## BrainWash 如何构造攻击

### 输入条件

攻击者需要：训练到任务 \(T-1\) 的白盒模型，以及最新任务 \(D_T\) 的完整有标签数据。攻击者不知道受害者具体采用 EWC、MAS、ER 等哪种 CL 算法，也没有旧任务真实数据。

### 第一步：模型反演旧任务代理集

对每个旧任务随机选择目标类 \(\hat y_t\)，从随机输入开始，在冻结模型上优化：

\[
\hat X_t
=\arg\min_X
\operatorname{CE}(h_t(f(X;\theta^*)),\hat Y_t)
+R_{\mathrm{TV}}(X)
+R_{\ell_2}(X)
+R_{\mathrm{BN}}(X).
\]

得到：

\[
\hat D_t=\{(\hat x_t^j,\hat y_t^j)\}_{j=1}^{M}.
\]

这些样本只供攻击者衡量旧知识，不交给受害者。随机目标类不是错误标签，而是反演时希望合成输入激活的旧类。

### 第二步：为最新任务设置逐样本噪声

\[
\widetilde D_T(\delta)
=\{(x_T^i+\delta^i,y_T^i)\},
\qquad \|\delta^i\|_\infty\le\epsilon.
\]

标签保持正确。

### 第三步：内层模拟一次学习

\[
w^{(1)}(\delta)
=w^{(0)}-alpha
\nabla_wL_T(w^{(0)},\delta),
\]

其中 \(w=(\theta,\psi_T)\)。攻击者只模拟普通 fine-tuning，不加入受害者实际 CL 正则，因此不需要知道 CL 算法。

### 第四步：外层直接优化遗忘

Reckless：

\[
\max_\delta
\sum_{t<T}\sum_j
\mathcal L(\hat x_t^j,\hat y_t^j;w^{(1)}(\delta)).
\]

Cautious：

\[
\max_\delta
L_{\mathrm{old}}(w^{(1)}(\delta))
-\eta L_T^{\mathrm{clean}}(w^{(1)}(\delta)).
\]

前者只追求旧任务遗忘；后者同时保持最新任务干净性能。

### 第五步：元梯度更新扰动

\[
\nabla_\delta J
=-\alpha
(\nabla^2_{\delta,w}L_T)^\top
\nabla_wJ.
\]

反复更新 \(\delta\) 并投影回 \(\ell_\infty\) 范数球。最终受害者用自己的 CL 方法训练这些投毒样本，共享模型参数被推向遗忘旧任务的方向。

### Replay 情况

CVPR 正式版还测试 ER/ER-ACE：

- 无 memory 读权限时，用模型反演旧知识。
- 有只读 memory 权限时，直接用 buffer 样本作为旧目标，攻击更强。
- 不考虑写 buffer。

攻击核心仍是参数更新，不是直接改 buffer。

## Method Comparison

| Dimension | PACOL | BrainWash |
| --- | --- | --- |
| 遗忘目标 | 选择一个指定旧任务 | 默认同时破坏所有旧任务 |
| 核心构造 | 匹配旧任务错误标签梯度 | 最大化一次模拟训练后的旧任务代理损失 |
| 旧任务信息 | 白盒需真实旧数据；gray/black 需辅助旧分布 | 无旧数据时使用模型反演；replay 场景也可读 memory |
| 模型权限 | White/gray/surrogate black-box | 白盒模型 |
| CL 算法知识 | 不需要 importance/generator 细节 | 不知道具体 CL 算法和超参数 |
| 当前/未来数据 | 需要未来任务有标签数据 | 需要最新任务有标签数据 |
| 最终 poison 标签 | 正确标签 | 正确标签 |
| 错误标签的作用 | 只生成参考梯度 | 不使用错误标签；随机类用于模型反演 |
| 求解方式 | Gradient matching + PGD + 多模型状态 | Model inversion + truncated bilevel meta-gradient |
| 主要注入比例 | 1%-5% | 主实验通常污染最终任务；另做部分注入消融 |
| 当前任务隐蔽性 | 靠小扰动、正确标签和低比例 | Cautious 目标显式保护当前任务准确率 |
| 被攻击 CL | EWC、Online EWC、SI、DGR | EWC、MAS、RWALK、AFEC、ANCL、ER、ER-ACE |
| 攻击后主要改变 | 共享模型参数；DGR generator 可能间接受污染 | 共享模型参数；不直接写 buffer |
| 计算代价 | \(K\times S\) 次二阶梯度匹配 | 每旧任务长时间 MI + 数千轮噪声元优化 |
| 主要复现风险 | 伪代码符号冲突、无完整代码 | 官方代码未覆盖正式论文全部实验 |

## Experiment Comparison

| Dimension | PACOL | BrainWash |
| --- | --- | --- |
| 数据集 | R/S-MNIST、S-SVHN、S-CIFAR10 | CIFAR-100、miniImageNet、tinyImageNet |
| 任务规模 | 5 tasks 为主 | 10 或 20 tasks |
| 模型 | MLP、CNN、ResNet20、VGG surrogate、WGAN-GP | ResNet-18，补充 RegNetX |
| 主要攻击预算 | 1%-5%，\(16/255\) 或 \(25.5/255\) | \(\epsilon=0.1,0.3\)，并消融注入率 |
| 遗忘指标 | 各任务最终准确率 | BWT + 最新任务准确率 |
| 防御实验 | 基础离群检测 | 未系统评估防御 |
| 统计报告 | 正则化实验十次运行和区间 | 主表无均值方差 |
| 自然图像效果 | S-CIFAR 上较弱 | 在三个较大 benchmark 上明显，但高预算也明显损害当前任务 |
| 可复现性 | 论文细节不足，无完整 PACOL repo | 核心代码与资产公开，但覆盖不完整 |

两篇结果不能直接比较“谁更强”，因为目标、数据集、投毒比例、噪声预算和受害模型均不同。

## 对文档中几个关键疑问的直接结论

### 攻击针对无标签数据吗？

不是。两篇原论文都使用监督分类损失，需要当前/未来任务标签。PACOL 还需要目标旧任务标签来构造错误标签参考梯度。最终 poison 虽然是 clean-label，但 clean-label 的意思是“标签保持正确”，不是“没有标签”。

### 攻击是基于 CL 正则化吗？

不是直接基于。两篇推导都故意不依赖受害者的 EWC/MAS/SI 正则细节：

- PACOL 在推导中先忽略正则和 replay，再测试恶意梯度能否跨方法迁移。
- BrainWash 内层只模拟普通 fine-tuning，不加入 victim 的 CL 正则。

两篇利用的是共享参数仍要吸收新任务梯度这一共同弱点。

### 攻击后影响 buffer 还是模型参数？

核心影响是模型参数。PACOL 在 DGR 中可能间接污染 generator；BrainWash 正式版的 ER/ER-ACE 只读 memory，不写 memory。若真实系统把毒化当前样本存入 buffer，会产生额外持久化效应，但这不是两篇的统一核心设定。

### BrainWash 为什么要先 MI？

因为它的外层必须知道“旧任务是否变差”。没有旧数据时，模型反演提供一个可微的旧任务代理验证集。代理集不是给受害者训练，而是作为攻击优化的评分函数。

## Which One To Build On

### 研究“定向遗忘某个旧任务”

优先以 PACOL 为基线。它的攻击目标更精确，而且已有 gray/black-box surrogate 设定。需要先修复伪代码符号并明确完整实现。

### 研究“攻击者没有旧数据”

优先以 BrainWash 为基线。模型反演正是为去除旧数据访问而设计，但需承认白盒、计算成本和多头信息泄露假设。

### 研究 BrainUICL / EEG 无标签个体持续适应

最值得做的不是直接复制任一论文，而是组合两者：

1. 用模型反演、prototype inversion 或可信 source model 统计构造旧知识代理。
2. 若目标是指定旧 subject/class，用 PACOL 式目标构造恶意参考梯度。
3. 若目标是全局稳定性破坏，用 BrainWash 式更新后旧损失目标。
4. 把监督 CE 内层替换成 BrainUICL 实际采用的伪标签、teacher consistency、DCB replay 与 CEA 对齐损失。
5. 用 EEG 物理约束替换纯像素 \(\ell_\infty\)：SNR、PSD、频带功率、时间平滑、通道协方差和电极拓扑。
6. 同时报告旧 subject/class 遗忘、当前 subject 适应、未见 subject 泛化和毒样本可检测性。

一个可研究的混合目标是：

\[
\max_\delta
L_{\mathrm{old-proxy}}(\theta'(\delta))
-\eta L_{\mathrm{current-unsup}}(\theta'(\delta))
-\gamma R_{\mathrm{EEG}}(X+\delta),
\]

其中：

\[
\theta'(\delta)
=\theta-\alpha
\nabla_\theta L_{\mathrm{BrainUICL}}(X+\delta;\theta).
\]

这才是对 BrainUICL 场景的合理改写；原论文结论不能直接声称已经覆盖无标签 EEG。

## Future Directions

| 方向 | 研究问题 | 可验证假设 |
| --- | --- | --- |
| Hybrid MI-PACOL | 反演旧任务后能否生成定向错误标签梯度 | 无真实旧数据仍可选择性遗忘某类/某 subject |
| Unsupervised BrainWash | 用熵最小化或一致性损失作为内层是否仍可元投毒 | 无标签适应同样存在恶意遗忘方向 |
| EEG semantic constraints | 生理约束是否显著降低攻击但提高隐蔽性 | \(\ell_\infty\) 不是 EEG 中最合理预算 |
| Task-vector defense | 恶意 session 更新能否通过方向、范数或 Fisher 漂移识别 | task-level 检测优于逐样本离群检测 |
| Trusted anchor | 少量可信旧数据能否阻断两类攻击 | 外层旧损失监控可触发回滚或更新裁剪 |
| Recovery | 检测后如何去除恶意参数增量 | checkpoint/task-vector subtraction 可恢复旧知识 |

## Sources

- [PACOL arXiv](https://arxiv.org/abs/2311.10919)
- [PACOL 2022 前驱代码](https://github.com/HuayuLiArizona/Data-Poison-Incremental-Learning)
- [BrainWash CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.html)
- [BrainWash 官方代码](https://github.com/mint-vu/Brainwash)
- [BrainWash 官方资产](https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main)
- [STP 2025](https://arxiv.org/abs/2507.04106)
- [Theory of CL Against Data Poisoning 2026](https://arxiv.org/abs/2606.29841)
- [Amnesia 2026](https://arxiv.org/abs/2606.12655)
